"""Machine Models page — admin CRUD over PLC-code-tagged settings profiles.

A profile is a snapshot of, per camera: the tunable fields (ROI, exposure,
gain, gamma, brightness, resolution, trigger mode), the full detection
strategy configuration (its own algorithm + parameters), and the active
pixel-to-mm calibration (scale, homography, lens distortion) — captured from
whatever is *currently* live via the Camera/Detection/Calibration pages;
there is no separate settings editor here by design. "New from Current" /
"Update Selected from Current" do the capturing; "Apply Now" pushes a
profile live without persisting camera.json/detection.json or writing a new
calibration-history row (the same live-preview path the PLC-driven
auto-switch in main.py's ``Application._on_machine_model_changed`` uses).

The right-hand panel is a tree, one top-level branch per camera, each with
"Camera", "Detection" and "Calibration" sub-branches of individual
property/value rows — deliberately structured rather than a hand-formatted
text dump, since a profile now carries three independent settings domains
per camera and a flat paragraph stopped scaling once detection and
calibration became per-camera too.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.utilities.exceptions import VisionSystemError
from models.app_state import AppState
from services.auth_service import AuthService
from services.machine_model_service import MachineModelService


def _code_spin() -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(0, 65535)
    return spin


class MachineModelsPage(QWidget):
    """List/capture/apply machine-model profiles."""

    def __init__(
        self,
        machine_model_service: MachineModelService,
        auth_service: AuthService,
        app_state: AppState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._svc = machine_model_service
        self._auth = auth_service
        self._app_state = app_state
        self._row_ids: list[int] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        title = QLabel("Machine Models")
        title.setProperty("class", "pageTitle")
        root.addWidget(title)

        name, code = app_state.active_machine_model
        self._active_label = QLabel(self._active_text(name, code))
        self._active_label.setProperty("class", "dim")
        root.addWidget(self._active_label)

        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, stretch=1)

        # ------------------------------------------------------ left column
        left_box = QGroupBox("Profiles")
        left_box.setFixedWidth(340)
        left = QVBoxLayout(left_box)
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_select)
        left.addWidget(self._list, stretch=1)

        form = QFormLayout()
        self._name = QLineEdit()
        self._code = _code_spin()
        form.addRow("Name", self._name)
        form.addRow("PLC Code", self._code)
        left.addLayout(form)

        new_btn = QPushButton("New from Current")
        new_btn.setProperty("class", "primary")
        new_btn.setToolTip(
            "Capture the camera ROI/exposure + detection settings currently "
            "live (Camera/Detection pages) as a new profile under this "
            "name and PLC code"
        )
        new_btn.clicked.connect(self._on_new_from_current)
        update_btn = QPushButton("Update Selected from Current")
        update_btn.setToolTip("Re-capture current settings into the selected profile")
        update_btn.clicked.connect(self._on_update_from_current)
        rename_btn = QPushButton("Rename / Change Code")
        rename_btn.clicked.connect(self._on_rename)
        apply_btn = QPushButton("Apply Now")
        apply_btn.setToolTip("Push the selected profile live, without waiting for the PLC")
        apply_btn.clicked.connect(self._on_apply_now)
        delete_btn = QPushButton("Delete")
        delete_btn.setProperty("class", "danger")
        delete_btn.clicked.connect(self._on_delete)
        for button in (new_btn, update_btn, rename_btn, apply_btn, delete_btn):
            left.addWidget(button)

        hint = QLabel(
            "A profile stores, per camera: the part-dependent camera fields "
            "(ROI, exposure, gain, resolution, trigger), the camera's own "
            "detection strategy + parameters, and its active pixel-to-mm "
            "calibration — hardware wiring (driver, connection) is never "
            "touched by applying one."
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "dim")
        left.addWidget(hint)
        body.addWidget(left_box)

        # ----------------------------------------------------- right column
        right_box = QGroupBox("Selected Profile")
        right = QVBoxLayout(right_box)
        self._meta_label = QLabel("—")
        self._meta_label.setWordWrap(True)
        right.addWidget(self._meta_label)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Property", "Value"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        right.addWidget(self._tree, stretch=1)
        self._status = QLabel("—")
        self._status.setProperty("class", "dim")
        self._status.setWordWrap(True)
        right.addWidget(self._status)
        body.addWidget(right_box, stretch=1)

        # ---------------------------------------------------------- wiring
        app_state.active_machine_model_changed.connect(self._on_active_model_changed)
        self._reload()

    # -------------------------------------------------------------- loading
    def _reload(self) -> None:
        selected = self._current_id()
        self._list.clear()
        self._row_ids = []
        for profile in self._svc.list_profiles():
            self._row_ids.append(int(profile["id"]))
            self._list.addItem(f"{profile['plc_code']}: {profile['name']}")
        if self._row_ids:
            row = self._row_ids.index(selected) if selected in self._row_ids else 0
            self._list.setCurrentRow(row)
        else:
            self._meta_label.setText("No machine model profiles yet.")
            self._tree.clear()

    def _current_id(self) -> int | None:
        row = self._list.currentRow()
        return self._row_ids[row] if 0 <= row < len(self._row_ids) else None

    def _current_profile(self) -> dict | None:
        profile_id = self._current_id()
        return None if profile_id is None else self._svc.get_by_id(profile_id)

    def _select_id(self, profile_id: int) -> None:
        if profile_id in self._row_ids:
            self._list.setCurrentRow(self._row_ids.index(profile_id))

    def _on_select(self, _row: int) -> None:
        profile = self._current_profile()
        if profile is None:
            self._meta_label.setText("—")
            self._tree.clear()
            return
        self._name.setText(profile["name"])
        self._code.setValue(int(profile["plc_code"]))
        self._populate_profile(profile)

    # -------------------------------------------------------------- actions
    def _on_new_from_current(self) -> None:
        try:
            profile = self._svc.capture_current(
                self._name.text().strip(), self._code.value(), created_by=self._username()
            )
        except VisionSystemError as exc:
            QMessageBox.warning(self, "New Machine Model", str(exc))
            return
        self._reload()
        self._select_id(profile["id"])
        self._status.setText(
            f"Captured {profile['name']!r} (code {profile['plc_code']}) from current settings."
        )

    def _on_update_from_current(self) -> None:
        profile_id = self._current_id()
        if profile_id is None:
            return
        try:
            profile = self._svc.update_from_current(profile_id, updated_by=self._username())
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Update Machine Model", str(exc))
            return
        self._reload()
        self._select_id(profile["id"])
        self._status.setText(f"{profile['name']!r} updated from current settings.")

    def _on_rename(self) -> None:
        profile_id = self._current_id()
        if profile_id is None:
            return
        try:
            self._svc.rename(profile_id, self._name.text().strip(), self._code.value())
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Rename", str(exc))
            return
        self._reload()
        self._select_id(profile_id)

    def _on_delete(self) -> None:
        profile = self._current_profile()
        if profile is None:
            return
        answer = QMessageBox.question(
            self, "Delete Machine Model", f"Delete machine model {profile['name']!r}?"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._svc.delete(int(profile["id"]))
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Delete", str(exc))
            return
        self._reload()

    def _on_apply_now(self) -> None:
        profile = self._current_profile()
        if profile is None:
            return
        try:
            warnings = self._svc.apply_profile(profile)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Apply Now", str(exc))
            return
        self._app_state.set_active_machine_model(profile["name"], int(profile["plc_code"]))
        text = f"Applied {profile['name']!r} (code {profile['plc_code']})."
        if warnings:
            text += " " + "; ".join(warnings)
        self._status.setText(text)

    # --------------------------------------------------------------- state
    def _on_active_model_changed(self, name: str, code: int) -> None:
        self._active_label.setText(self._active_text(name, code))

    @staticmethod
    def _active_text(name: str, code: int | None) -> str:
        return f"Currently active: {name} (code {code})" if name else "No machine model applied yet"

    def _username(self) -> str:
        user = self._auth.current_user
        return user.username if user is not None else ""

    # -------------------------------------------------------------- summary
    def _populate_profile(self, profile: dict) -> None:
        """Fill the right-hand tree: one top-level branch per camera, each
        with "Camera", "Detection" and "Calibration" sub-branches — replaces
        the old flat-text dump now that all three settings domains are
        per-camera (see module docstring)."""
        self._meta_label.setText(
            f"{profile['name']}  (PLC code {profile['plc_code']})\n"
            f"created by {profile.get('created_by', '?')} at {profile.get('created_at', '?')}"
            f"  ·  updated at {profile.get('updated_at', '?')}"
        )

        self._tree.clear()
        cameras = profile.get("cameras", {})
        positions = profile.get("jog_positions", {})
        calibrations = profile.get("calibration", {})
        detection = profile.get("detection", {})
        detection_by_camera = "cameras" in detection

        indices = sorted(
            {*cameras, *positions, *calibrations, *(detection.get("cameras", {}) if detection_by_camera else [])},
            key=int,
        )
        if not indices:
            self._tree.addTopLevelItem(QTreeWidgetItem(["(no camera data captured)", ""]))
            return

        for index in indices:
            camera_item = QTreeWidgetItem([f"Camera {index}", ""])
            font = camera_item.font(0)
            font.setBold(True)
            camera_item.setFont(0, font)
            self._tree.addTopLevelItem(camera_item)

            self._add_camera_branch(camera_item, cameras.get(index, {}), positions.get(index))
            self._add_detection_branch(
                camera_item,
                detection["cameras"].get(index, {}) if detection_by_camera else detection,
            )
            self._add_calibration_branch(camera_item, calibrations.get(index))
            camera_item.setExpanded(True)

        self._tree.expandAll()

    @staticmethod
    def _leaf(parent: QTreeWidgetItem, label: str, value: object) -> None:
        parent.addChild(QTreeWidgetItem([label, str(value)]))

    def _add_camera_branch(
        self, camera_item: QTreeWidgetItem, cam: dict, position: dict | None
    ) -> None:
        branch = QTreeWidgetItem(["Camera Settings", ""])
        camera_item.addChild(branch)
        if not cam:
            self._leaf(branch, "(not captured)", "")
        else:
            roi = cam.get("roi", {})
            self._leaf(
                branch, "ROI",
                f"({roi.get('x', 0)}, {roi.get('y', 0)}) "
                f"{roi.get('width', 0)}x{roi.get('height', 0)}",
            )
            self._leaf(branch, "Exposure", f"{cam.get('exposure_us', '—')} us")
            self._leaf(branch, "Gain", f"{cam.get('gain_db', '—')} dB")
            self._leaf(branch, "Gamma", cam.get("gamma", "—"))
            self._leaf(branch, "Light Brightness", cam.get("brightness", "—"))
            self._leaf(branch, "Resolution", f"{cam.get('width', '—')}x{cam.get('height', '—')}")
            self._leaf(branch, "Trigger Mode", cam.get("trigger_mode", "—"))
        if position:
            pos_branch = QTreeWidgetItem(["Capture Position", ""])
            camera_item.addChild(pos_branch)
            self._leaf(pos_branch, "X", position.get("x", 0))
            self._leaf(pos_branch, "Y", position.get("y", 0))
            self._leaf(pos_branch, "Z", position.get("z", 0))

    def _add_detection_branch(self, camera_item: QTreeWidgetItem, detection: dict) -> None:
        branch = QTreeWidgetItem(["Detection", ""])
        camera_item.addChild(branch)
        if not detection:
            self._leaf(branch, "(not captured)", "")
            return
        active = detection.get("active_detector", "—")
        common = detection.get("common", {})
        self._leaf(branch, "Active Strategy", active)
        self._leaf(branch, "Confidence Threshold", common.get("confidence_threshold", "—"))
        self._leaf(branch, "Expected Hole Count", common.get("expected_hole_count", "—"))
        self._leaf(branch, "Position Tolerance", f"{common.get('position_tolerance_mm', '—')} mm")

        params = detection.get(active, {})
        if isinstance(params, dict) and params:
            params_branch = QTreeWidgetItem([f"{active} Parameters", ""])
            branch.addChild(params_branch)
            for key, value in params.items():
                self._leaf(params_branch, key, value)

    def _add_calibration_branch(
        self, camera_item: QTreeWidgetItem, calibration: dict | None
    ) -> None:
        branch = QTreeWidgetItem(["Calibration", ""])
        camera_item.addChild(branch)
        if not calibration:
            self._leaf(branch, "(uncalibrated — identity fallback)", "")
            return
        self._leaf(
            branch, "Pixels per mm",
            f"{calibration.get('pixels_per_mm_x', '—')} x {calibration.get('pixels_per_mm_y', '—')}",
        )
        self._leaf(branch, "Homography", "set" if calibration.get("homography") else "not set")
        has_lens = bool(calibration.get("camera_matrix") and calibration.get("dist_coeffs"))
        self._leaf(branch, "Lens Distortion Correction", "set" if has_lens else "not set")
        ref = calibration.get("ref_point_mm", [0.0, 0.0])
        self._leaf(branch, "Reference Point", f"({ref[0]:.2f}, {ref[1]:.2f}) mm")
        self._leaf(branch, "RMS Error", f"{calibration.get('rms_error', 0.0):.3f} mm")
        if calibration.get("calibrated_by"):
            self._leaf(branch, "Calibrated By", calibration["calibrated_by"])

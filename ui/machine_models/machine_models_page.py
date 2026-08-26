"""Machine Models page — admin CRUD over PLC-code-tagged settings profiles.

A profile is a snapshot of the tunable per-camera fields (ROI, exposure,
gain, gamma, brightness, resolution, trigger mode) and the full detection
strategy configuration, captured from whatever is *currently* live via the
Camera/Detection pages — there is no separate settings editor here by
design. "New from Current" / "Update Selected from Current" do the
capturing; "Apply Now" pushes a profile live without touching camera.json/
detection.json (the same live-preview path the PLC-driven auto-switch in
main.py's ``Application._on_machine_model_changed`` uses).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
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
            "A profile only stores the part-dependent camera fields (ROI, "
            "exposure, gain, resolution, trigger) plus the full detection "
            "configuration — hardware wiring (driver, connection) is never "
            "touched by applying one."
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "dim")
        left.addWidget(hint)
        body.addWidget(left_box)

        # ----------------------------------------------------- right column
        right_box = QGroupBox("Selected Profile")
        right = QVBoxLayout(right_box)
        self._summary = QPlainTextEdit()
        self._summary.setReadOnly(True)
        right.addWidget(self._summary, stretch=1)
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
            self._summary.setPlainText("No machine model profiles yet.")

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
            self._summary.setPlainText("")
            return
        self._name.setText(profile["name"])
        self._code.setValue(int(profile["plc_code"]))
        self._summary.setPlainText(self._format_profile(profile))

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

    # ------------------------------------------------------------- summary
    @staticmethod
    def _format_profile(profile: dict) -> str:
        lines = [
            f"{profile['name']}  (PLC code {profile['plc_code']})",
            f"created by {profile.get('created_by', '?')} at {profile.get('created_at', '?')}",
            f"updated at {profile.get('updated_at', '?')}",
            "",
            "Cameras:",
        ]
        cameras = profile.get("cameras", {})
        for index in sorted(cameras, key=int):
            cam = cameras[index]
            roi = cam.get("roi", {})
            lines.append(
                f"  {index}: ROI ({roi.get('x', 0)}, {roi.get('y', 0)}, "
                f"{roi.get('width', 0)}x{roi.get('height', 0)})  "
                f"exposure {cam.get('exposure_us', '—')} us  "
                f"gain {cam.get('gain_db', '—')} dB  "
                f"{cam.get('width', '—')}x{cam.get('height', '—')}  "
                f"trigger {cam.get('trigger_mode', '—')}"
            )
        if not cameras:
            lines.append("  (none captured)")

        positions = profile.get("jog_positions", {})
        if positions:
            lines += ["", "Default Positions:"]
            for index in sorted(positions, key=int):
                pos = positions[index]
                lines.append(f"  {index}: ({pos.get('x', 0)}, {pos.get('y', 0)})")

        detection = profile.get("detection", {})
        active = detection.get("active_detector", "—")
        common = detection.get("common", {})
        lines += [
            "",
            "Detection:",
            f"  active strategy: {active}",
            f"  confidence >= {common.get('confidence_threshold', '—')}, "
            f"expected holes {common.get('expected_hole_count', '—')}, "
            f"tolerance {common.get('position_tolerance_mm', '—')} mm",
        ]
        params = detection.get(active, {})
        if isinstance(params, dict) and params:
            lines.append(f"  {active} params: " + ", ".join(f"{k}={v}" for k, v in params.items()))
        return "\n".join(lines)

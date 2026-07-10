"""Camera Configuration page.

Left: camera list + lifecycle buttons + health. Centre: parameter form
(exposure/gain/gamma/brightness/resolution/trigger/ROI). Right: live preview
on a :class:`RoiEditor` — "Draw ROI" lets the operator drag the region
directly on the image and the spin boxes follow.

"Apply Live" pushes settings to the connected device without persisting;
"Save" writes camera.json (+ DB mirror). Worker/manager rebuilds after a
save are handled by the composition root's config subscription.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.utilities.enums import CameraDriver, ConnectionState, TriggerMode
from core.utilities.exceptions import VisionSystemError
from models.app_state import AppState
from services.camera_service import CameraService
from ui.widgets import LabeledLed, RoiEditor


class CameraPage(QWidget):
    """Add / remove / tune / test cameras."""

    def __init__(
        self, app_state: AppState, camera_service: CameraService, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._svc = camera_service
        self._row_indexes: list[int] = []  # list row -> camera index
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        title = QLabel("Camera Configuration")
        title.setProperty("class", "pageTitle")
        root.addWidget(title)

        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, stretch=1)

        # ------------------------------------------------------ left column
        left = QVBoxLayout()
        self._list = QListWidget()
        self._list.setFixedWidth(230)
        self._list.currentRowChanged.connect(self._on_select)
        left.addWidget(self._list, stretch=1)

        row1 = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add)
        remove_btn = QPushButton("Remove")
        remove_btn.setProperty("class", "danger")
        remove_btn.clicked.connect(self._on_remove)
        row1.addWidget(add_btn)
        row1.addWidget(remove_btn)
        left.addLayout(row1)

        row2 = QHBoxLayout()
        connect_btn = QPushButton("Connect")
        connect_btn.clicked.connect(lambda: self._lifecycle("connect"))
        disconnect_btn = QPushButton("Disconnect")
        disconnect_btn.clicked.connect(lambda: self._lifecycle("disconnect"))
        row2.addWidget(connect_btn)
        row2.addWidget(disconnect_btn)
        left.addLayout(row2)

        test_btn = QPushButton("Test Camera")
        test_btn.clicked.connect(self._on_test)
        left.addWidget(test_btn)
        self._health = LabeledLed("no camera selected")
        left.addWidget(self._health)
        body.addLayout(left)

        # ------------------------------------------------------- form column
        form_box = QGroupBox("Parameters")
        form_box.setFixedWidth(330)
        form = QFormLayout(form_box)

        self._name = QLineEdit()
        self._driver = QComboBox()
        self._driver.addItems([d.value for d in CameraDriver])
        self._conn_id = QLineEdit()
        self._enabled = QCheckBox("Enabled")
        self._exposure = QSpinBox()
        self._exposure.setRange(10, 1_000_000)
        self._exposure.setSingleStep(500)
        self._exposure.setSuffix(" µs")
        self._gain = QDoubleSpinBox()
        self._gain.setRange(0.0, 48.0)
        self._gain.setSingleStep(0.5)
        self._gain.setSuffix(" dB")
        self._gamma = QDoubleSpinBox()
        self._gamma.setRange(0.1, 4.0)
        self._gamma.setSingleStep(0.05)
        self._brightness = QSpinBox()
        self._brightness.setRange(-100, 100)
        self._width = QSpinBox()
        self._width.setRange(64, 8192)
        self._height = QSpinBox()
        self._height.setRange(64, 8192)
        self._trigger = QComboBox()
        self._trigger.addItems([t.value for t in TriggerMode])

        form.addRow("Name", self._name)
        form.addRow("Driver", self._driver)
        form.addRow("Connection ID", self._conn_id)
        form.addRow("", self._enabled)
        form.addRow("Exposure", self._exposure)
        form.addRow("Gain", self._gain)
        form.addRow("Gamma", self._gamma)
        form.addRow("Brightness", self._brightness)
        form.addRow("Width", self._width)
        form.addRow("Height", self._height)
        form.addRow("Trigger Mode", self._trigger)

        # ROI block
        self._roi_spins = []
        roi_row = QHBoxLayout()
        for caption in ("X", "Y", "W", "H"):
            spin = QSpinBox()
            spin.setRange(0, 8192)
            spin.setToolTip(f"ROI {caption}")
            spin.valueChanged.connect(self._on_roi_spins_changed)
            self._roi_spins.append(spin)
            roi_row.addWidget(spin)
        roi_widget = QWidget()
        roi_widget.setLayout(roi_row)
        form.addRow("ROI x/y/w/h", roi_widget)

        roi_buttons = QHBoxLayout()
        self._draw_roi = QPushButton("Draw ROI")
        self._draw_roi.setCheckable(True)
        self._draw_roi.toggled.connect(self._on_draw_toggled)
        clear_roi = QPushButton("Clear ROI")
        clear_roi.clicked.connect(self._on_clear_roi)
        roi_buttons.addWidget(self._draw_roi)
        roi_buttons.addWidget(clear_roi)
        roi_buttons_w = QWidget()
        roi_buttons_w.setLayout(roi_buttons)
        form.addRow("", roi_buttons_w)

        apply_btn = QPushButton("Apply Live")
        apply_btn.clicked.connect(self._on_apply)
        save_btn = QPushButton("Save Configuration")
        save_btn.setProperty("class", "primary")
        save_btn.clicked.connect(self._on_save)
        form.addRow(apply_btn)
        form.addRow(save_btn)
        body.addWidget(form_box)

        # ---------------------------------------------------- preview column
        preview_col = QVBoxLayout()
        self._preview = RoiEditor()
        self._preview.setMinimumSize(480, 360)
        self._preview.roi_changed.connect(self._on_roi_drawn)
        preview_col.addWidget(self._preview, stretch=1)
        hint = QLabel("Wheel: zoom · Drag: pan · Double-click: fit · Draw ROI: drag a region")
        hint.setProperty("class", "dim")
        preview_col.addWidget(hint)
        body.addLayout(preview_col, stretch=1)

        # ---------------------------------------------------------- wiring
        app_state.preview_frame.connect(self._on_preview_frame)
        app_state.camera_state_changed.connect(self._on_camera_state)
        self.reload()

    # -------------------------------------------------------------- loading
    def reload(self) -> None:
        """Repopulate the camera list from configuration."""
        selected = self._current_index()
        self._list.clear()
        self._row_indexes = []
        for cfg in self._svc.get_configs():
            self._row_indexes.append(int(cfg["index"]))
            self._list.addItem(f"{cfg['index']}: {cfg.get('name', '')}")
        if self._row_indexes:
            row = self._row_indexes.index(selected) if selected in self._row_indexes else 0
            self._list.setCurrentRow(row)

    def _current_index(self) -> int | None:
        row = self._list.currentRow()
        return self._row_indexes[row] if 0 <= row < len(self._row_indexes) else None

    def _current_config(self) -> dict | None:
        index = self._current_index()
        for cfg in self._svc.get_configs():
            if int(cfg["index"]) == index:
                return cfg
        return None

    def _on_select(self, row: int) -> None:
        cfg = self._current_config()
        if cfg is None:
            return
        self._loading = True
        try:
            self._name.setText(cfg.get("name", ""))
            self._driver.setCurrentText(cfg.get("driver", "simulated"))
            self._conn_id.setText(str(cfg.get("connection_id", "")))
            self._enabled.setChecked(bool(cfg.get("enabled", True)))
            self._exposure.setValue(int(cfg.get("exposure_us", 10000)))
            self._gain.setValue(float(cfg.get("gain_db", 0.0)))
            self._gamma.setValue(float(cfg.get("gamma", 1.0)))
            self._brightness.setValue(int(cfg.get("brightness", 0)))
            self._width.setValue(int(cfg.get("width", 1280)))
            self._height.setValue(int(cfg.get("height", 1024)))
            self._trigger.setCurrentText(cfg.get("trigger_mode", "software"))
            roi = cfg.get("roi", {})
            values = (roi.get("x", 0), roi.get("y", 0), roi.get("width", 0), roi.get("height", 0))
            for spin, value in zip(self._roi_spins, values):
                spin.setValue(int(value))
            self._preview.set_roi(*values)
        finally:
            self._loading = False
        self._refresh_health()

    # ------------------------------------------------------------ collecting
    def _collect(self) -> dict:
        index = self._current_index()
        cfg = self._current_config() or {}
        cfg = dict(cfg)
        cfg.update(
            {
                "index": index,
                "name": self._name.text().strip() or f"Camera {index}",
                "driver": self._driver.currentText(),
                "connection_id": self._conn_id.text().strip(),
                "enabled": self._enabled.isChecked(),
                "exposure_us": self._exposure.value(),
                "gain_db": self._gain.value(),
                "gamma": self._gamma.value(),
                "brightness": self._brightness.value(),
                "width": self._width.value(),
                "height": self._height.value(),
                "trigger_mode": self._trigger.currentText(),
                "roi": {
                    "x": self._roi_spins[0].value(),
                    "y": self._roi_spins[1].value(),
                    "width": self._roi_spins[2].value(),
                    "height": self._roi_spins[3].value(),
                },
            }
        )
        return cfg

    # -------------------------------------------------------------- actions
    def _on_add(self) -> None:
        next_index = max(self._row_indexes, default=0) + 1
        try:
            self._svc.save_camera(
                {
                    "index": next_index,
                    "name": f"Camera {next_index}",
                    "driver": "simulated",
                    "connection_id": "",
                    "enabled": True,
                }
            )
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Add Camera", str(exc))
            return
        self.reload()
        self._list.setCurrentRow(self._row_indexes.index(next_index))

    def _on_remove(self) -> None:
        index = self._current_index()
        if index is None:
            return
        answer = QMessageBox.question(
            self, "Remove Camera", f"Remove camera {index} from the configuration?"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._svc.remove_camera(index)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Remove Camera", str(exc))
        self.reload()

    def _lifecycle(self, action: str) -> None:
        index = self._current_index()
        if index is None:
            return
        try:
            getattr(self._svc, action)(index)
        except VisionSystemError as exc:
            QMessageBox.warning(self, action.title(), str(exc))
        self._refresh_health()

    def _on_test(self) -> None:
        index = self._current_index()
        if index is None:
            return
        try:
            frame = self._svc.test_capture(index)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Test Camera", str(exc))
            return
        self._preview.set_frame(frame)

    def _on_apply(self) -> None:
        try:
            self._svc.apply_live(self._collect())
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Apply Live", str(exc))

    def _on_save(self) -> None:
        try:
            self._svc.save_camera(self._collect())
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Save Configuration", str(exc))
            return
        self.reload()

    # ------------------------------------------------------------------ ROI
    def _on_draw_toggled(self, checked: bool) -> None:
        self._preview.set_roi_mode(checked)

    def _on_roi_drawn(self, x: int, y: int, w: int, h: int) -> None:
        self._loading = True
        try:
            for spin, value in zip(self._roi_spins, (x, y, w, h)):
                spin.setValue(value)
        finally:
            self._loading = False

    def _on_roi_spins_changed(self) -> None:
        if not self._loading:
            values = [spin.value() for spin in self._roi_spins]
            self._preview.set_roi(*values)

    def _on_clear_roi(self) -> None:
        for spin in self._roi_spins:
            spin.setValue(0)
        self._preview.clear_roi()

    # --------------------------------------------------------------- events
    def _on_preview_frame(self, camera_index: int, frame) -> None:
        if self.isVisible() and camera_index == self._current_index():
            self._preview.set_frame(frame)

    def _on_camera_state(self, camera_index: int, state: str) -> None:
        if camera_index == self._current_index():
            self._refresh_health()

    def _refresh_health(self) -> None:
        index = self._current_index()
        if index is None:
            return
        try:
            health = self._svc.health(index)
        except KeyError:
            return
        state = (
            ConnectionState.ERROR
            if health.last_error
            else (ConnectionState.CONNECTED if health.connected else ConnectionState.DISCONNECTED)
        )
        detail = health.last_error or (
            f"{health.frames_captured} frames" if health.connected else "disconnected"
        )
        self._health.set_state(state, f"Camera {index}: {detail}")

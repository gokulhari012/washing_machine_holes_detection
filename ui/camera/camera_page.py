"""Camera Configuration page.

Left: camera list + lifecycle buttons + health + PLC jog D-pad (physical
camera-mount alignment). Centre: parameter form (exposure/gain/gamma/
light brightness/frame rate/rotation/resolution/trigger/ROI). Right: live preview on a
:class:`RoiEditor` — "Draw ROI" lets the operator drag the region directly
on the image and the spin boxes follow.

"Frame Rate (fps)" paces every *continuous* viewing mode for this camera —
the live preview/video workers, "Continuous Capture" below, and the
Calibration page's Auto Calibrate board scan all size their loop from it
(core.camera.frame_interval_ms), so none of them keeps a fixed rate of its
own. It throttles how often the app asks the camera for a frame; it does not
program an acquisition rate into the device. Whether the background preview
threads run at all remains a separate station-wide switch
(``app_config.ui.live_preview_fps``) — this is the rate they use once they
do. Continuous Capture re-paces the moment the value changes; the preview
workers are rebuilt on Save, by the composition root's config subscription.

"Rotation" turns every frame from this camera by a quarter turn (clockwise)
for a camera physically mounted on its side. It is applied *before* the ROI
crop, so the ROI drawn on the preview means what it looks like — but that
also makes an existing ROI and calibration meaningless once the rotation
changes: re-draw the ROI and re-run the calibration for that camera.

"Light Brightness" (0-255) is not an in-camera setting — it is the level
pushed to that camera's PLC register (CameraService._push_brightness),
driving an external, PLC-controlled light source, whenever settings are
applied or saved.

The jog controls nudge the camera's physical mounting position through PLC
registers (a read-modify-write, separate from anything in camera.json) one
axis at a time — X+/X-/Y+/Y-/Z+/Z- — used to centre a hole in frame before
capturing it as a machine model's image capture position. "Home" moves all
three axes to zero (the mount's true home, not a stored value) — Z is
skipped if this camera has no Z jog register configured. "Continuous
Capture" repeatedly re-captures the selected camera into the preview so the
operator can watch the effect of each nudge instead of clicking "Test
Camera" after every one.

Below the jog controls, a machine-model picker + "Save as Default"/"Go to
Default" let the operator snapshot the camera's current X/Y/Z position as
that model's image capture position (via
MachineModelService.save_camera_position) or restore it on demand. That
saved position is also pushed automatically whenever the model is applied —
MachineModelService.apply_profile, invoked both by "Apply Now" on the
Machine Models page and by the PLC's own model_select signal (main.py's
Application._on_machine_model_changed) — and by the Home button on each
dashboard camera panel, which moves that camera to the *active* model's
saved capture position rather than to zero (see
DashboardPage._on_home_requested).

"Apply Live" pushes settings to the connected device without persisting;
"Save" writes camera.json (+ DB mirror). Worker/manager rebuilds after a
save are handled by the composition root's config subscription.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
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

from core.camera import (
    DEFAULT_VIEW_FPS,
    VALID_ROTATIONS,
    frame_interval_ms,
    rotate_frame,
)
from core.camera.image_file_camera import IMAGE_NAME_FILTER, IMAGE_PATTERNS, read_image
from core.utilities.enums import CameraDriver, ConnectionState, TriggerMode
from core.utilities.exceptions import VisionSystemError
from models.app_state import AppState
from services.auth_service import AuthService
from services.camera_service import CameraService
from services.machine_model_service import MachineModelService
from services.plc_service import PlcService
from ui.widgets import LabeledLed, RoiEditor, minus_icon, plus_icon

# Adapters actually wired up in core.camera.create_camera(); USB/HikRobot/Daheng/IDS
# stay in CameraDriver for config-file compatibility but are hidden from this dropdown.
_SUPPORTED_DRIVERS = (CameraDriver.SIMULATED, CameraDriver.IMAGE_FILE, CameraDriver.BASLER)

_JOG_ICON_PX = 15


def _jog_button(icon, tooltip: str) -> QPushButton:
    button = QPushButton()
    button.setIcon(icon)
    button.setIconSize(QSize(_JOG_ICON_PX, _JOG_ICON_PX))
    button.setFixedSize(32, 32)
    button.setToolTip(tooltip)
    return button


class CameraPage(QWidget):
    """Add / remove / tune / test cameras."""

    def __init__(
        self,
        app_state: AppState,
        camera_service: CameraService,
        plc_service: PlcService,
        machine_model_service: MachineModelService,
        auth_service: AuthService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._app_state = app_state
        self._svc = camera_service
        self._plc = plc_service
        self._machine_models = machine_model_service
        self._auth = auth_service
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
        # A bordered, titled panel -- matching "Parameters" below -- instead
        # of bare controls floating directly on the page background. Without
        # this, an unframed list+buttons cluster next to a bordered
        # "Parameters" box looks unfinished, and on a maximized screen the
        # blank space a plain QGroupBox stretches into (see "Parameters")
        # reads as a broken empty page rather than part of a panel.
        left_box = QGroupBox("Cameras")
        left_box.setFixedWidth(230)
        left = QVBoxLayout(left_box)
        self._list = QListWidget()
        # A handful of cameras, not a scrolling record list -- capped so it
        # doesn't dominate the panel; addStretch() below keeps list+buttons
        # anchored at the top, with any leftover height sitting as blank
        # space *inside* the bordered panel instead of pushing the buttons
        # away from the list they act on.
        self._list.setMaximumHeight(200)
        self._list.currentRowChanged.connect(self._on_select)
        left.addWidget(self._list)

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
        self._continuous_btn = QPushButton("Continuous Capture")
        self._continuous_btn.setCheckable(True)
        self._continuous_btn.setToolTip(
            "Keep re-capturing the selected camera into the preview — "
            "use while jogging so you can see each nudge take effect"
        )
        self._continuous_btn.toggled.connect(self._on_continuous_toggled)
        left.addWidget(self._continuous_btn)
        self._health = LabeledLed("no camera selected")
        left.addWidget(self._health)

        self._jog_box = QGroupBox("Physical Position (PLC Jog)")
        jog_layout = QVBoxLayout(self._jog_box)
        pad = QGridLayout()
        pad.setSpacing(4)
        self._jog_plus: dict[str, QPushButton] = {}
        self._jog_minus: dict[str, QPushButton] = {}
        for col, axis in enumerate(("X", "Y", "Z")):
            pad.addWidget(QLabel(axis), 0, col)
            plus = _jog_button(plus_icon(), f"Jog camera {axis}+")
            minus = _jog_button(minus_icon(), f"Jog camera {axis}-")
            plus.clicked.connect(lambda _checked, a=axis: self._on_jog(f"{a.lower()}+"))
            minus.clicked.connect(lambda _checked, a=axis: self._on_jog(f"{a.lower()}-"))
            pad.addWidget(plus, 1, col)
            pad.addWidget(minus, 2, col)
            self._jog_plus[axis] = plus
            self._jog_minus[axis] = minus
        pad_row = QHBoxLayout()
        pad_row.addStretch()
        pad_row.addLayout(pad)
        pad_row.addStretch()
        jog_layout.addLayout(pad_row)

        self._jog_home = QPushButton("Home")
        self._jog_home.setToolTip("Move all axes (X/Y/Z) to zero")
        self._jog_home.clicked.connect(self._on_home)
        jog_layout.addWidget(self._jog_home)

        self._model_combo = QComboBox()
        self._model_combo.setToolTip(
            "Machine model to save/restore this camera's image capture position for"
        )
        jog_layout.addWidget(self._model_combo)
        self._save_default_btn = QPushButton("Save as Default")
        self._save_default_btn.setToolTip(
            "Store the camera's current X/Y/Z position as this machine "
            "model's image capture position"
        )
        self._save_default_btn.clicked.connect(self._on_save_default)
        jog_layout.addWidget(self._save_default_btn)
        self._goto_default_btn = QPushButton("Go to Default")
        self._goto_default_btn.setToolTip(
            "Jog the camera to this machine model's saved image capture position"
        )
        self._goto_default_btn.clicked.connect(self._on_go_to_default)
        jog_layout.addWidget(self._goto_default_btn)

        self._jog_hint = QLabel("")
        self._jog_hint.setProperty("class", "dim")
        self._jog_hint.setWordWrap(True)
        jog_layout.addWidget(self._jog_hint)
        left.addWidget(self._jog_box)

        left.addStretch()
        body.addWidget(left_box)

        # ------------------------------------------------------- form column
        form_box = QGroupBox("Parameters")
        form_box.setFixedWidth(330)
        form = QFormLayout(form_box)

        self._name = QLineEdit()
        self._driver = QComboBox()
        self._driver.addItems([d.value for d in _SUPPORTED_DRIVERS])
        self._driver.currentTextChanged.connect(self._on_driver_changed)
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
        self._brightness.setRange(0, 255)
        self._brightness.setToolTip(
            "Light-brightness level pushed to this camera's PLC register "
            "(not an in-camera setting) — takes effect on Apply/Save"
        )
        self._fps = QDoubleSpinBox()
        self._fps.setRange(0.1, 120.0)
        self._fps.setDecimals(1)
        self._fps.setSingleStep(1.0)
        self._fps.setSuffix(" fps")
        self._fps.setToolTip(
            "Frames per second for every continuous view of this camera — live "
            "preview, Continuous Capture and Auto Calibrate's board scan all "
            "run at this rate."
        )
        self._fps.valueChanged.connect(self._on_fps_changed)
        self._rotation = QComboBox()
        self._rotation.addItems([f"{degrees}°" for degrees in VALID_ROTATIONS])
        self._rotation.setToolTip(
            "Turn every frame from this camera clockwise — for a camera mounted "
            "on its side. Applied before the ROI crop, so changing it invalidates "
            "this camera's ROI and calibration; re-draw and re-calibrate after."
        )
        self._width = QSpinBox()
        self._width.setRange(64, 8192)
        self._height = QSpinBox()
        self._height.setRange(64, 8192)
        self._trigger = QComboBox()
        self._trigger.addItems([t.value for t in TriggerMode])

        # "image_file" driver: the uploaded picture that is streamed as frames
        self._image_source = QLineEdit()
        self._image_source.setPlaceholderText("no image chosen")
        self._image_source.setToolTip(
            "Picture (or folder of pictures) streamed as this camera's frames"
        )
        choose_image = QPushButton("Choose Image…")
        choose_image.setToolTip("Pick a picture to inspect live")
        choose_image.clicked.connect(self._on_choose_image)
        choose_folder = QPushButton("Folder…")
        choose_folder.setToolTip("Pick a folder of pictures to cycle through")
        choose_folder.clicked.connect(self._on_choose_folder)
        image_buttons = QHBoxLayout()
        image_buttons.setContentsMargins(0, 0, 0, 0)
        image_buttons.addWidget(choose_image)
        image_buttons.addWidget(choose_folder)
        image_column = QVBoxLayout()
        image_column.setContentsMargins(0, 0, 0, 0)
        image_column.addWidget(self._image_source)
        image_column.addLayout(image_buttons)
        self._image_widget = QWidget()
        self._image_widget.setLayout(image_column)

        form.addRow("Name", self._name)
        form.addRow("Driver", self._driver)
        form.addRow("Image Source", self._image_widget)
        form.addRow("Connection ID", self._conn_id)
        self._form = form
        form.addRow("", self._enabled)
        form.addRow("Exposure", self._exposure)
        form.addRow("Gain", self._gain)
        form.addRow("Gamma", self._gamma)
        form.addRow("Light Brightness (PLC)", self._brightness)
        form.addRow("Frame Rate", self._fps)
        form.addRow("Rotation", self._rotation)
        form.addRow("Width", self._width)
        form.addRow("Height", self._height)
        detect_res_btn = QPushButton("Detect Resolution")
        detect_res_btn.setToolTip(
            "Read the actual resolution from the connected camera or the chosen image"
        )
        detect_res_btn.clicked.connect(self._on_detect_resolution)
        form.addRow("", detect_res_btn)
        form.addRow("Trigger Mode", self._trigger)

        # ROI block -- 2x2 table: X/Y on the first row, W/H on the second
        self._roi_spins = []
        roi_grid = QGridLayout()
        roi_grid.setContentsMargins(0, 0, 0, 0)
        roi_grid.setHorizontalSpacing(6)
        for position, caption in enumerate(("X", "Y", "W", "H")):
            spin = QSpinBox()
            spin.setRange(0, 8192)
            spin.setToolTip(f"ROI {caption}")
            spin.valueChanged.connect(self._on_roi_spins_changed)
            self._roi_spins.append(spin)
            row, col = divmod(position, 2)
            roi_grid.addWidget(QLabel(caption), row, col * 2)
            roi_grid.addWidget(spin, row, col * 2 + 1)
        roi_widget = QWidget()
        roi_widget.setLayout(roi_grid)
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
        # Bordered/titled like the other two panels: an unframed ImageView
        # is nearly the same colour as the page background, so with no live
        # camera connected it reads as a hole in the page rather than an
        # actual "no signal yet" panel.
        preview_box = QGroupBox("Live Preview")
        preview_col = QVBoxLayout(preview_box)
        self._preview = RoiEditor()
        self._preview.setMinimumSize(480, 360)
        self._preview.roi_changed.connect(self._on_roi_drawn)
        preview_col.addWidget(self._preview, stretch=1)
        hint = QLabel("Wheel: zoom · Drag: pan · Double-click: fit · Draw ROI: drag a region")
        hint.setProperty("class", "dim")
        preview_col.addWidget(hint)
        self._image_hint = QLabel()
        self._image_hint.setProperty("class", "dim")
        self._image_hint.setWordWrap(True)
        preview_col.addWidget(self._image_hint)
        body.addWidget(preview_box, stretch=1)

        # ---------------------------------------------------------- wiring
        self._continuous_timer = QTimer(self)
        self._continuous_timer.setInterval(frame_interval_ms(DEFAULT_VIEW_FPS))
        self._continuous_timer.timeout.connect(self._on_continuous_tick)

        app_state.preview_frame.connect(self._on_preview_frame)
        app_state.camera_state_changed.connect(self._on_camera_state)
        app_state.active_machine_model_changed.connect(lambda *_: self._reload_model_combo())
        self._on_driver_changed(self._driver.currentText())
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
        self._continuous_btn.setChecked(False)  # stop streaming the camera we're leaving
        cfg = self._current_config()
        if cfg is None:
            return
        self._loading = True
        try:
            self._name.setText(cfg.get("name", ""))
            self._driver.setCurrentText(cfg.get("driver", "simulated"))
            self._image_source.setText(str(cfg.get("image_source", "")))
            self._conn_id.setText(str(cfg.get("connection_id", "")))
            self._enabled.setChecked(bool(cfg.get("enabled", True)))
            self._exposure.setValue(int(cfg.get("exposure_us", 10000)))
            self._gain.setValue(float(cfg.get("gain_db", 0.0)))
            self._gamma.setValue(float(cfg.get("gamma", 1.0)))
            self._brightness.setValue(int(cfg.get("brightness", 0)))
            self._fps.setValue(float(cfg.get("fps", DEFAULT_VIEW_FPS)))
            self._rotation.setCurrentText(f"{int(cfg.get('rotation', 0))}°")
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
        self._refresh_jog()

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
                "fps": self._fps.value(),
                "rotation": self._current_rotation(),
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
        image_source = self._image_source.text().strip()
        if image_source:  # only the image_file driver uses it — keep other entries clean
            cfg["image_source"] = image_source
        else:
            cfg.pop("image_source", None)
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

    # ---------------------------------------------------------- image source
    def _on_driver_changed(self, driver: str) -> None:
        """Only the ``image_file`` driver needs the picture picker."""
        is_image = driver == CameraDriver.IMAGE_FILE.value
        self._form.setRowVisible(self._image_widget, is_image)
        self._image_hint.setVisible(is_image)

    def _on_choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Image", self._image_start_dir(), IMAGE_NAME_FILTER
        )
        if path:
            self._set_image_source(path)

    def _on_choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose Image Folder", self._image_start_dir()
        )
        if path:
            self._set_image_source(path)

    def _image_start_dir(self) -> str:
        current = Path(self._image_source.text().strip() or ".")
        if current.is_dir():
            return str(current)
        return str(current.parent) if current.is_file() else ""

    def _set_image_source(self, path: str) -> None:
        """Adopt the chosen picture: switch to the image driver, auto-detect
        its resolution, and preview it."""
        self._image_source.setText(path)
        self._driver.setCurrentText(CameraDriver.IMAGE_FILE.value)
        frame = read_image(self._first_image(Path(path)))
        if frame is None:
            self._image_hint.setText("")
            QMessageBox.warning(self, "Choose Image", f"Cannot read an image from:\n{path}")
            return
        height, width = frame.shape[:2]
        # Width/Height describe the *source* picture, as the driver reads it;
        # the preview shows it turned the way capture() will, so an ROI drawn
        # here means the same thing as one drawn on a live frame.
        self._width.setValue(width)
        self._height.setValue(height)
        self._preview.set_frame(rotate_frame(frame, self._current_rotation()))
        self._image_hint.setText(
            f"Resolution detected from the image ({width}x{height}). Press "
            f"“Save Configuration” to stream it live into the preview and the "
            f"inspection pipeline."
        )

    @staticmethod
    def _first_image(source: Path) -> Path:
        """``source`` itself for a file, else its first image (folder case)."""
        if not source.is_dir():
            return source
        candidates = sorted(p for pattern in IMAGE_PATTERNS for p in source.glob(pattern))
        return candidates[0] if candidates else source

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

    def _on_continuous_toggled(self, checked: bool) -> None:
        if not checked:
            self._continuous_timer.stop()
            return
        if self._current_index() is None:
            self._continuous_btn.setChecked(False)
            return
        self._apply_continuous_interval()
        self._continuous_timer.start()

    def _apply_continuous_interval(self) -> None:
        """Pace the capture loop from the form's frame rate.

        Read off the spin box rather than the saved config so the operator can
        dial the rate in while watching the stream, before committing it.
        """
        self._continuous_timer.setInterval(frame_interval_ms(self._fps.value()))

    def _current_rotation(self) -> int:
        """The form's rotation in degrees clockwise (the combo shows e.g. "90°")."""
        return int(self._rotation.currentText().rstrip("°"))

    def _on_fps_changed(self, _value: float) -> None:
        """Re-pace a running Continuous Capture the moment the rate changes."""
        if self._loading:
            return
        self._apply_continuous_interval()

    def _on_continuous_tick(self) -> None:
        index = self._current_index()
        if index is None:
            self._continuous_btn.setChecked(False)
            return
        try:
            frame = self._svc.test_capture(index)
        except VisionSystemError as exc:
            self._continuous_btn.setChecked(False)  # stops the timer via toggled(False)
            QMessageBox.warning(self, "Continuous Capture", str(exc))
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

    def _on_detect_resolution(self) -> None:
        """Read the actual resolution from the camera or its image source.

        Cameras: the device must already be connected (see 'Connect'), so its
        real sensor/native resolution can be queried. Image file: works
        without connecting, since it just reads the picture on disk.
        """
        index = self._current_index()
        if index is None:
            return
        try:
            width, height = self._svc.detect_resolution(index)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Detect Resolution", str(exc))
            return
        self._width.setValue(width)
        self._height.setValue(height)

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

    # ---------------------------------------------------------------- jog
    def _refresh_jog(self) -> None:
        index = self._current_index()
        configured = index is not None and self._plc.jog_configured(index)
        self._jog_box.setEnabled(configured)
        z_configured = index is not None and self._plc.jog_z_configured(index)
        z_tooltip = "Jog camera Z{}" if z_configured else "No Z jog register configured for this camera"
        self._jog_plus["Z"].setEnabled(z_configured)
        self._jog_plus["Z"].setToolTip(z_tooltip.format("+"))
        self._jog_minus["Z"].setEnabled(z_configured)
        self._jog_minus["Z"].setToolTip(z_tooltip.format("-"))
        self._jog_hint.setText(
            "" if configured else "No PLC jog registers configured for this camera"
        )
        self._reload_model_combo()

    def _reload_model_combo(self) -> None:
        """Repopulate the default-position model picker, preferring the
        previously selected profile, else the currently active model."""
        previous_id = self._model_combo.currentData()
        active_name, active_code = self._app_state.active_machine_model
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for profile in self._machine_models.list_profiles():
            self._model_combo.addItem(
                f"{profile['name']} (code {profile['plc_code']})", profile["id"]
            )
        restored = False
        if previous_id is not None:
            row = self._model_combo.findData(previous_id)
            if row >= 0:
                self._model_combo.setCurrentIndex(row)
                restored = True
        if not restored and active_name:
            active_profile = self._machine_models.get_by_code(active_code)
            if active_profile is not None:
                row = self._model_combo.findData(active_profile["id"])
                if row >= 0:
                    self._model_combo.setCurrentIndex(row)
        self._model_combo.blockSignals(False)

    def _selected_profile_id(self) -> int | None:
        return self._model_combo.currentData()

    def _on_save_default(self) -> None:
        index = self._current_index()
        if index is None:
            return
        profile_id = self._selected_profile_id()
        if profile_id is None:
            QMessageBox.warning(self, "Save Default Position", "No machine model selected.")
            return
        if not self._auth.is_admin:
            QMessageBox.warning(
                self, "Save Default Position",
                "Administrator login required (toolbar Login button).",
            )
            return
        try:
            x, y, z = self._plc.read_camera_position(index)
            profile = self._machine_models.save_camera_position(
                profile_id, index, x, y, z, updated_by=self._username()
            )
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Save Default Position", str(exc))
            return
        self._jog_hint.setText(
            f"Saved ({x}, {y}, {z}) as image capture position for {profile['name']!r}."
        )

    def _on_go_to_default(self) -> None:
        index = self._current_index()
        if index is None:
            return
        profile_id = self._selected_profile_id()
        if profile_id is None:
            QMessageBox.warning(self, "Go to Default Position", "No machine model selected.")
            return
        profile = self._machine_models.get_by_id(profile_id)
        position = (profile or {}).get("jog_positions", {}).get(str(index))
        if position is None:
            QMessageBox.warning(
                self, "Go to Default Position",
                "No image capture position saved for this camera on the selected machine model.",
            )
            return
        if not self._auth.is_admin:
            QMessageBox.warning(
                self, "Go to Default Position",
                "Administrator login required (toolbar Login button).",
            )
            return
        try:
            self._plc.set_camera_position(
                index, int(position["x"]), int(position["y"]), int(position.get("z", 0))
            )
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Go to Default Position", str(exc))

    def _username(self) -> str:
        user = self._auth.current_user
        return user.username if user is not None else ""

    def _on_jog(self, direction: str) -> None:
        index = self._current_index()
        if index is None:
            return
        if not self._auth.is_admin:
            QMessageBox.warning(
                self, "Jog Camera", "Administrator login required (toolbar Login button)."
            )
            return
        try:
            self._plc.jog_camera(index, direction)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Jog Camera", str(exc))

    def _on_home(self) -> None:
        index = self._current_index()
        if index is None:
            return
        if not self._auth.is_admin:
            QMessageBox.warning(
                self, "Home Camera", "Administrator login required (toolbar Login button)."
            )
            return
        try:
            self._plc.home_camera(index)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Home Camera", str(exc))

    # --------------------------------------------------------------- events
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._reload_model_combo()  # profiles may have changed on the Machine Models page

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self._continuous_btn.setChecked(False)  # never keep hammering test_capture off-screen

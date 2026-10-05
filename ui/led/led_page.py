"""LED Controller page: connection settings (set up once), manual light
control, each camera's light brightness, a raw command tester and a
communication log for troubleshooting the hardware directly.

Two brightness panels that look alike and deliberately are not:

- **Manual Light Control** — ON/OFF per channel and ALL ON/ALL OFF, each
  channel at its own *manual* level (``led.json``'s ``manual_brightness``).
  For testing lights by hand only: nothing else reads those levels, so the
  inspection cycle is unaffected, and the next cycle's strobe (or a camera's
  steady-brightness push) overwrites whatever a channel was left at.
- **Camera Light Brightness** — the very same ``brightness`` field as the
  Camera Configuration page's "Light Brightness", per camera or all at once,
  through ``CameraService.set_brightness``: persisted, live for the next
  trigger cycle, pushed to the camera's channel, folded into the applied
  machine model, and reflected on the Camera page via
  ``AppState.camera_brightness_changed``. This page re-reads the live values
  whenever it is shown or a machine model is applied, since the Camera page
  and a model switch can change them too.

  Each camera row also carries a **Strobe** switch (plus Strobe "Enable All"
  / "Disable All"), the very same ``led_strobe`` field as the Camera page's
  "Strobe" checkbox, through ``CameraService.set_strobe``: saved and live at
  once, the channel moved to its new resting state (off for strobe, steady
  brightness otherwise), and mirrored on the Camera page via
  ``AppState.camera_strobe_changed``.

Which channel lights which camera ("LED Channel") still lives on the Camera
Configuration page only — it is wiring, not a level. Every channel command,
from either panel, is clamped to ``max_brightness``.

The connection auto-connects when the application starts
(``Application.start``) and again immediately after a Save
(``Application._on_led_config_saved``), the same way the PLC link does —
Connect/Disconnect here are for manual reconnect/troubleshooting, not normal
operation.
"""

from __future__ import annotations

from html import escape

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from workers import LedCommandWorker

from core.led import protocol
from core.utilities.exceptions import VisionSystemError
from models.app_state import AppState
from services.camera_service import CameraService
from services.led_service import CHANNELS, LedService
from ui import theme
from ui.widgets import LabeledLed

BAUD_OPTIONS = ["9600", "19200", "38400", "57600", "115200"]
#: Log line kind -> palette token. The log is drawn as inline HTML rather than
#: styled by the QSS (a QTextEdit's contents are a document, not widgets), so
#: these are looked up through ``theme.color`` at render time and the whole
#: transcript is re-rendered when the scheme changes — an inline hex here
#: would keep the dark pastels on a white background, where they vanish.
_LOG_TOKENS = {
    "tx": "led-log-tx",
    "rx": "led-log-rx",
    "error": "led-log-error",
    "info": "led-log-info",
}
#: Transcript lines kept for re-rendering after a theme change. Bounded so a
#: long troubleshooting session cannot grow the buffer without limit.
_LOG_LIMIT = 500


class LedPage(QWidget):
    """LED Controller (KDC-24V60W-4T): connection, manual light control,
    per-camera light brightness, raw command tester."""

    def __init__(
        self,
        app_state: AppState,
        led_service: LedService,
        camera_service: CameraService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._svc = led_service
        self._cameras = camera_service
        # Manual Light Control, per channel: level spin box, wiring label,
        # last-sent label, and what this page last sent the channel (None
        # until it sends anything — a trigger cycle may have changed it since).
        self._manual_spins: dict[int, QSpinBox] = {}
        self._manual_wiring: dict[int, QLabel] = {}
        self._manual_state: dict[int, QLabel] = {}
        self._manual_buttons: list[QPushButton] = []
        self._manual_on: dict[int, bool | None] = {channel: None for channel in CHANNELS}
        # Camera Light Brightness, per camera index. The rows are rebuilt when
        # the configured cameras change (see _refresh_cameras).
        self._camera_rows_host: QWidget | None = None
        self._camera_spins: dict[int, QSpinBox] = {}
        self._camera_names: dict[int, QLabel] = {}
        self._camera_wiring: dict[int, QLabel] = {}
        self._camera_strobes: dict[int, QCheckBox] = {}
        # (timestamp, message, kind) for every line on screen, so the log can
        # be re-rendered in the other scheme's colours on a theme switch.
        self._entries: list[tuple[str, str, str]] = []
        # The raw tester is the one LED command that waits for a reply, so it
        # runs on LedCommandWorker rather than here — see _on_raw_send.
        self._raw_worker: LedCommandWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)

        title_row = QHBoxLayout()
        title = QLabel("LED Controller")
        title.setProperty("class", "pageTitle")
        self._state_led = LabeledLed("LED")
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self._state_led)
        root.addLayout(title_row)

        warning = QLabel(
            "⚠ Verify the LED/light source wiring and controller output ratings "
            "before testing. Output: 1–24 V, max 60 W total, max 2 A per channel, "
            "4 channels, trigger 12–24 V."
        )
        warning.setWordWrap(True)
        warning.setProperty("class", "danger")
        root.addWidget(warning)

        note = QLabel(
            "Which channel lights which camera is set on the Camera "
            "Configuration page."
        )
        note.setWordWrap(True)
        note.setProperty("class", "dim")
        root.addWidget(note)

        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter, stretch=1)
        splitter.addWidget(
            self._side_by_side(self._build_connection_group(), self._build_raw_command_group())
        )
        splitter.addWidget(
            self._side_by_side(self._build_manual_group(), self._build_camera_brightness_group())
        )
        splitter.addWidget(self._build_log_group())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)

        app_state.led_state_changed.connect(self._on_led_state_changed)
        app_state.active_machine_model_changed.connect(self._on_machine_model_applied)
        app_state.camera_brightness_changed.connect(self._on_camera_brightness_changed)
        theme.subscribe(self._repaint_log)
        self._load()

    @staticmethod
    def _side_by_side(left: QWidget, right: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(left, stretch=1)
        layout.addWidget(right, stretch=1)
        return row

    # ------------------------------------------------------------------ build
    def _build_connection_group(self) -> QWidget:
        box = QGroupBox("Connection")
        form = QFormLayout(box)

        self._driver = QComboBox()
        self._driver.addItems(["serial", "simulated"])
        self._driver.currentTextChanged.connect(self._on_driver_changed)
        form.addRow("Driver", self._driver)

        port_row = QHBoxLayout()
        self._port = QComboBox()
        self._port.setEditable(True)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_ports)
        port_row.addWidget(self._port, stretch=1)
        port_row.addWidget(refresh_btn)
        form.addRow("COM Port", port_row)

        self._baud = QComboBox()
        self._baud.addItems(BAUD_OPTIONS)
        form.addRow("Baud Rate", self._baud)

        self._timeout = QSpinBox()
        self._timeout.setRange(100, 10000)
        self._timeout.setSingleStep(100)
        self._timeout.setSuffix(" ms")
        form.addRow("Response Timeout", self._timeout)

        self._max_brightness = QSpinBox()
        self._max_brightness.setRange(protocol.MIN_BRIGHTNESS, protocol.MAX_BRIGHTNESS)
        self._max_brightness.setToolTip(
            "Safety ceiling: no channel command sent through this application "
            "— from any camera's Light Brightness or the raw command box below "
            "— exceeds this value."
        )
        form.addRow("Max Brightness", self._max_brightness)

        buttons = QHBoxLayout()
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._on_connect)
        self._disconnect_btn = QPushButton("Disconnect")
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        save_btn = QPushButton("Save")
        save_btn.setProperty("class", "primary")
        save_btn.clicked.connect(self._on_save)
        buttons.addWidget(self._connect_btn)
        buttons.addWidget(self._disconnect_btn)
        buttons.addWidget(save_btn)
        form.addRow(buttons)

        save_note = QLabel(
            "The application connects automatically at startup and again "
            "right after Save. Connect/Disconnect here are for manual "
            "reconnect and troubleshooting."
        )
        save_note.setWordWrap(True)
        save_note.setProperty("class", "dim")
        form.addRow(save_note)

        return box

    def _build_raw_command_group(self) -> QWidget:
        box = QGroupBox("Raw RS232 Command")
        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Command:"))
        self._raw_command = QLineEdit()
        self._raw_command.setPlaceholderText("e.g. SA0200#  or  S100T128T025F000TC#")
        self._raw_command.returnPressed.connect(self._on_raw_send)
        row.addWidget(self._raw_command, stretch=1)
        self._raw_send_btn = QPushButton("SEND")
        self._raw_send_btn.clicked.connect(self._on_raw_send)
        row.addWidget(self._raw_send_btn)
        layout.addLayout(row)

        response_row = QHBoxLayout()
        response_row.addWidget(QLabel("Response:"))
        self._raw_response = QLineEdit()
        self._raw_response.setReadOnly(True)
        response_row.addWidget(self._raw_response, stretch=1)
        layout.addLayout(response_row)

        self._raw_crlf = QCheckBox("Append CR/LF")
        layout.addWidget(self._raw_crlf)

        examples = QLabel(
            "Examples:  SA0200#   SB0100#   SC0200#   SD0255#\n"
            "Multi-channel documented format:  S100T128T025F000TC#"
        )
        examples.setProperty("class", "dim")
        layout.addWidget(examples)
        return box

    def _build_manual_group(self) -> QWidget:
        box = QGroupBox("Manual Light Control")
        layout = QVBoxLayout(box)

        note = QLabel(
            "For testing lights by hand. These levels are used only by the "
            "buttons here — trigger cycles light each camera at its own Camera "
            "Light Brightness, and may switch a channel you set here."
        )
        note.setWordWrap(True)
        note.setProperty("class", "dim")
        layout.addWidget(note)

        grid = QGridLayout()
        for column, heading in ((0, "Channel"), (1, "Wired to"), (2, "Brightness"), (5, "Sent")):
            label = QLabel(heading)
            label.setProperty("class", "dim")
            grid.addWidget(label, 0, column)
        for row, channel in enumerate(CHANNELS, start=1):
            grid.addWidget(QLabel(f"{channel} ({protocol.CHANNEL_LETTERS[channel]})"), row, 0)

            wiring = QLabel("—")
            wiring.setProperty("class", "dim")
            self._manual_wiring[channel] = wiring
            grid.addWidget(wiring, row, 1)

            spin = QSpinBox()
            spin.setRange(protocol.MIN_BRIGHTNESS, protocol.MAX_BRIGHTNESS)
            # Arrow steps re-send while the channel is on; typed digits wait
            # for Enter/focus-out rather than sending "1", "15", "150".
            spin.setKeyboardTracking(False)
            spin.setToolTip("Manual level for this channel (clamped to Max Brightness)")
            spin.valueChanged.connect(
                lambda _value, ch=channel: self._on_manual_level_changed(ch)
            )
            self._manual_spins[channel] = spin
            grid.addWidget(spin, row, 2)

            on_btn = QPushButton("ON")
            on_btn.clicked.connect(lambda _checked=False, ch=channel: self._on_manual_on(ch))
            off_btn = QPushButton("OFF")
            off_btn.clicked.connect(lambda _checked=False, ch=channel: self._on_manual_off(ch))
            self._manual_buttons += [on_btn, off_btn]
            grid.addWidget(on_btn, row, 3)
            grid.addWidget(off_btn, row, 4)

            state = QLabel("—")
            self._manual_state[channel] = state
            grid.addWidget(state, row, 5)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        all_row = QHBoxLayout()
        all_on = QPushButton("ALL ON")
        all_on.setToolTip("Light every channel, each at its own level above")
        all_on.clicked.connect(self._on_manual_all_on)
        all_off = QPushButton("ALL OFF")
        all_off.clicked.connect(self._on_manual_all_off)
        self._manual_buttons += [all_on, all_off]
        all_row.addStretch()
        all_row.addWidget(all_on)
        all_row.addWidget(all_off)
        layout.addLayout(all_row)
        layout.addStretch()
        return box

    def _build_camera_brightness_group(self) -> QWidget:
        box = QGroupBox("Camera Light Brightness")
        self._camera_box_layout = QVBoxLayout(box)

        note = QLabel(
            "The same setting as Cameras → Light Brightness. Set saves it "
            "immediately, updates the Cameras page, and is used from the next "
            "trigger cycle. Strobe lights a camera only while it captures; "
            "switching it saves immediately too."
        )
        note.setWordWrap(True)
        note.setProperty("class", "dim")
        self._camera_box_layout.addWidget(note)

        # The per-camera rows are a host widget inserted at index 1, between
        # this note and the "All cameras" row — see _build_camera_rows.
        all_row = QHBoxLayout()
        all_row.addWidget(QLabel("All cameras"), stretch=1)
        self._all_cameras_spin = QSpinBox()
        self._all_cameras_spin.setRange(protocol.MIN_BRIGHTNESS, protocol.MAX_BRIGHTNESS)
        all_row.addWidget(self._all_cameras_spin)
        set_all = QPushButton("Set All")
        set_all.setProperty("class", "primary")
        set_all.clicked.connect(self._on_set_all_cameras)
        all_row.addWidget(set_all)
        self._camera_box_layout.addLayout(all_row)

        strobe_row = QHBoxLayout()
        strobe_row.addWidget(QLabel("Strobe, all cameras"), stretch=1)
        enable_all = QPushButton("Enable All")
        enable_all.setToolTip("Light every camera only while it is capturing")
        enable_all.clicked.connect(lambda: self._on_set_all_strobe(True))
        strobe_row.addWidget(enable_all)
        disable_all = QPushButton("Disable All")
        disable_all.setToolTip("Hold every camera's light steady at its brightness")
        disable_all.clicked.connect(lambda: self._on_set_all_strobe(False))
        strobe_row.addWidget(disable_all)
        self._camera_box_layout.addLayout(strobe_row)
        self._camera_box_layout.addStretch()
        return box

    def _build_camera_rows(self, indexes: list[int]) -> None:
        """(Re)build one row per configured camera. The old host is
        ``deleteLater``-ed rather than deleted, because this can run from
        inside one of its own buttons' click handlers."""
        if self._camera_rows_host is not None:
            self._camera_box_layout.removeWidget(self._camera_rows_host)
            self._camera_rows_host.deleteLater()
        self._camera_spins.clear()
        self._camera_names.clear()
        self._camera_wiring.clear()
        self._camera_strobes.clear()

        host = QWidget()
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        for row, index in enumerate(indexes):
            name = QLabel(f"Camera {index}")
            self._camera_names[index] = name
            grid.addWidget(name, row, 0)

            wiring = QLabel("")
            wiring.setProperty("class", "dim")
            self._camera_wiring[index] = wiring
            grid.addWidget(wiring, row, 1)

            spin = QSpinBox()
            spin.setRange(protocol.MIN_BRIGHTNESS, protocol.MAX_BRIGHTNESS)
            self._camera_spins[index] = spin
            grid.addWidget(spin, row, 2)

            set_btn = QPushButton("Set")
            set_btn.clicked.connect(
                lambda _checked=False, i=index: self._on_set_camera_brightness(i)
            )
            grid.addWidget(set_btn, row, 3)

            strobe = QCheckBox("Strobe")
            strobe.setToolTip(
                "On: the light is lit only while this camera captures.\n"
                "Off: the light is held steady at this brightness.\n"
                "Saved immediately — same setting as Cameras → Strobe."
            )
            # clicked, not toggled: only an operator click applies, never the
            # setChecked of a refresh.
            strobe.clicked.connect(
                lambda checked, i=index: self._on_set_camera_strobe(i, checked)
            )
            self._camera_strobes[index] = strobe
            grid.addWidget(strobe, row, 4)
        grid.setColumnStretch(1, 1)
        self._camera_box_layout.insertWidget(1, host)
        self._camera_rows_host = host

    def _build_log_group(self) -> QWidget:
        box = QGroupBox("Communication Log")
        layout = QVBoxLayout(box)
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        layout.addWidget(self._log_view, stretch=1)
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self._clear_log)
        layout.addWidget(clear_btn)
        return box

    # ------------------------------------------------------------------- load
    def _load(self) -> None:
        cfg = self._svc.get_config()
        connection = cfg.get("connection", {})
        self._driver.setCurrentText(str(connection.get("driver", "simulated")))
        port = str(connection.get("port", ""))
        self._refresh_ports()
        if port:
            self._port.setCurrentText(port)
        self._baud.setCurrentText(str(connection.get("baud_rate", protocol.DEFAULT_BAUD_RATE)))
        self._timeout.setValue(int(connection.get("timeout_ms", protocol.DEFAULT_TIMEOUT_MS)))
        self._max_brightness.setValue(
            int(cfg.get("max_brightness", protocol.DEFAULT_MAX_BRIGHTNESS))
        )
        self._on_driver_changed(self._driver.currentText())
        self._state_led.set_state(self._svc.state, f"LED {self._svc.state.value}")
        self._set_controls_enabled(self._svc.state.value == "connected")
        for channel, level in self._svc.manual_levels().items():
            spin = self._manual_spins[channel]
            spin.blockSignals(True)  # loading is not an operator edit
            spin.setValue(level)
            spin.blockSignals(False)
        self._refresh_cameras()

    def _refresh_cameras(self) -> None:
        """Re-read every camera's live brightness and channel wiring.

        Reads the live-effective configs, not camera.json: after a machine
        model switch the two differ, and this panel must show what the next
        cycle will actually use.
        """
        configs = self._cameras.get_effective_configs()
        indexes = [int(cfg["index"]) for cfg in configs]
        if indexes != list(self._camera_spins):
            self._build_camera_rows(indexes)

        wired: dict[int, list[str]] = {channel: [] for channel in CHANNELS}
        for cfg in configs:
            index = int(cfg["index"])
            channel = int(cfg.get("led_channel", 0))
            strobe = bool(cfg.get("led_strobe", False))
            name = str(cfg.get("name", "")).strip()
            self._camera_names[index].setText(
                f"Camera {index}: {name}" if name else f"Camera {index}"
            )
            if channel in wired:
                self._camera_wiring[index].setText(
                    f"Ch {channel} · {'strobe' if strobe else 'steady'}"
                )
                wired[channel].append(f"Cam {index}" + (" (strobe)" if strobe else ""))
            else:
                self._camera_wiring[index].setText("no LED channel")
            self._camera_spins[index].setValue(int(cfg.get("brightness", 0)))
            self._camera_strobes[index].setChecked(strobe)
        for channel, label in self._manual_wiring.items():
            label.setText(", ".join(wired[channel]) or "not wired")

    def _collect(self) -> dict:
        cfg = self._svc.get_config()
        cfg["connection"] = {
            **cfg.get("connection", {}),
            "driver": self._driver.currentText(),
            "port": self._port.currentText().strip(),
            "baud_rate": int(self._baud.currentText()),
            "timeout_ms": self._timeout.value(),
        }
        cfg["max_brightness"] = self._max_brightness.value()
        return cfg

    def showEvent(self, event) -> None:  # noqa: N802
        """Brightness and wiring can change on the Camera page (or by a
        model switch) while this page is hidden — re-read on every show."""
        super().showEvent(event)
        self._refresh_cameras()

    # ---------------------------------------------------------------- actions
    def _on_machine_model_applied(self, _name: str, _plc_code: int) -> None:
        self._refresh_cameras()

    def _on_camera_brightness_changed(self, camera_index: int, brightness: int) -> None:
        spin = self._camera_spins.get(camera_index)
        if spin is not None:
            spin.setValue(int(brightness))

    def _on_led_state_changed(self, value: str) -> None:
        """Keep the status LED and Connect/Disconnect buttons in sync with a
        state change from anywhere — startup auto-connect, a Save-triggered
        reconnect, or a comm error surfaced by a camera's brightness push."""
        self._state_led.set_state(value, f"LED {value}")
        self._set_controls_enabled(value == "connected")

    def _refresh_ports(self) -> None:
        current = self._port.currentText()
        ports = self._svc.list_ports()
        self._port.clear()
        self._port.addItems(ports)
        if current:
            self._port.setCurrentText(current)

    def _on_driver_changed(self, driver: str) -> None:
        is_serial = driver == "serial"
        self._port.setEnabled(is_serial)
        self._baud.setEnabled(is_serial)

    def _on_save(self) -> None:
        try:
            self._svc.save_config(self._collect())
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Save", str(exc))
            return
        self._log("LED controller configuration saved — reconnecting.", "info")
        self._set_controls_enabled(self._svc.state.value == "connected")

    def _on_connect(self) -> None:
        try:
            self._svc.connect()
        except VisionSystemError as exc:
            self._log(f"ERROR: {exc}", "error")
            QMessageBox.warning(self, "Connect", str(exc))
            return
        self._set_controls_enabled(True)
        self._log("Connected", "info")

    def _on_disconnect(self) -> None:
        self._svc.disconnect()
        self._set_controls_enabled(False)
        self._log("Disconnected", "info")

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._connect_btn.setEnabled(not enabled)
        self._disconnect_btn.setEnabled(enabled)
        # Manual commands need the link. Camera Light Brightness does not: it
        # is saved and live regardless, and lights on the next cycle.
        for button in self._manual_buttons:
            button.setEnabled(enabled)

    # ---------------------------------------------------------------- manual
    def _on_manual_on(self, channel: int) -> None:
        level = self._manual_spins[channel].value()
        if self._send_manual(
            f"Channel {channel} ON",
            lambda: self._svc.manual_on(channel, level),
            self._svc.channel_command(channel, level),
        ):
            self._mark_manual(channel, True)
            self._remember_manual_levels()

    def _on_manual_off(self, channel: int) -> None:
        if self._send_manual(
            f"Channel {channel} OFF",
            lambda: self._svc.manual_off(channel),
            self._svc.channel_command(channel, 0),
        ):
            self._mark_manual(channel, False)

    def _on_manual_all_on(self) -> None:
        levels = {channel: spin.value() for channel, spin in self._manual_spins.items()}
        commands = " ".join(self._svc.channel_command(ch, lv) for ch, lv in levels.items())
        if self._send_manual("ALL ON", lambda: self._svc.manual_all_on(levels), commands):
            for channel in CHANNELS:
                self._mark_manual(channel, True)
            self._remember_manual_levels()

    def _on_manual_all_off(self) -> None:
        commands = " ".join(self._svc.channel_command(ch, 0) for ch in CHANNELS)
        if self._send_manual("ALL OFF", self._svc.manual_all_off, commands):
            for channel in CHANNELS:
                self._mark_manual(channel, False)

    def _on_manual_level_changed(self, channel: int) -> None:
        """A channel this page last switched ON follows its spin box live, so
        the level can be dialled in while looking at the light."""
        if self._manual_on[channel] and self._svc.state.value == "connected":
            self._on_manual_on(channel)

    def _send_manual(self, what: str, send, commands: str) -> bool:
        """Run one manual command and log it; False on an LED fault.

        Fire-and-forget like every command except the raw tester (see
        ``core.led.led_manager``), so it never waits on the controller and
        is safe on the GUI thread.
        """
        self._log(f"TX -> {commands}   [{what}]", "tx")
        try:
            send()
        except VisionSystemError as exc:
            self._log(f"ERROR: {exc}", "error")
            self._set_controls_enabled(self._svc.state.value == "connected")
            return False
        return True

    def _mark_manual(self, channel: int, on: bool) -> None:
        self._manual_on[channel] = on
        if on:
            level = min(self._manual_spins[channel].value(), self._svc.max_brightness)
            self._manual_state[channel].setText(f"ON · {level}")
        else:
            self._manual_state[channel].setText("OFF")

    def _remember_manual_levels(self) -> None:
        levels = {channel: spin.value() for channel, spin in self._manual_spins.items()}
        try:
            self._svc.save_manual_levels(levels)
        except VisionSystemError as exc:
            self._log(f"Manual levels not saved: {exc}", "error")

    # --------------------------------------------------------- camera levels
    def _on_set_camera_brightness(self, index: int) -> None:
        spin = self._camera_spins.get(index)
        if spin is not None:
            self._apply_camera_levels({index: spin.value()})

    def _on_set_all_cameras(self) -> None:
        if not self._camera_spins:
            return
        value = self._all_cameras_spin.value()
        answer = QMessageBox.question(
            self,
            "Set All Cameras",
            f"Set the Light Brightness of all {len(self._camera_spins)} cameras "
            f"to {value}?\n\nIt is saved immediately and used from the next "
            "trigger cycle.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._apply_camera_levels({index: value for index in self._camera_spins})

    def _apply_camera_levels(self, levels: dict[int, int]) -> None:
        try:
            self._cameras.set_brightness(levels)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Camera Light Brightness", str(exc))
            return
        for index, value in sorted(levels.items()):
            self._log(f"Camera {index} light brightness set to {value} (saved)", "info")
        if self._svc.state.value != "connected":
            self._log(
                "LED controller not connected: steady lights were not updated now. "
                "Strobe cameras use the new level from the next trigger cycle.",
                "error",
            )

    # --------------------------------------------------------- camera strobe
    def _on_set_camera_strobe(self, index: int, strobe: bool) -> None:
        self._apply_camera_strobe({index: strobe})

    def _on_set_all_strobe(self, strobe: bool) -> None:
        if not self._camera_strobes:
            return
        verb = "Enable" if strobe else "Disable"
        answer = QMessageBox.question(
            self,
            f"{verb} Strobe",
            f"{verb} strobe on all {len(self._camera_strobes)} cameras?\n\n"
            + (
                "Each light will be lit only while its camera captures."
                if strobe
                else "Each light will be held steady at its camera's brightness."
            )
            + "\nIt is saved immediately and used from the next trigger cycle.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._apply_camera_strobe({index: strobe for index in self._camera_strobes})

    def _apply_camera_strobe(self, states: dict[int, bool]) -> None:
        try:
            self._cameras.set_strobe(states)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Camera Strobe", str(exc))
            self._refresh_cameras()  # put the checkboxes back to what is stored
            return
        for index, value in sorted(states.items()):
            self._log(
                f"Camera {index} strobe {'enabled' if value else 'disabled'} (saved)", "info"
            )
        if self._svc.state.value != "connected":
            self._log(
                "LED controller not connected: lights were not switched now. "
                "The new mode applies from the next trigger cycle.",
                "error",
            )
        self._refresh_cameras()  # wiring labels show strobe/steady

    # ---------------------------------------------------------------- raw
    def _on_raw_send(self) -> None:
        """Send the typed command on a worker thread and return immediately.

        This is the only LED command in the application that waits for the
        controller to answer (everything else is fire-and-forget — see
        ``core.led.led_manager``), so it is also the only one that could
        freeze the window while a mute controller runs down the timeout.
        The button is disabled until the reply or the timeout arrives, which
        both stops a queue of overlapping sends and shows the operator that
        something is in flight.
        """
        if self._raw_worker is not None:  # a send is already in flight
            return
        command = self._raw_command.text()
        if not command:
            return

        self._log(f"TX -> {command}", "tx")
        self._raw_response.setText("")
        self._raw_send_btn.setEnabled(False)

        worker = LedCommandWorker(
            self._svc.send_raw,
            command,
            append_terminator=self._raw_crlf.isChecked(),
            parent=self,
        )
        worker.succeeded.connect(self._on_raw_succeeded)
        worker.failed.connect(self._on_raw_failed)
        worker.finished.connect(self._on_raw_worker_finished)
        self._raw_worker = worker
        worker.start()

    def _on_raw_succeeded(self, response: str) -> None:
        self._raw_response.setText(response)
        self._log(f"RX <- {response}", "rx")

    def _on_raw_failed(self, message: str) -> None:
        """A timeout leaves the link up (only a broken port drops it), so the
        Connect/Disconnect buttons are refreshed from the service's *actual*
        state rather than assumed to be disconnected."""
        self._raw_response.setText("ERROR")
        self._log(f"ERROR: {message}", "error")
        self._set_controls_enabled(self._svc.state.value == "connected")

    def _on_raw_worker_finished(self) -> None:
        """Runs on either outcome, so the button always comes back."""
        self._raw_send_btn.setEnabled(True)
        if self._raw_worker is not None:
            self._raw_worker.deleteLater()
            self._raw_worker = None

    # -------------------------------------------------------------- logging
    def _log(self, message: str, kind: str) -> None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        self._entries.append((timestamp, message, kind))
        del self._entries[:-_LOG_LIMIT]
        self._log_view.append(self._format_entry(timestamp, message, kind))

    def _format_entry(self, timestamp: str, message: str, kind: str) -> str:
        colour = theme.color(_LOG_TOKENS.get(kind, "led-log-default"))
        return f'<span style="color:{colour}">{timestamp} {escape(message)}</span>'

    def _clear_log(self) -> None:
        """Clear the view *and* the transcript behind it.

        Both, or a theme switch would resurrect the lines the operator just
        cleared when :meth:`_repaint_log` re-renders from ``_entries``.
        """
        self._entries.clear()
        self._log_view.clear()

    def _repaint_log(self, _theme) -> None:
        """Re-render the transcript in the new scheme's colours.

        Registered with :func:`ui.theme.subscribe` because this page paints
        its log rather than styling it. The page is built once by the
        composition root and lives as long as the window, so it never
        unsubscribes.
        """
        self._log_view.clear()
        for timestamp, message, kind in self._entries:
            self._log_view.append(self._format_entry(timestamp, message, kind))

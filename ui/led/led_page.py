"""LED Controller page: connection settings (set up once) plus a raw command
tester and a communication log for troubleshooting the hardware directly.

Per-channel brightness is no longer driven from here: each camera's
"Light Brightness" and "LED Channel" (Camera Configuration page) are what
choose a channel and level, pushed automatically by
``CameraService._push_brightness`` whenever that camera's settings are
applied or saved. This page only owns the RS232 link itself — port, baud,
timeout, and the ``max_brightness`` safety ceiling every channel command
(wherever it originates) is clamped to.

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

from core.led import protocol
from core.utilities.exceptions import VisionSystemError
from models.app_state import AppState
from services.led_service import LedService
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
    """LED Controller (KDC-24V60W-4T) connection settings + raw command tester."""

    def __init__(
        self,
        app_state: AppState,
        led_service: LedService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._svc = led_service
        # (timestamp, message, kind) for every line on screen, so the log can
        # be re-rendered in the other scheme's colours on a theme switch.
        self._entries: list[tuple[str, str, str]] = []

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
            "Per-camera brightness and channel assignment live on the Camera "
            "Configuration page now — this page only sets up the RS232 link "
            "itself, once."
        )
        note.setWordWrap(True)
        note.setProperty("class", "dim")
        root.addWidget(note)

        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter, stretch=1)
        splitter.addWidget(self._build_connection_group())
        splitter.addWidget(self._build_raw_command_group())
        splitter.addWidget(self._build_log_group())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)

        app_state.led_state_changed.connect(self._on_led_state_changed)
        theme.subscribe(self._repaint_log)
        self._load()

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
        raw_send_btn = QPushButton("SEND")
        raw_send_btn.clicked.connect(self._on_raw_send)
        row.addWidget(raw_send_btn)
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

    # ---------------------------------------------------------------- actions
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

    # ---------------------------------------------------------------- raw
    def _on_raw_send(self) -> None:
        command = self._raw_command.text()
        if not command:
            return
        try:
            response = self._svc.send_raw(command, append_terminator=self._raw_crlf.isChecked())
        except VisionSystemError as exc:
            self._raw_response.setText("ERROR")
            self._log(f"ERROR: {exc}", "error")
            self._set_controls_enabled(False)
            return
        self._raw_response.setText(response)
        self._log(f"TX -> {command}", "tx")
        self._log(f"RX <- {response}", "rx")

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

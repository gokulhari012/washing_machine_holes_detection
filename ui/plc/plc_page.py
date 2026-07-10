"""PLC Configuration page.

Left: connection settings + full register map + scaling, with Test/Save/
Reconnect. Right: live register viewer (auto-refresh while the page is
visible) and manual read/write — writes require an admin login (Settings
page). Connection-level changes take effect after an application restart;
the register map is re-read per save by the composition root's subscription.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.utilities.exceptions import VisionSystemError
from models.app_state import AppState
from services.auth_service import AuthService
from services.plc_service import PlcService
from ui.widgets import LabeledLed

REFRESH_MS = 500


def _reg_spin(value: int = 0) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(0, 65535)
    spin.setValue(value)
    return spin


class PlcPage(QWidget):
    """Connection + register map + live viewer + manual access."""

    def __init__(
        self,
        app_state: AppState,
        plc_service: PlcService,
        auth_service: AuthService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._svc = plc_service
        self._auth = auth_service

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        title_row = QHBoxLayout()
        title = QLabel("PLC Configuration")
        title.setProperty("class", "pageTitle")
        self._state_led = LabeledLed("PLC")
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self._state_led)
        root.addLayout(title_row)

        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, stretch=1)

        # --------------------------------------------------- left: settings
        left = QVBoxLayout()

        conn_box = QGroupBox("Connection")
        conn_form = QFormLayout(conn_box)
        self._ip = QLineEdit()
        self._port = QSpinBox()
        self._port.setRange(1, 65535)
        self._protocol = QComboBox()
        self._protocol.addItems(["modbus_tcp", "simulated"])
        self._unit = QSpinBox()
        self._unit.setRange(0, 255)
        self._timeout = QSpinBox()
        self._timeout.setRange(100, 60000)
        self._timeout.setSuffix(" ms")
        self._poll = QSpinBox()
        self._poll.setRange(10, 5000)
        self._poll.setSuffix(" ms")
        conn_form.addRow("IP Address", self._ip)
        conn_form.addRow("Port", self._port)
        conn_form.addRow("Protocol", self._protocol)
        conn_form.addRow("Unit ID", self._unit)
        conn_form.addRow("Timeout", self._timeout)
        conn_form.addRow("Poll Interval", self._poll)
        left.addWidget(conn_box)

        reg_box = QGroupBox("Registers")
        reg_grid = QGridLayout(reg_box)
        self._reg = {
            "trigger": _reg_spin(),
            "machine_number": _reg_spin(),
            "heartbeat": _reg_spin(),
            "result": _reg_spin(),
            "vision_complete": _reg_spin(),
        }
        labels = ["Trigger", "Machine No", "Heartbeat", "Result", "Vision Complete"]
        for row, (key, label) in enumerate(zip(self._reg, labels)):
            reg_grid.addWidget(QLabel(label), row, 0)
            reg_grid.addWidget(self._reg[key], row, 1)
        self._cam_regs: dict[int, tuple[QSpinBox, QSpinBox]] = {}
        for position, camera in enumerate((1, 2, 3, 4)):
            row = 5 + position
            x_spin, y_spin = _reg_spin(), _reg_spin()
            self._cam_regs[camera] = (x_spin, y_spin)
            reg_grid.addWidget(QLabel(f"Camera {camera} X / Y"), row, 0)
            pair = QHBoxLayout()
            pair.addWidget(x_spin)
            pair.addWidget(y_spin)
            pair_w = QWidget()
            pair_w.setLayout(pair)
            reg_grid.addWidget(pair_w, row, 1)
        self._scale = _reg_spin(10)
        self._offset = _reg_spin(10000)
        reg_grid.addWidget(QLabel("Position Scale"), 9, 0)
        reg_grid.addWidget(self._scale, 9, 1)
        reg_grid.addWidget(QLabel("Position Offset"), 10, 0)
        reg_grid.addWidget(self._offset, 10, 1)
        left.addWidget(reg_box)

        buttons = QHBoxLayout()
        test_btn = QPushButton("Test Connection")
        test_btn.clicked.connect(self._on_test)
        reconnect_btn = QPushButton("Reconnect")
        reconnect_btn.clicked.connect(self._on_reconnect)
        save_btn = QPushButton("Save")
        save_btn.setProperty("class", "primary")
        save_btn.clicked.connect(self._on_save)
        buttons.addWidget(test_btn)
        buttons.addWidget(reconnect_btn)
        buttons.addWidget(save_btn)
        left.addLayout(buttons)
        note = QLabel("Connection changes apply after application restart.")
        note.setProperty("class", "dim")
        left.addWidget(note)
        left.addStretch()
        body.addLayout(left)

        # ------------------------------------------------ right: live viewer
        right = QVBoxLayout()

        viewer_box = QGroupBox("Live Register Viewer")
        viewer_layout = QVBoxLayout(viewer_box)
        controls = QHBoxLayout()
        self._view_start = _reg_spin(100)
        self._view_count = QSpinBox()
        self._view_count.setRange(1, 60)
        self._view_count.setValue(20)
        self._auto = QCheckBox("Auto refresh")
        self._auto.setChecked(True)
        read_btn = QPushButton("Read")
        read_btn.clicked.connect(self._refresh_viewer)
        controls.addWidget(QLabel("Start"))
        controls.addWidget(self._view_start)
        controls.addWidget(QLabel("Count"))
        controls.addWidget(self._view_count)
        controls.addWidget(self._auto)
        controls.addWidget(read_btn)
        controls.addStretch()
        viewer_layout.addLayout(controls)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Register", "Value"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        viewer_layout.addWidget(self._table, stretch=1)
        right.addWidget(viewer_box, stretch=1)

        manual_box = QGroupBox("Manual Write (admin)")
        manual = QHBoxLayout(manual_box)
        self._write_addr = _reg_spin(119)
        self._write_value = _reg_spin(0)
        write_btn = QPushButton("Write")
        write_btn.setProperty("class", "danger")
        write_btn.clicked.connect(self._on_manual_write)
        manual.addWidget(QLabel("Register"))
        manual.addWidget(self._write_addr)
        manual.addWidget(QLabel("Value"))
        manual.addWidget(self._write_value)
        manual.addWidget(write_btn)
        manual.addStretch()
        right.addWidget(manual_box)
        body.addLayout(right, stretch=1)

        # ---------------------------------------------------------- wiring
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._maybe_refresh)
        app_state.plc_state_changed.connect(
            lambda value: self._state_led.set_state(value, f"PLC {value}")
        )
        self._load()

    # ---------------------------------------------------------------- load
    def _load(self) -> None:
        cfg = self._svc.get_config()
        connection = cfg.get("connection", {})
        registers = cfg.get("registers", {})
        scaling = cfg.get("scaling", {})
        self._ip.setText(connection.get("ip", ""))
        self._port.setValue(int(connection.get("port", 502)))
        self._protocol.setCurrentText(connection.get("protocol", "modbus_tcp"))
        self._unit.setValue(int(connection.get("unit_id", 1)))
        self._timeout.setValue(int(connection.get("timeout_ms", 1000)))
        self._poll.setValue(int(connection.get("poll_interval_ms", 50)))
        for key, spin in self._reg.items():
            spin.setValue(int(registers.get(key, 0)))
        for camera, (x_spin, y_spin) in self._cam_regs.items():
            addresses = registers.get("camera_positions", {}).get(str(camera), {})
            x_spin.setValue(int(addresses.get("x", 0)))
            y_spin.setValue(int(addresses.get("y", 0)))
        self._scale.setValue(int(scaling.get("position_scale", 10)))
        self._offset.setValue(int(scaling.get("position_offset", 10000)))
        self._state_led.set_state(self._svc.state, f"PLC {self._svc.state.value}")

    def _collect(self) -> dict:
        cfg = self._svc.get_config()
        cfg["connection"] = {
            **cfg.get("connection", {}),
            "ip": self._ip.text().strip(),
            "port": self._port.value(),
            "protocol": self._protocol.currentText(),
            "unit_id": self._unit.value(),
            "timeout_ms": self._timeout.value(),
            "poll_interval_ms": self._poll.value(),
        }
        cfg["registers"] = {
            **{key: spin.value() for key, spin in self._reg.items()},
            "camera_positions": {
                str(camera): {"x": x_spin.value(), "y": y_spin.value()}
                for camera, (x_spin, y_spin) in self._cam_regs.items()
            },
        }
        cfg["scaling"] = {
            "position_scale": self._scale.value(),
            "position_offset": self._offset.value(),
        }
        return cfg

    # -------------------------------------------------------------- actions
    def _on_test(self) -> None:
        ok, message = self._svc.test_connection(self._collect())
        if ok:
            QMessageBox.information(self, "Test Connection", message)
        else:
            QMessageBox.warning(self, "Test Connection", message)

    def _on_reconnect(self) -> None:
        self._svc.request_reconnect()

    def _on_save(self) -> None:
        try:
            self._svc.save_config(self._collect())
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Save", str(exc))
            return
        QMessageBox.information(
            self, "Save", "PLC configuration saved.\nConnection changes apply after restart."
        )

    def _on_manual_write(self) -> None:
        if not self._auth.is_admin:
            QMessageBox.warning(
                self, "Manual Write", "Administrator login required (Settings page)."
            )
            return
        try:
            self._svc.write_register(self._write_addr.value(), self._write_value.value())
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Manual Write", str(exc))

    # ------------------------------------------------------------- viewer
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._timer.start(REFRESH_MS)

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self._timer.stop()

    def _maybe_refresh(self) -> None:
        if self._auto.isChecked():
            self._refresh_viewer()

    def _refresh_viewer(self) -> None:
        start = self._view_start.value()
        count = self._view_count.value()
        try:
            values = self._svc.read_register(start, count)
        except VisionSystemError:
            values = None  # link down — keep last values, LED shows the state
        if values is None:
            return
        self._table.setRowCount(len(values))
        for row, value in enumerate(values):
            self._table.setItem(row, 0, QTableWidgetItem(str(start + row)))
            self._table.setItem(row, 1, QTableWidgetItem(str(value)))

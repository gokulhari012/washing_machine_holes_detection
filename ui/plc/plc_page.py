"""PLC Configuration page.

Left: connection settings + the position-scale scalar (not a register — it
has no PLC address of its own), with Test/Save/Reconnect. Right: one
table listing every named PLC register this application knows about — its
configured address (an embedded, editable spin box) and its current live
value (auto-refreshed while the page is visible). Saving connection or
register-address changes takes effect immediately, without an application
restart: ``Application._on_plc_config_saved`` (main.py) stops the poll
worker, rebuilds ``PlcManager`` in place via ``PlcManager.rebuild`` — every
other holder of that instance (PlcService, InspectionService) keeps working
unchanged — and starts a fresh poll worker sized to the new intervals. The
table itself just replaces what used to be two separate register grids plus
a second, differently-addressed live viewer table. Manual write of an
arbitrary register/value stays a separate, explicitly admin-gated control
below the table — pushing a value to a *live* register right now is a
different, more dangerous action than editing which address a name refers
to, so it keeps its own confirmation path.

"Pause Communication" suspends every outgoing register/coil write except the
heartbeat (see ``PlcManager.pause``) — reads and the inspection pipeline keep
running, but the PLC never sees vision_complete or a cleared trigger while
paused, so the line dead-waits exactly as if stopped, without the link itself
dropping. A red "PAUSED" badge next to the state LED mirrors
``AppState.plc_paused_changed`` so the page reflects a pause requested from
anywhere, not just its own checkbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
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
CAMERAS = (1, 2, 3, 4)
# Two register addresses are read in the same bulk transaction when they are
# at most this far apart — cheaper than a second PLC round trip for the
# handful of addresses in between that no named register actually uses.
MAX_READ_GAP = 4


def _reg_spin(value: int = 0) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(0, 65535)
    spin.setValue(value)
    return spin


@dataclass
class _RegisterField:
    """One row of the register table: display metadata plus how its
    configured address is read from and written into the plc.json document."""

    name: str
    tooltip: str
    optional: bool  # True: 0 displays as "Not used" and is saved as absent
    get: Callable[[dict], int]
    set: Callable[[dict, int], None]
    clear: Callable[[dict], None] | None = None  # only ever called when optional
    kind: str = "holding"  # "holding" or "coil" — a completely separate address space


def _core_field(key: str, name: str, tooltip: str) -> _RegisterField:
    return _RegisterField(
        name, tooltip, False,
        get=lambda cfg, key=key: int(cfg["registers"].get(key, 0)),
        set=lambda cfg, v, key=key: cfg["registers"].__setitem__(key, v),
    )


def _camera_pair_field(block: str, camera: int, axis: str, name: str, tooltip: str) -> _RegisterField:
    idx = str(camera)
    return _RegisterField(
        name, tooltip, False,
        get=lambda cfg, block=block, idx=idx, axis=axis: int(
            cfg["registers"].get(block, {}).get(idx, {}).get(axis, 0)
        ),
        set=lambda cfg, v, block=block, idx=idx, axis=axis: (
            cfg["registers"].setdefault(block, {}).setdefault(idx, {}).__setitem__(axis, v)
        ),
    )


def _camera_scalar_field(
    block: str, camera: int, name: str, tooltip: str, optional: bool = False
) -> _RegisterField:
    """One camera's entry in a ``registers.<block>`` mapping.

    *optional* makes 0 display as "Not used" and drop that camera out of the
    block on save, so the feature can be left unwired for a camera the same
    way ``model_select`` can be left unwired entirely.
    """
    idx = str(camera)
    return _RegisterField(
        name, tooltip, optional,
        get=lambda cfg, block=block, idx=idx: int(cfg["registers"].get(block, {}).get(idx, 0)),
        set=lambda cfg, v, block=block, idx=idx: (
            cfg["registers"].setdefault(block, {}).__setitem__(idx, v)
        ),
        clear=(
            lambda cfg, block=block, idx=idx: cfg["registers"].get(block, {}).pop(idx, None)
        )
        if optional
        else None,
    )


def _model_select_field() -> _RegisterField:
    return _RegisterField(
        "Machine Model Select",
        "PLC → PC: machine-model code, polled every model_poll_interval_ms. "
        "The profile whose PLC Code matches the value read here is applied "
        "(Machine Models page). \"Not used\" disables model switching — the "
        "register is then never read.",
        True,
        get=lambda cfg: int(cfg["registers"].get("model_select") or 0),
        set=lambda cfg, v: cfg["registers"].__setitem__("model_select", v),
        clear=lambda cfg: cfg["registers"].__setitem__("model_select", None),
    )


def _serial_number_field() -> _RegisterField:
    return _RegisterField(
        "Serial Number",
        "PLC → PC: serial number of the machine being inspected, read at the "
        "start of each cycle. The Serial Prefix from the Settings page is "
        "added in front of it on the PC. \"Not used\" falls back to the "
        "machine number, as before this register existed.",
        True,
        get=lambda cfg: int(cfg["registers"].get("serial_number") or 0),
        set=lambda cfg, v: cfg["registers"].__setitem__("serial_number", v),
        clear=lambda cfg: cfg["registers"].__setitem__("serial_number", None),
    )


def _build_fields() -> list[_RegisterField]:
    fields = [
        _core_field(
            "trigger", "Trigger",
            "PLC → PC: rising edge starts an inspection cycle; PC writes 0 back on detection",
        ),
        _core_field(
            "machine_number", "Machine Number",
            "PLC → PC: identifies the machine being inspected this cycle",
        ),
        _serial_number_field(),
        _core_field(
            "heartbeat", "Heartbeat",
            "PC → PLC: toggles every heartbeat_interval_ms so the PLC can watchdog the PC",
        ),
        _model_select_field(),
        _core_field(
            "result", "Result",
            "PC → PLC: overall cycle result — 1=GOOD, 2=NG, 3=ERROR",
        ),
        _core_field(
            "vision_complete", "Vision Complete",
            "PC → PLC: set to 1 once results are ready; PLC reads it then resets it and the trigger",
        ),
    ]
    for camera in CAMERAS:
        fields += [
            _camera_pair_field(
                "camera_positions", camera, "x", f"Camera {camera} X",
                "PC → PLC: detected hole X, encoded as an absolute servo target",
            ),
            _camera_pair_field(
                "camera_positions", camera, "y", f"Camera {camera} Y",
                "PC → PLC: detected hole Y, encoded as an absolute servo target",
            ),
            _camera_scalar_field(
                "camera_results", camera, f"Camera {camera} Result",
                "PC → PLC: this camera's own GOOD/NG/ERROR verdict",
            ),
            _camera_pair_field(
                "servo_home_positions", camera, "x", f"Camera {camera} Servo Home X",
                "PLC → PC: this axis's servo home — the datum camera positions are measured from",
            ),
            _camera_pair_field(
                "servo_home_positions", camera, "y", f"Camera {camera} Servo Home Y",
                "PLC → PC: this axis's servo home — the datum camera positions are measured from",
            ),
            _camera_scalar_field(
                "camera_triggers", camera, f"Camera {camera} Trigger",
                "PLC → PC: inspect this camera alone on a 0→1 edge",
            ),
            _camera_scalar_field(
                "camera_vision_complete", camera, f"Camera {camera} Vision Complete",
                "PC → PLC: this camera's own completion handshake",
            ),
            _camera_scalar_field(
                "camera_status", camera, f"Camera {camera} Status",
                "PC → PLC: 1 while this camera is connected and grabbing normally, 0 otherwise",
            ),
            _camera_scalar_field(
                "camera_brightness", camera, f"Camera {camera} Brightness",
                "PC → PLC: this camera's light-brightness level (0-255), pushed on every "
                "Camera page Apply/Save — drives an external light, not the camera itself",
            ),
            _camera_scalar_field(
                "gantry_status", camera, f"Camera {camera} Gantry Status", (
                    "PLC → PC: 1 = this camera's gantry is in position, so the camera "
                    "takes part in the cycle. Anything else skips it — no capture, no "
                    "detection, and its position/result registers are left holding "
                    "whatever the last cycle that really inspected it wrote. Read at "
                    "the start of every cycle, global and per-camera alike. \"Not used\" "
                    "means this camera is always inspected, as before this register "
                    "existed."
                ),
                optional=True,
            ),
        ]
    return fields


def _read_clusters(addresses: list[int]) -> list[tuple[int, int]]:
    """Group sorted, de-duplicated addresses into (start, count) spans,
    merging two addresses into one bulk read whenever the gap between them
    is small — cheaper than a second PLC round trip for the handful of
    in-between addresses no named register uses."""
    if not addresses:
        return []
    ordered = sorted(set(addresses))
    clusters: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for address in ordered[1:]:
        if address - prev <= MAX_READ_GAP:
            prev = address
            continue
        clusters.append((start, prev - start + 1))
        start = prev = address
    clusters.append((start, prev - start + 1))
    return clusters


class PlcPage(QWidget):
    """Connection + unified register table (config + live monitor) + manual access."""

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
        self._fields = _build_fields()

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        title_row = QHBoxLayout()
        title = QLabel("PLC Configuration")
        title.setProperty("class", "pageTitle")
        self._state_led = LabeledLed("PLC")
        self._paused_label = QLabel("PAUSED")
        self._paused_label.setProperty("class", "danger")
        self._paused_label.setVisible(False)
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self._paused_label)
        title_row.addWidget(self._state_led)
        root.addLayout(title_row)

        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, stretch=1)

        # --------------------------------------------------- left: settings
        left_container = QWidget()
        left_container.setMaximumWidth(420)
        left = QVBoxLayout(left_container)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)

        conn_box = QGroupBox("Connection")
        conn_form = QFormLayout(conn_box)
        self._ip = QLineEdit()
        self._port = QSpinBox()
        self._port.setRange(1, 65535)
        self._protocol = QComboBox()
        self._protocol.addItems(["modbus_tcp", "slmp", "simulated"])
        self._protocol.currentTextChanged.connect(self._on_protocol_changed)
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

        pause_box = QGroupBox("Communication")
        pause_layout = QVBoxLayout(pause_box)
        self._pause_checkbox = QCheckBox("Pause Communication")
        self._pause_checkbox.setToolTip(
            "While paused, no register or coil write is sent to the PLC "
            "except the heartbeat — triggers are not acknowledged, no "
            "inspection output is written, and vision_complete never goes "
            "high, so the PLC dead-waits exactly as if the line were "
            "stopped. The heartbeat keeps toggling so the PLC's watchdog "
            "does not trip the link."
        )
        self._pause_checkbox.toggled.connect(self._on_pause_toggled)
        pause_layout.addWidget(self._pause_checkbox)
        left.addWidget(pause_box)

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
        left.addWidget(manual_box)

        coil_box = QGroupBox("Manual Coil Write (admin)")
        coil = QHBoxLayout(coil_box)
        self._write_coil_addr = _reg_spin(1)
        coil_zero_btn = QPushButton("0")
        coil_zero_btn.setProperty("class", "danger")
        coil_zero_btn.clicked.connect(lambda: self._on_manual_coil_write(False))
        coil_one_btn = QPushButton("1")
        coil_one_btn.setProperty("class", "danger")
        coil_one_btn.clicked.connect(lambda: self._on_manual_coil_write(True))
        coil.addWidget(QLabel("Coil"))
        coil.addWidget(self._write_coil_addr)
        coil.addWidget(coil_zero_btn)
        coil.addWidget(coil_one_btn)
        coil.addStretch()
        left.addWidget(coil_box)

        scale_box = QGroupBox("Scaling")
        scale_form = QFormLayout(scale_box)
        self._scale = _reg_spin(10)
        self._scale.setToolTip(
            "Millimetres are multiplied by this before being added to the "
            "servo home position (10 = one decimal place)"
        )
        scale_form.addRow("Position Scale", self._scale)
        left.addWidget(scale_box)

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
        note = QLabel(
            "Saving reconnects the PLC with the new connection and register "
            "addresses immediately — no application restart needed. Any "
            "cycle mid-flight at that instant may report ERROR once."
        )
        note.setWordWrap(True)
        note.setProperty("class", "dim")
        left.addWidget(note)
        left.addStretch()
        body.addWidget(left_container)

        # ------------------------------------------------- right: registers
        right = QVBoxLayout()

        table_box = QGroupBox("Registers")
        table_layout = QVBoxLayout(table_box)
        controls = QHBoxLayout()
        self._auto = QCheckBox("Auto refresh")
        self._auto.setChecked(True)
        read_btn = QPushButton("Read Now")
        read_btn.clicked.connect(self._refresh_viewer)
        controls.addWidget(self._auto)
        controls.addWidget(read_btn)
        controls.addStretch()
        table_layout.addLayout(controls)

        self._table = QTableWidget(len(self._fields), 4)
        self._table.setHorizontalHeaderLabels(
            ["Register Name", "Type", "Register Number", "Live Value"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 80)
        header.resizeSection(2, 130)
        header.resizeSection(3, 100)

        self._row_spins: list[QSpinBox] = []
        for row, field in enumerate(self._fields):
            name_item = QTableWidgetItem(field.name)
            name_item.setToolTip(field.tooltip)
            self._table.setItem(row, 0, name_item)
            type_item = QTableWidgetItem("Coil" if field.kind == "coil" else "Holding")
            type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 1, type_item)
            spin = _reg_spin()
            spin.setMinimumWidth(110)
            if field.optional:
                spin.setSpecialValueText("Not used")
            spin.setToolTip(field.tooltip)
            self._table.setCellWidget(row, 2, spin)
            self._row_spins.append(spin)
            value_item = QTableWidgetItem("—")
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 3, value_item)
        table_layout.addWidget(self._table, stretch=1)
        right.addWidget(table_box, stretch=1)
        body.addLayout(right, stretch=1)

        # ---------------------------------------------------------- wiring
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._maybe_refresh)
        app_state.plc_state_changed.connect(
            lambda value: self._state_led.set_state(value, f"PLC {value}")
        )
        app_state.plc_paused_changed.connect(self._on_plc_paused_changed)
        self._load()
        self._on_plc_paused_changed(self._svc.paused)

    # ---------------------------------------------------------------- load
    def _load(self) -> None:
        cfg = self._svc.get_config()
        connection = cfg.get("connection", {})
        scaling = cfg.get("scaling", {})
        self._ip.setText(connection.get("ip", ""))
        self._port.setValue(int(connection.get("port", 502)))
        self._protocol.setCurrentText(connection.get("protocol", "modbus_tcp"))
        self._unit.setValue(int(connection.get("unit_id", 1)))
        self._timeout.setValue(int(connection.get("timeout_ms", 1000)))
        self._poll.setValue(int(connection.get("poll_interval_ms", 50)))
        self._scale.setValue(int(scaling.get("position_scale", 10)))
        for field, spin in zip(self._fields, self._row_spins):
            spin.setValue(int(field.get(cfg) or 0))
        self._on_protocol_changed(self._protocol.currentText())
        self._state_led.set_state(self._svc.state, f"PLC {self._svc.state.value}")

    def _on_protocol_changed(self, protocol: str) -> None:
        """Unit ID is a Modbus concept; SLMP addresses the CPU by network/station."""
        self._unit.setEnabled(protocol == "modbus_tcp")

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
        for field, spin in zip(self._fields, self._row_spins):
            value = spin.value()
            if field.optional and value == 0:
                if field.clear is not None:
                    field.clear(cfg)
                continue
            field.set(cfg, value)
        cfg["scaling"] = {"position_scale": self._scale.value()}
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

    def _on_pause_toggled(self, checked: bool) -> None:
        self._svc.set_paused(checked)

    def _on_plc_paused_changed(self, paused: bool) -> None:
        self._pause_checkbox.blockSignals(True)
        self._pause_checkbox.setChecked(paused)
        self._pause_checkbox.blockSignals(False)
        self._paused_label.setVisible(paused)

    def _on_save(self) -> None:
        try:
            self._svc.save_config(self._collect())
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Save", str(exc))
            return
        QMessageBox.information(
            self,
            "Save",
            "PLC configuration saved. Reconnecting with the new settings now.",
        )

    def _on_manual_write(self) -> None:
        if not self._auth.is_admin:
            QMessageBox.warning(
                self,
                "Manual Write",
                "Administrator or developer login required (toolbar Login button).",
            )
            return
        try:
            self._svc.write_register(self._write_addr.value(), self._write_value.value())
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Manual Write", str(exc))

    def _on_manual_coil_write(self, value: bool) -> None:
        if not self._auth.is_admin:
            QMessageBox.warning(
                self,
                "Manual Coil Write",
                "Administrator or developer login required (toolbar Login button).",
            )
            return
        try:
            self._svc.write_coil(self._write_coil_addr.value(), value)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Manual Coil Write", str(exc))

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
        # Holding registers and coils are separate address spaces (see
        # core.plc.plc_client_base) — address 5 means different physical
        # memory in each, so they must never be clustered/read together.
        holding_rows: dict[int, list[int]] = {}
        coil_rows: dict[int, list[int]] = {}
        for row, (field, spin) in enumerate(zip(self._fields, self._row_spins)):
            address = spin.value()
            if field.optional and address == 0:
                self._table.item(row, 3).setText("—")
                continue
            group = coil_rows if field.kind == "coil" else holding_rows
            group.setdefault(address, []).append(row)

        self._refresh_group(holding_rows, self._svc.read_register)
        self._refresh_group(coil_rows, self._svc.read_coil)

    def _refresh_group(
        self, address_rows: dict[int, list[int]], reader: Callable[[int, int], list]
    ) -> None:
        """Bulk-read one address space's rows (clustered to cut round trips)
        and write the results into the Live Value column. Silently keeps the
        last displayed values on a comm failure — the LED already shows link
        state, and one address space being briefly down shouldn't blank out
        the other's rows too."""
        values: dict[int, object] = {}
        try:
            for start, count in _read_clusters(list(address_rows)):
                block = reader(start, count)
                for offset, value in enumerate(block):
                    values[start + offset] = value
        except VisionSystemError:
            return

        for address, rows in address_rows.items():
            value = values.get(address)
            if value is None:
                continue
            for row in rows:
                self._table.item(row, 3).setText(str(int(value)))

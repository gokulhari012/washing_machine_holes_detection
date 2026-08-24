"""Dashboard: manual trigger bar, stat tiles, 2×2 camera grid, recent history.

Entirely event-driven off :class:`AppState` signals — the page performs no
polling and touches no hardware. Initial values (counters, history) come from
the database once at construction.

The trigger bar drives the manual cycle: "Simulate Trigger" runs one complete
inspection, and the delay spin box sets how long the pipeline waits between
cameras when the sequential capture mode is on. The delay is persisted to
app_config.json immediately, so the next cycle — and the next start of the
application — uses it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.logging import get_logger
from core.utilities import ConfigManager
from core.utilities.enums import InspectionResult, LogSource
from core.utilities.exceptions import ConfigurationError
from models.app_state import AppState
from models.dto import CameraInspectionData, InspectionCycleData
from services.database_service import DatabaseService
from services.inspection_service import DEFAULT_CAMERA_DELAY_MS, SEQUENTIAL_MODE
from ui.dashboard.camera_panel import CameraPanel
from ui.theme import COLOR_ACCENT, COLOR_DIM, COLOR_GOOD, COLOR_NG, COLOR_WARN
from ui.widgets import StatTile

logger = get_logger(LogSource.UI)

HISTORY_LIMIT = 50
MAX_CAMERA_DELAY_MS = 60000

_RESULT_COLORS = {
    InspectionResult.GOOD.value: COLOR_GOOD,
    InspectionResult.NG.value: COLOR_NG,
    InspectionResult.ERROR.value: COLOR_WARN,
}


class DashboardPage(QWidget):
    """The operator's main screen."""

    def __init__(
        self,
        app_state: AppState,
        database_service: DatabaseService,
        camera_configs: list[dict],
        config_manager: ConfigManager | None = None,
        on_simulate_trigger=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._app_state = app_state
        self._database = database_service
        self._config = config_manager
        self._on_simulate_trigger = on_simulate_trigger
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(10)

        # ------------------------------------------------------ trigger bar
        header = QHBoxLayout()
        title = QLabel("Dashboard")
        title.setProperty("class", "pageTitle")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self._build_trigger_controls())
        root.addLayout(header)

        # ------------------------------------------------------- stat tiles
        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        self._tile_serial = StatTile("Serial Number")
        self._tile_status = StatTile("Status", "IDLE", accent=COLOR_DIM)
        self._tile_total = StatTile("Product Count", "0")
        self._tile_good = StatTile("Good", "0", accent=COLOR_GOOD)
        self._tile_ng = StatTile("NG", "0", accent=COLOR_NG)
        self._tile_trigger = StatTile("Last Trigger")
        for tile in (
            self._tile_serial,
            self._tile_status,
            self._tile_total,
            self._tile_good,
            self._tile_ng,
            self._tile_trigger,
        ):
            tiles.addWidget(tile)
        root.addLayout(tiles)

        # ------------------------------------------------- 2×2 camera grid
        grid = QGridLayout()
        grid.setSpacing(10)
        self._panels: dict[int, CameraPanel] = {}
        enabled = [cfg for cfg in camera_configs if cfg.get("enabled", True)][:4]
        for position, cfg in enumerate(sorted(enabled, key=lambda c: int(c["index"]))):
            panel = CameraPanel(int(cfg["index"]), str(cfg.get("name", f"Camera {cfg['index']}")))
            self._panels[int(cfg["index"])] = panel
            grid.addWidget(panel, position // 2, position % 2)
        root.addLayout(grid, stretch=1)

        # ------------------------------------------------------ history table
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["S.No", "Time", "Machine Number", "Result", "Cycle (ms)"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setMaximumHeight(190)
        root.addWidget(self._table)

        # --------------------------------------------------------- wiring
        app_state.preview_frame.connect(self._on_preview)
        app_state.camera_captured.connect(self._on_camera_captured)
        app_state.camera_inspected.connect(self._on_camera_inspected)
        app_state.camera_state_changed.connect(self._on_camera_state)
        app_state.trigger_received.connect(self._on_trigger)
        app_state.inspection_completed.connect(self._on_inspection)
        app_state.counters_changed.connect(self._on_counters)

        self._load_initial()

    # ------------------------------------------------------- trigger bar
    def _build_trigger_controls(self) -> QWidget:
        """Manual trigger button + the delay between cameras."""
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        caption = QLabel("Delay between cameras")
        caption.setProperty("class", "dim")
        layout.addWidget(caption)

        self._delay = QSpinBox()
        self._delay.setRange(0, MAX_CAMERA_DELAY_MS)
        self._delay.setSingleStep(100)
        self._delay.setSuffix(" ms")
        self._delay.setToolTip(
            "Pause between one camera finishing and the next one taking its picture"
        )
        self._delay.setValue(self._stored_delay_ms())
        self._delay.valueChanged.connect(self._on_delay_changed)
        layout.addWidget(self._delay)

        self._simulate_button = QPushButton("▶  Simulate Trigger")
        self._simulate_button.setProperty("class", "primary")
        self._simulate_button.setToolTip(
            "Run one inspection: every enabled camera takes a picture in turn"
        )
        if self._on_simulate_trigger is None:
            self._simulate_button.setEnabled(False)
        else:
            self._simulate_button.clicked.connect(self._on_simulate_clicked)
        layout.addWidget(self._simulate_button)
        return bar

    def _stored_delay_ms(self) -> int:
        """Configured inter-camera delay, or the pipeline default."""
        if self._config is None:
            return DEFAULT_CAMERA_DELAY_MS
        try:
            value = self._config.get_value(
                "app_config", "inspection.camera_delay_ms", DEFAULT_CAMERA_DELAY_MS
            )
            return max(0, min(int(value), MAX_CAMERA_DELAY_MS))
        except (ConfigurationError, TypeError, ValueError):
            return DEFAULT_CAMERA_DELAY_MS

    def _on_delay_changed(self, value: int) -> None:
        """Persist the delay so the running pipeline picks it up next cycle."""
        if self._config is None or self._loading:
            return
        try:
            document = self._config.load("app_config")
            inspection = document.setdefault("inspection", {})
            inspection["camera_delay_ms"] = int(value)
            inspection.setdefault("capture_mode", SEQUENTIAL_MODE)
            self._config.save("app_config", document)
        except ConfigurationError as exc:
            logger.error("Could not save the camera delay: %s", exc)
            self._app_state.raise_alarm("warning", f"Could not save the camera delay: {exc}")

    def _on_simulate_clicked(self) -> None:
        self._simulate_button.setEnabled(False)  # re-enabled when the cycle ends
        self._arm_button_watchdog()
        self._on_simulate_trigger()

    def _arm_button_watchdog(self) -> None:
        """Never leave the operator without a button if a cycle dies silently."""
        budget_ms = 60000 + len(self._panels) * self._delay.value()
        QTimer.singleShot(
            budget_ms,
            lambda: self._simulate_button.setEnabled(self._on_simulate_trigger is not None),
        )

    # ------------------------------------------------------------- initial
    def _load_initial(self) -> None:
        total, good, ng = self._app_state.counters
        self._on_counters(total, good, ng)
        for row in reversed(self._database.inspections.get_recent(HISTORY_LIMIT)):
            self._insert_history_row(
                row.id,
                row.created_at.strftime("%H:%M:%S"),
                row.serial_number or str(row.machine_number),
                row.overall_result,
                f"{row.plc_cycle_time_ms:.0f}",
            )

    # ----------------------------------------------------------------- slots
    def _on_preview(self, camera_index: int, frame) -> None:
        panel = self._panels.get(camera_index)
        if panel is not None:
            panel.update_preview(frame)

    def _on_camera_captured(self, camera_index: int, frame) -> None:
        """A camera just took its picture — show it before detection runs."""
        panel = self._panels.get(camera_index)
        if panel is not None:
            panel.show_capture(frame)

    def _on_camera_inspected(self, camera_index: int, data: CameraInspectionData) -> None:
        """A camera has been judged, mid-cycle."""
        panel = self._panels.get(camera_index)
        if panel is not None:
            panel.show_result(data)

    def _on_camera_state(self, camera_index: int, state: str) -> None:
        panel = self._panels.get(camera_index)
        if panel is not None:
            panel.set_camera_state(state)

    def _on_trigger(self, machine_number: int) -> None:
        self._tile_status.set_value("RUNNING")
        self._tile_status.set_accent(COLOR_ACCENT)
        self._tile_serial.set_value(str(machine_number))
        self._simulate_button.setEnabled(False)  # one cycle at a time

    def _on_inspection(self, cycle: InspectionCycleData) -> None:
        self._simulate_button.setEnabled(self._on_simulate_trigger is not None)
        result = cycle.overall_result.value
        self._tile_status.set_value(result)
        self._tile_status.set_accent(_RESULT_COLORS.get(result, COLOR_DIM))
        self._tile_serial.set_value(cycle.serial_number)
        self._tile_trigger.set_value(cycle.started_at.strftime("%H:%M:%S"))

        for camera_index, data in cycle.cameras.items():
            panel = self._panels.get(camera_index)
            if panel is not None:
                panel.show_result(data)

        self._insert_history_row(
            cycle.inspection_id if cycle.inspection_id is not None else "—",
            cycle.started_at.strftime("%H:%M:%S"),
            cycle.serial_number,
            result,
            f"{cycle.plc_cycle_time_ms:.0f}",
        )

    def _on_counters(self, total: int, good: int, ng: int) -> None:
        self._tile_total.set_value(total)
        self._tile_good.set_value(good)
        self._tile_ng.set_value(ng)

    # ------------------------------------------------------------- internal
    def _insert_history_row(
        self, serial_no, time_text: str, machine: str, result: str, cycle_ms: str
    ) -> None:
        self._table.insertRow(0)
        for column, value in enumerate(
            (str(serial_no), time_text, machine, result, cycle_ms)
        ):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if column == 3:
                item.setForeground(QColor(_RESULT_COLORS.get(result, COLOR_DIM)))
            self._table.setItem(0, column, item)
        while self._table.rowCount() > HISTORY_LIMIT:
            self._table.removeRow(self._table.rowCount() - 1)

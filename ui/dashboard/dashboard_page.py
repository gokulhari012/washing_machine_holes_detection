"""Dashboard: stat tiles, 2×2 live camera grid, recent inspection history.

Entirely event-driven off :class:`AppState` signals — the page performs no
polling and touches no hardware. Initial values (counters, history) come from
the database once at construction.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.utilities.enums import InspectionResult
from models.app_state import AppState
from models.dto import InspectionCycleData
from services.database_service import DatabaseService
from ui.dashboard.camera_panel import CameraPanel
from ui.theme import COLOR_ACCENT, COLOR_DIM, COLOR_GOOD, COLOR_NG, COLOR_WARN
from ui.widgets import StatTile

HISTORY_LIMIT = 50

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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._app_state = app_state
        self._database = database_service

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(10)

        title = QLabel("Dashboard")
        title.setProperty("class", "pageTitle")
        root.addWidget(title)

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
        app_state.camera_state_changed.connect(self._on_camera_state)
        app_state.trigger_received.connect(self._on_trigger)
        app_state.inspection_completed.connect(self._on_inspection)
        app_state.counters_changed.connect(self._on_counters)

        self._load_initial()

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

    def _on_camera_state(self, camera_index: int, state: str) -> None:
        panel = self._panels.get(camera_index)
        if panel is not None:
            panel.set_camera_state(state)

    def _on_trigger(self, machine_number: int) -> None:
        self._tile_status.set_value("RUNNING")
        self._tile_status.set_accent(COLOR_ACCENT)
        self._tile_serial.set_value(str(machine_number))

    def _on_inspection(self, cycle: InspectionCycleData) -> None:
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

"""Database Viewer: searchable, paginated inspection history with export.

Filters: quick ranges (Today/Yesterday/7 days/All/Custom), machine number,
result, serial substring. Pagination is server-side (repository offset/limit)
so the page stays fast at millions of rows; sorting within the loaded page is
client-side via the table headers. Export writes the **entire current filter**
(capped) to CSV/Excel/PDF.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.database import Inspection, InspectionFilter
from core.utilities.exceptions import VisionSystemError
from services.database_service import DatabaseService
from services.export_service import ExportService
from ui.theme import COLOR_DIM, COLOR_GOOD, COLOR_NG, COLOR_WARN

PAGE_SIZE = 50
EXPORT_CAP = 5000

_RESULT_COLORS = {"GOOD": COLOR_GOOD, "NG": COLOR_NG, "ERROR": COLOR_WARN}

COLUMNS = [
    "ID", "Date", "Time", "Machine", "Serial",
    "Cam1 (mm)", "Cam2 (mm)", "Cam3 (mm)", "Cam4 (mm)",
    "Overall", "Cycle (ms)", "Detect (ms)", "Operator",
]


class DatabasePage(QWidget):
    """Inspection history browser."""

    def __init__(
        self,
        database_service: DatabaseService,
        export_service: ExportService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._db = database_service
        self._export = export_service
        self._page = 0
        self._total = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        title = QLabel("Database Viewer")
        title.setProperty("class", "pageTitle")
        root.addWidget(title)

        # ------------------------------------------------------- filter bar
        filters = QHBoxLayout()
        self._quick = QComboBox()
        self._quick.addItems(["Today", "Yesterday", "Last 7 days", "All", "Custom"])
        self._quick.setCurrentText("All")
        self._quick.currentTextChanged.connect(self._on_quick_changed)
        self._date_from = QDateEdit(date.today())
        self._date_from.setCalendarPopup(True)
        self._date_to = QDateEdit(date.today())
        self._date_to.setCalendarPopup(True)
        self._machine = QLineEdit()
        self._machine.setPlaceholderText("Machine No")
        self._machine.setFixedWidth(100)
        self._result = QComboBox()
        self._result.addItems(["All", "GOOD", "NG", "ERROR"])
        self._serial = QLineEdit()
        self._serial.setPlaceholderText("Serial contains…")
        search_btn = QPushButton("Search")
        search_btn.setProperty("class", "primary")
        search_btn.clicked.connect(self._on_search)

        filters.addWidget(QLabel("Range"))
        filters.addWidget(self._quick)
        filters.addWidget(self._date_from)
        filters.addWidget(self._date_to)
        filters.addWidget(self._machine)
        filters.addWidget(self._result)
        filters.addWidget(self._serial, stretch=1)
        filters.addWidget(search_btn)
        root.addLayout(filters)

        # ------------------------------------------------------------ table
        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._table, stretch=1)

        # ------------------------------------------------- pagination + export
        bottom = QHBoxLayout()
        self._prev = QPushButton("◀ Prev")
        self._prev.clicked.connect(lambda: self._go_page(self._page - 1))
        self._next = QPushButton("Next ▶")
        self._next.clicked.connect(lambda: self._go_page(self._page + 1))
        self._page_label = QLabel("Page 1")
        bottom.addWidget(self._prev)
        bottom.addWidget(self._page_label)
        bottom.addWidget(self._next)
        bottom.addStretch()
        for text, handler in (
            ("Export CSV", self._export_csv),
            ("Export Excel", self._export_excel),
            ("Export PDF", self._export_pdf),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            bottom.addWidget(button)
        root.addLayout(bottom)

        self._on_quick_changed(self._quick.currentText())
        self._on_search()

    # -------------------------------------------------------------- filters
    def _on_quick_changed(self, text: str) -> None:
        custom = text == "Custom"
        self._date_from.setEnabled(custom)
        self._date_to.setEnabled(custom)

    def _current_filter(self) -> InspectionFilter:
        quick = self._quick.currentText()
        today = date.today()
        date_from = date_to = None
        if quick == "Today":
            date_from = datetime.combine(today, time.min)
            date_to = date_from + timedelta(days=1)
        elif quick == "Yesterday":
            date_from = datetime.combine(today - timedelta(days=1), time.min)
            date_to = date_from + timedelta(days=1)
        elif quick == "Last 7 days":
            date_from = datetime.combine(today - timedelta(days=6), time.min)
            date_to = datetime.combine(today, time.min) + timedelta(days=1)
        elif quick == "Custom":
            date_from = datetime.combine(self._date_from.date().toPython(), time.min)
            date_to = datetime.combine(self._date_to.date().toPython(), time.min) + timedelta(days=1)

        machine_text = self._machine.text().strip()
        result = self._result.currentText()
        return InspectionFilter(
            date_from=date_from,
            date_to=date_to,
            machine_number=int(machine_text) if machine_text.isdigit() else None,
            result=result if result != "All" else None,
            serial_number=self._serial.text().strip() or None,
        )

    # --------------------------------------------------------------- search
    def _on_search(self) -> None:
        self._go_page(0)

    def _go_page(self, page: int) -> None:
        page = max(0, page)
        rows, total = self._db.inspections.search(
            self._current_filter(), offset=page * PAGE_SIZE, limit=PAGE_SIZE
        )
        self._page, self._total = page, total
        self._fill_table(rows)
        pages = max(1, -(-total // PAGE_SIZE))
        self._page_label.setText(f"Page {page + 1} / {pages}   ({total} records)")
        self._prev.setEnabled(page > 0)
        self._next.setEnabled(page + 1 < pages)

    def _fill_table(self, rows: list[Inspection]) -> None:
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(rows))
        for row_index, inspection in enumerate(rows):
            details = {d.camera_index: d for d in inspection.details}
            cam_cells = []
            for camera in (1, 2, 3, 4):
                detail = details.get(camera)
                if detail is None:
                    cam_cells.append("")
                elif not detail.hole_found:
                    cam_cells.append("—")
                else:
                    cam_cells.append(f"{detail.x_mm:.1f}, {detail.y_mm:.1f}")
            values = [
                str(inspection.id),
                inspection.created_at.strftime("%Y-%m-%d"),
                inspection.created_at.strftime("%H:%M:%S"),
                str(inspection.machine_number),
                inspection.serial_number,
                *cam_cells,
                inspection.overall_result,
                f"{inspection.plc_cycle_time_ms:.0f}",
                f"{inspection.detection_time_ms:.0f}",
                inspection.operator,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 9:
                    item.setForeground(QColor(_RESULT_COLORS.get(value, COLOR_DIM)))
                self._table.setItem(row_index, column, item)
        self._table.setSortingEnabled(True)

    # --------------------------------------------------------------- export
    def _export_rows(self) -> list[Inspection]:
        rows, _ = self._db.inspections.search(
            self._current_filter(), offset=0, limit=EXPORT_CAP
        )
        return rows

    def _ask_path(self, caption: str, name_filter: str, default: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(self, caption, default, name_filter)
        return path or None

    def _export_csv(self) -> None:
        self._run_export("CSV", "CSV (*.csv)", "inspections.csv", self._export.export_csv)

    def _export_excel(self) -> None:
        self._run_export("Excel", "Excel (*.xlsx)", "inspections.xlsx", self._export.export_excel)

    def _export_pdf(self) -> None:
        self._run_export("PDF", "PDF (*.pdf)", "inspections.pdf", self._export.export_pdf)

    def _run_export(self, kind: str, name_filter: str, default: str, exporter) -> None:
        path = self._ask_path(f"Export {kind}", name_filter, default)
        if path is None:
            return
        try:
            exporter(path, self._export_rows())
        except VisionSystemError as exc:
            QMessageBox.warning(self, f"Export {kind}", str(exc))
            return
        QMessageBox.information(self, f"Export {kind}", f"Exported to:\n{path}")

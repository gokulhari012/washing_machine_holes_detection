"""The dashboard's right-hand column: cycle summary + camera coordinates.

Two read-only cards, laid out as bordered grid tables that fill their tile
top to bottom (no dead space, no floating text) — see
:class:`~ui.dashboard.dashboard_page.DashboardPage`, which gives both panels
a layout stretch factor instead of trailing off into an empty spacer:

* :class:`CycleSummaryPanel` — serial number across the top, then three
  label/value pairs a row (model / cycle time, clock / previous cycle time,
  shift / product count) and the last trigger time across the bottom.
* :class:`CameraCoordinatesPanel` — a 2×2 grid of small tables, one per
  camera, in the same position as that camera's picture in the grid beside
  it. Each table is 3 rows: the camera's name as a header cell merged across
  both columns, an "X"/"Y" bold column-header row beneath it, then the
  coordinate values themselves.

Every label/value pair in :class:`CycleSummaryPanel` is wrapped in its own
bordered "gridCell" card (see ``resources/styles/dark_theme.qss``) and every
grid row/column carries equal stretch, so as the panel is given more height
by its parent layout, the cards themselves grow to fill it — the border, not
just the text inside it, reaches the bottom of the tile. The coordinates
table fills the same way via stretched row/column resize modes instead.

Both are pure display: :class:`~ui.dashboard.dashboard_page.DashboardPage`
pushes every value in from ``AppState`` signals. The only thing either panel
drives itself is the wall clock, which no signal carries.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

EMPTY = "—"
CLOCK_INTERVAL_MS = 1000

# X and Y each get their own colour in the coordinates tables so the two axes
# are distinguishable at a glance without re-reading the column header every
# time — green for X, yellow for Y, requested explicitly (they overlap the
# GOOD/NG/ERROR palette elsewhere, but that's the point here: an axis, not a
# verdict, is what's being read at this spot).
X_VALUE_COLOR = QColor("#3fb950")
Y_VALUE_COLOR = QColor("#e3c53d")

CAMERA_HEADER_BG = QColor("#262e39")
AXIS_HEADER_BG = QColor("#20262f")
VALUE_ROW_BG = QColor("#1a2028")


def _caption(text: str, centered: bool = False) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("cellTitle")
    if centered:
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


def _value(text: str = EMPTY, centered: bool = False, large: bool = False) -> QLabel:
    label = QLabel(text)
    label.setObjectName("cellValueLarge" if large else "cellValue")
    if centered:
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


def _cell(caption_text: str, value_label: QLabel, *, centered: bool = False) -> QFrame:
    """One bordered label/value card — the atom every grid in this module is
    built from. Stretch added above and below centres the pair vertically,
    so a card that has been stretched taller than its content still reads as
    balanced rather than top-heavy."""
    cell = QFrame()
    cell.setObjectName("gridCell")
    layout = QVBoxLayout(cell)
    layout.setContentsMargins(14, 8, 14, 8)
    layout.setSpacing(4)
    layout.addStretch(1)
    layout.addWidget(_caption(caption_text, centered=centered))
    layout.addWidget(value_label)
    layout.addStretch(1)
    return cell


class CycleSummaryPanel(QFrame):
    """Serial / model / times / shift / count — one cycle at a glance."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "tile")

        grid = QGridLayout(self)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self._serial = _value(centered=True, large=True)
        grid.addWidget(_cell("Serial No", self._serial, centered=True), 0, 0, 1, 2)
        grid.setRowStretch(0, 1)

        self._model = _value()
        self._cycle_time = _value()
        self._clock = _value()
        self._previous_cycle_time = _value()
        self._shift = _value()
        self._product_count = _value()
        pairs = (
            ("Model Name", self._model, "Current Cycle Time", self._cycle_time),
            ("Current Time", self._clock, "Previous Cycle Time", self._previous_cycle_time),
            ("Current Shift", self._shift, "Product Count", self._product_count),
        )
        row = 1
        for left_caption, left_value, right_caption, right_value in pairs:
            grid.addWidget(_cell(left_caption, left_value), row, 0)
            grid.addWidget(_cell(right_caption, right_value), row, 1)
            grid.setRowStretch(row, 1)
            row += 1

        self._last_trigger = _value(centered=True)
        grid.addWidget(
            _cell("Last Trigger Time", self._last_trigger, centered=True), row, 0, 1, 2
        )
        grid.setRowStretch(row, 1)

        # The clock is the one value no signal delivers, so the panel ticks
        # it itself — one label a second, nothing else runs on this timer.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(CLOCK_INTERVAL_MS)
        self._tick()

    def _tick(self) -> None:
        self._clock.setText(datetime.now().strftime("%H:%M:%S"))

    # -------------------------------------------------------------- setters
    def set_serial(self, value: str) -> None:
        self._serial.setText(value or EMPTY)

    def set_model(self, value: str) -> None:
        self._model.setText(value or EMPTY)

    def set_shift(self, value: str) -> None:
        self._shift.setText(value or EMPTY)

    def set_product_count(self, value: int) -> None:
        self._product_count.setText(str(value))

    def set_last_trigger(self, value: str) -> None:
        self._last_trigger.setText(value or EMPTY)

    def push_cycle_time(self, milliseconds: float) -> None:
        """Record this cycle's time; the one it replaces becomes 'previous'."""
        self._previous_cycle_time.setText(self._cycle_time.text())
        self._cycle_time.setText(f"{milliseconds:.0f} ms")


def _table_item(
    text: str,
    *,
    background: QColor,
    color: QColor | None = None,
    bold: bool = False,
    pixel_size: int | None = None,
) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled)  # read-only, never selectable/editable
    item.setBackground(background)
    if color is not None:
        item.setForeground(color)
    if bold or pixel_size is not None:
        font = item.font()
        font.setBold(bold)
        if pixel_size is not None:
            font.setPixelSize(pixel_size)
        item.setFont(font)
    return item


def _camera_table() -> QTableWidget:
    """A bare 3-row × 2-column table, styled and locked down for display
    only — the shared shell every per-camera mini-table in
    :class:`CameraCoordinatesPanel` is built from."""
    table = QTableWidget(3, 2)
    table.setObjectName("coordinatesTable")
    table.horizontalHeader().hide()
    table.verticalHeader().hide()
    table.setShowGrid(True)
    # A long camera name wrapping to two lines would make that one table's
    # header row taller than the other three in the 2x2 grid, throwing their
    # rows out of alignment — elide instead, so all four stay in lockstep.
    table.setWordWrap(False)
    table.setTextElideMode(Qt.TextElideMode.ElideRight)
    table.setFrameShape(QFrame.Shape.NoFrame)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    return table


class CameraCoordinatesPanel(QFrame):
    """Detected hole X/Y in mm for every camera, as a 2×2 grid of tables.

    One small 3-row table per camera, arranged in the same 2×2 position as
    that camera's picture in the grid beside it: the camera's name as a
    header cell merged across both columns, a bold "X" / "Y" column-header
    row, then the values themselves — X in green, Y in yellow (see
    :data:`X_VALUE_COLOR` / :data:`Y_VALUE_COLOR`) so the two axes read apart
    instantly. Every table's row/column resize mode is ``Stretch`` and the
    outer grid's rows/columns carry equal stretch too, so all four fill the
    tile's full height with no leftover space, the same as
    :class:`CycleSummaryPanel`'s cards.
    """

    def __init__(self, cameras: list[tuple[int, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "tile")

        grid = QGridLayout(self)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        self._values: dict[int, tuple[QTableWidgetItem, QTableWidgetItem]] = {}
        for position, (index, name) in enumerate(cameras):
            table = _camera_table()

            name_item = _table_item(name, background=CAMERA_HEADER_BG, bold=True, pixel_size=13)
            # Each table is only half the panel's width in the 2x2 grid, so a
            # longer name (e.g. "Camera 4 - Bottom Right") can still be wider
            # than the column — the tooltip keeps it readable even elided.
            name_item.setToolTip(name)
            table.setItem(0, 0, name_item)
            table.setSpan(0, 0, 1, 2)

            axis_kwargs = dict(background=AXIS_HEADER_BG, bold=True, pixel_size=13)
            table.setItem(1, 0, _table_item("X", **axis_kwargs))
            table.setItem(1, 1, _table_item("Y", **axis_kwargs))

            x_value = _table_item(
                EMPTY, background=VALUE_ROW_BG, color=X_VALUE_COLOR, bold=True, pixel_size=12
            )
            y_value = _table_item(
                EMPTY, background=VALUE_ROW_BG, color=Y_VALUE_COLOR, bold=True, pixel_size=12
            )
            table.setItem(2, 0, x_value)
            table.setItem(2, 1, y_value)

            self._values[index] = (x_value, y_value)
            grid.addWidget(table, position // 2, position % 2)

    def set_position(self, camera_index: int, x_mm: float, y_mm: float) -> None:
        items = self._values.get(camera_index)
        if items is None:
            return
        # One decimal place — each value cell is only a quarter of the panel
        # wide in the 2x2 layout, and this matches the precision already
        # shown under each camera's own picture (ui.dashboard.camera_panel).
        items[0].setText(f"{x_mm:+.1f} mm")
        items[1].setText(f"{y_mm:+.1f} mm")

    def clear_position(self, camera_index: int) -> None:
        items = self._values.get(camera_index)
        if items is not None:
            items[0].setText(EMPTY)
            items[1].setText(EMPTY)

    def clear(self) -> None:
        for index in self._values:
            self.clear_position(index)

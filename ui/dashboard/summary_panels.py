"""The dashboard's right-hand column: cycle summary + camera coordinates.

Two read-only cards, laid out as the tables the line asked for:

* :class:`CycleSummaryPanel` — serial number across the top, then three
  label/value pairs a row (model / cycle time, clock / previous cycle time,
  shift / product count) and the last trigger time across the bottom.
* :class:`CameraCoordinatesPanel` — a 2×2 block of cameras, each showing the
  detected hole's X and Y in millimetres, in the same order as the picture
  grid beside it.

Both are pure display: :class:`~ui.dashboard.dashboard_page.DashboardPage`
pushes every value in from ``AppState`` signals. The only thing either panel
drives itself is the wall clock, which no signal carries.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

EMPTY = "—"
CLOCK_INTERVAL_MS = 1000


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


class CycleSummaryPanel(QFrame):
    """Serial / model / times / shift / count — one cycle at a glance."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "tile")

        grid = QGridLayout(self)
        grid.setContentsMargins(14, 10, 14, 12)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self._serial = _value(centered=True, large=True)
        grid.addWidget(_caption("Serial No", centered=True), 0, 0, 1, 2)
        grid.addWidget(self._serial, 1, 0, 1, 2)

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
        row = 2
        for left_caption, left_value, right_caption, right_value in pairs:
            grid.addWidget(_caption(left_caption), row, 0)
            grid.addWidget(_caption(right_caption), row, 1)
            grid.addWidget(left_value, row + 1, 0)
            grid.addWidget(right_value, row + 1, 1)
            grid.setRowMinimumHeight(row, 30)  # breathing room above each caption
            row += 2

        self._last_trigger = _value(centered=True)
        grid.setRowMinimumHeight(row, 30)
        grid.addWidget(_caption("Last Trigger Time", centered=True), row, 0, 1, 2)
        grid.addWidget(self._last_trigger, row + 1, 0, 1, 2)

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


class CameraCoordinatesPanel(QFrame):
    """Detected hole X/Y in mm for every camera, in picture-grid order."""

    def __init__(self, cameras: list[tuple[int, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "tile")

        grid = QGridLayout(self)
        grid.setContentsMargins(14, 10, 14, 12)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self._values: dict[int, tuple[QLabel, QLabel]] = {}
        for position, (index, name) in enumerate(cameras):
            block = QVBoxLayout()
            block.setSpacing(3)
            block.setContentsMargins(0, 4, 0, 4)
            title = QLabel(name)
            title.setObjectName("cellHeading")
            block.addWidget(title)

            axes = QGridLayout()
            axes.setHorizontalSpacing(12)
            axes.setVerticalSpacing(0)
            axes.setColumnStretch(0, 1)
            axes.setColumnStretch(1, 1)
            x_value, y_value = _value(), _value()
            axes.addWidget(_caption("X"), 0, 0)
            axes.addWidget(_caption("Y"), 0, 1)
            axes.addWidget(x_value, 1, 0)
            axes.addWidget(y_value, 1, 1)
            block.addLayout(axes)

            self._values[index] = (x_value, y_value)
            grid.addLayout(block, position // 2, position % 2)

    def set_position(self, camera_index: int, x_mm: float, y_mm: float) -> None:
        labels = self._values.get(camera_index)
        if labels is None:
            return
        labels[0].setText(f"{x_mm:+.2f} mm")
        labels[1].setText(f"{y_mm:+.2f} mm")

    def clear_position(self, camera_index: int) -> None:
        labels = self._values.get(camera_index)
        if labels is not None:
            labels[0].setText(EMPTY)
            labels[1].setText(EMPTY)

    def clear(self) -> None:
        for index in self._values:
            self.clear_position(index)

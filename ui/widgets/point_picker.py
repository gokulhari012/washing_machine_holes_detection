"""Click-to-place point picker on top of :class:`ImageView`.

Used by the Calibration page's ruler workflow: in pick mode, left-click adds
a point (image pixel coordinates) and right-click removes the last one.
Points are drawn as numbered markers joined by a dashed polyline so the
consecutive segments — each assumed a fixed physical spacing, e.g. 1 mm ruler
ticks, see ``CameraCalibration.scale_from_points`` — are visible before
computing a scale from them. A new frame (:meth:`set_frame`) clears any
points from the previous one, since pixel coordinates on the old image are
meaningless on a new capture.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsView

from ui.widgets.image_view import ImageView

_POINT_COLOR = QColor("#3fb950")
_POINT_RADIUS = 5.0


class PointPicker(ImageView):
    """ImageView + click-to-place calibration points."""

    points_changed = Signal(list)  # list[tuple[float, float]], image pixel coords

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pick_mode = False
        self._points: list[QPointF] = []
        self._markers: list[QGraphicsEllipseItem] = []
        self._lines: list[QGraphicsLineItem] = []

    # ------------------------------------------------------------------ api
    def set_pick_mode(self, enabled: bool) -> None:
        """Toggle between point-picking and normal pan/zoom interaction."""
        self._pick_mode = enabled
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag if enabled else QGraphicsView.DragMode.ScrollHandDrag
        )
        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor
        )

    def points(self) -> list[tuple[float, float]]:
        return [(point.x(), point.y()) for point in self._points]

    def clear_points(self) -> None:
        for item in (*self._markers, *self._lines):
            self.scene().removeItem(item)
        self._markers.clear()
        self._lines.clear()
        self._points.clear()
        self.points_changed.emit(self.points())

    def undo_last_point(self) -> None:
        if not self._points:
            return
        self._points.pop()
        self.scene().removeItem(self._markers.pop())
        if self._lines:
            self.scene().removeItem(self._lines.pop())
        self.points_changed.emit(self.points())

    def set_frame(self, frame: np.ndarray) -> None:  # noqa: D102 — see class docstring
        super().set_frame(frame)
        self.clear_points()

    # --------------------------------------------------------------- events
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._pick_mode and not self.image_rect.isEmpty():
            if event.button() == Qt.MouseButton.LeftButton:
                self._add_point(self._clamp(self.mapToScene(event.position().toPoint())))
                event.accept()
                return
            if event.button() == Qt.MouseButton.RightButton:
                self.undo_last_point()
                event.accept()
                return
        super().mousePressEvent(event)

    # ------------------------------------------------------------- internal
    def _add_point(self, point: QPointF) -> None:
        marker = QGraphicsEllipseItem(
            point.x() - _POINT_RADIUS,
            point.y() - _POINT_RADIUS,
            _POINT_RADIUS * 2,
            _POINT_RADIUS * 2,
        )
        marker.setBrush(QBrush(_POINT_COLOR))
        marker.setPen(QPen(Qt.GlobalColor.black, 1))
        marker.setZValue(15)
        self.scene().addItem(marker)
        self._markers.append(marker)

        if self._points:
            previous = self._points[-1]
            pen = QPen(_POINT_COLOR, 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)  # constant width regardless of zoom
            line = QGraphicsLineItem(previous.x(), previous.y(), point.x(), point.y())
            line.setPen(pen)
            line.setZValue(14)
            self.scene().addItem(line)
            self._lines.append(line)

        self._points.append(point)
        self.points_changed.emit(self.points())

    def _clamp(self, point: QPointF) -> QPointF:
        bounds = self.image_rect
        return QPointF(
            min(max(point.x(), 0.0), bounds.width()),
            min(max(point.y(), 0.0), bounds.height()),
        )

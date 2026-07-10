"""Interactive ROI selection on top of :class:`ImageView`.

In ROI mode, click-drag draws a rectangle (dashed accent outline, translucent
fill); on release the rectangle is clamped to the image bounds and emitted as
``roi_changed(x, y, w, h)`` in **image pixel coordinates** — exactly what
camera.json's ``roi`` block and the calibration pages consume. Outside ROI
mode the widget behaves like a normal zoom/pan ImageView.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsView

from ui.widgets.image_view import ImageView

_ACCENT = QColor("#2f81f7")


class RoiEditor(ImageView):
    """ImageView + rubber-band ROI drawing."""

    roi_changed = Signal(int, int, int, int)  # x, y, w, h (image pixels)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._roi_mode = False
        self._drag_origin: QPointF | None = None
        self._roi_item = QGraphicsRectItem()
        pen = QPen(_ACCENT, 2, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)  # constant width regardless of zoom
        self._roi_item.setPen(pen)
        self._roi_item.setBrush(QBrush(QColor(47, 129, 247, 40)))
        self._roi_item.setZValue(10)
        self._roi_item.hide()
        self.scene().addItem(self._roi_item)

    # ------------------------------------------------------------------ api
    def set_roi_mode(self, enabled: bool) -> None:
        """Toggle between ROI drawing and normal pan/zoom interaction."""
        self._roi_mode = enabled
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if enabled
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor
        )

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Show an existing ROI (w/h <= 0 hides it — 'full frame')."""
        if w <= 0 or h <= 0:
            self.clear_roi()
            return
        self._roi_item.setRect(QRectF(x, y, w, h))
        self._roi_item.show()

    def clear_roi(self) -> None:
        self._roi_item.hide()

    def current_roi(self) -> tuple[int, int, int, int]:
        """Displayed ROI as (x, y, w, h); (0,0,0,0) when none."""
        if not self._roi_item.isVisible():
            return 0, 0, 0, 0
        rect = self._roi_item.rect()
        return int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())

    # --------------------------------------------------------------- events
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (
            self._roi_mode
            and event.button() == Qt.MouseButton.LeftButton
            and not self.image_rect.isEmpty()
        ):
            self._drag_origin = self._clamp(self.mapToScene(event.position().toPoint()))
            self._roi_item.setRect(QRectF(self._drag_origin, self._drag_origin))
            self._roi_item.show()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_origin is not None:
            current = self._clamp(self.mapToScene(event.position().toPoint()))
            self._roi_item.setRect(QRectF(self._drag_origin, current).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_origin is not None:
            self._drag_origin = None
            rect = self._roi_item.rect()
            if rect.width() >= 4 and rect.height() >= 4:
                self.roi_changed.emit(
                    int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
                )
            else:  # a click, not a drag — treat as "clear"
                self.clear_roi()
                self.roi_changed.emit(0, 0, 0, 0)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------- internal
    def _clamp(self, point: QPointF) -> QPointF:
        bounds = self.image_rect
        return QPointF(
            min(max(point.x(), 0.0), bounds.width()),
            min(max(point.y(), 0.0), bounds.height()),
        )

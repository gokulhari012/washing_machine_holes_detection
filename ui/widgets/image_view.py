"""Zoomable/pannable image display for numpy frames.

``numpy_to_qpixmap`` is the single BGR/grayscale → QPixmap bridge used across
the UI. The QImage is **copied** so the pixmap never dangles on a numpy
buffer that a worker thread reuses.

Interactions: wheel = zoom (anchored under cursor), drag = pan,
double-click = re-fit. While auto-fit is active the image refits on resize.

Every frame shown also gets a magenta centre crosshair overlay, marking the
same point that ``CalibrationManager.evaluate`` treats as the position
origin (0, 0) for the x_mm/y_mm reported to the PLC/dashboard/database — a
display-only scene item, like the ROI rectangle in :class:`RoiEditor`, so it
never gets baked into saved inspection images.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

_CENTER_MARK_COLOR = QColor("#ff2d95")
_CENTER_MARK_FRACTION = 0.04  # crosshair arm length, as a fraction of the shorter image side
_CENTER_MARK_MIN_HALF_LENGTH = 6.0


def numpy_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """Convert a BGR (H,W,3) or grayscale (H,W) uint8 array to a QPixmap."""
    if frame.ndim == 2:
        height, width = frame.shape
        data = np.ascontiguousarray(frame)
        image = QImage(
            data.data, width, height, data.strides[0], QImage.Format.Format_Grayscale8
        )
    else:
        height, width = frame.shape[:2]
        rgb = np.ascontiguousarray(frame[..., ::-1])  # BGR -> RGB
        image = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
        )
    return QPixmap.fromImage(image.copy())  # detach from the numpy buffer


class ImageView(QGraphicsView):
    """Graphics-view based frame display with zoom/pan/fit."""

    ZOOM_IN_FACTOR = 1.25

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)

        pen = QPen(_CENTER_MARK_COLOR, 2, Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)  # constant width regardless of zoom
        self._center_h_line = QGraphicsLineItem()
        self._center_v_line = QGraphicsLineItem()
        for line in (self._center_h_line, self._center_v_line):
            line.setPen(pen)
            line.setZValue(20)
            line.hide()
            self._scene.addItem(line)

        self.setScene(self._scene)

        self._auto_fit = True
        self._has_image = False

        self.setBackgroundBrush(QColor("#0d1117"))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

    # ------------------------------------------------------------------ api
    @property
    def image_rect(self) -> QRectF:
        """Bounds of the displayed image in scene coordinates."""
        return QRectF(self._pixmap_item.pixmap().rect())

    def set_frame(self, frame: np.ndarray) -> None:
        """Display a new frame; keeps the current zoom unless auto-fit is on."""
        pixmap = numpy_to_qpixmap(frame)
        size_changed = pixmap.size() != self._pixmap_item.pixmap().size()
        self._pixmap_item.setPixmap(pixmap)
        if size_changed:
            self._scene.setSceneRect(QRectF(pixmap.rect()))
            self._update_center_mark(pixmap.width(), pixmap.height())
        if self._auto_fit or not self._has_image:
            self._fit()
        self._has_image = True

    def clear_frame(self) -> None:
        self._pixmap_item.setPixmap(QPixmap())
        self._has_image = False
        self._center_h_line.hide()
        self._center_v_line.hide()

    def reset_view(self) -> None:
        self._auto_fit = True
        self._fit()

    # --------------------------------------------------------------- events
    def wheelEvent(self, event) -> None:  # noqa: N802
        if not self._has_image:
            return
        self._auto_fit = False
        factor = (
            self.ZOOM_IN_FACTOR
            if event.angleDelta().y() > 0
            else 1.0 / self.ZOOM_IN_FACTOR
        )
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.reset_view()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._auto_fit and self._has_image:
            self._fit()

    # ------------------------------------------------------------- internal
    def _fit(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _update_center_mark(self, width: int, height: int) -> None:
        """Re-centre the crosshair on the image now displayed — its arm
        length scales with the image so it reads consistently at any
        resolution, from a small ROI crop to a full-sensor frame."""
        if width <= 0 or height <= 0:
            self._center_h_line.hide()
            self._center_v_line.hide()
            return
        center_x, center_y = width / 2, height / 2
        half_length = max(_CENTER_MARK_MIN_HALF_LENGTH, min(width, height) * _CENTER_MARK_FRACTION)
        self._center_h_line.setLine(center_x - half_length, center_y, center_x + half_length, center_y)
        self._center_v_line.setLine(center_x, center_y - half_length, center_x, center_y + half_length)
        self._center_h_line.show()
        self._center_v_line.show()

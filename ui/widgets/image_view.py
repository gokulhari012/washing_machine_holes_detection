"""Zoomable/pannable image display for numpy frames.

``numpy_to_qpixmap`` is the single BGR/grayscale → QPixmap bridge used across
the UI. The QImage is **copied** so the pixmap never dangles on a numpy
buffer that a worker thread reuses.

Interactions: wheel = zoom (anchored under cursor), drag = pan,
double-click = re-fit. While auto-fit is active the image refits on resize.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


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
        if self._auto_fit or not self._has_image:
            self._fit()
        self._has_image = True

    def clear_frame(self) -> None:
        self._pixmap_item.setPixmap(QPixmap())
        self._has_image = False

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

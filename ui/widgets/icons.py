"""Vector icons painted at runtime.

The dashboard's per-camera buttons originally carried text glyphs (``▶``,
``⌂``). Several of the UI fonts Windows picks by default are missing them, so
a button rendered as a replacement box — or as nothing at all, which reads as
a blank dark square on a dark panel.

Painting the shapes ourselves removes the font dependency entirely: the icons
look identical on every station, stay crisp on HiDPI displays (the pixmap is
rendered at the screen's device-pixel ratio), and take whatever colour the
theme asks for. Each factory returns a :class:`QIcon` carrying both a normal
and a dimmed disabled pixmap, so a greyed-out button still reads correctly
instead of relying on Qt's automatic — and rather washed out — fading.

Add a new icon by writing one ``_paint_*`` function that draws inside a
``size × size`` box and a matching public factory.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPixmap

#: nominal logical size the icons are painted at; the button sets the display size
ICON_SIZE = 32

_DEFAULT_COLOR = "#d6dbe3"
_DEFAULT_DISABLED = "#57606a"


# --------------------------------------------------------------- painters
def _paint_play(painter: QPainter, size: float, color: QColor) -> None:
    """Solid right-pointing triangle — 'run this camera now'."""
    path = QPainterPath()
    path.moveTo(size * 0.32, size * 0.22)
    path.lineTo(size * 0.78, size * 0.50)
    path.lineTo(size * 0.32, size * 0.78)
    path.closeSubpath()
    painter.fillPath(path, color)


# ---------------------------------------------------------------- factory
def _pixmap(paint, color: str, size: int) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    paint(painter, float(size), QColor(color))
    painter.end()
    return pixmap


def _icon(paint, color: str, disabled_color: str, size: int) -> QIcon:
    icon = QIcon()
    icon.addPixmap(_pixmap(paint, color, size), QIcon.Mode.Normal)
    icon.addPixmap(_pixmap(paint, disabled_color, size), QIcon.Mode.Disabled)
    return icon


def play_icon(
    color: str = _DEFAULT_COLOR,
    disabled_color: str = _DEFAULT_DISABLED,
    size: int = ICON_SIZE,
) -> QIcon:
    """Triangular 'run' icon for the per-camera trigger button."""
    return _icon(_paint_play, color, disabled_color, size)

"""Vector icons painted at runtime.

The dashboard's per-camera buttons originally carried text glyphs (``▶``,
``⌂``). ``⌂`` (U+2302 HOUSE) in particular is absent from several of the UI
fonts Windows picks by default, so the button rendered as a replacement box —
or as nothing at all, which reads as a blank dark square on a dark panel.

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

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

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


def _paint_plus(painter: QPainter, size: float, color: QColor) -> None:
    """Plus sign — 'nudge this axis in the positive direction'."""
    pen = QPen(
        color, max(1.5, size * 0.11), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap
    )
    painter.setPen(pen)
    painter.drawLine(QPointF(size * 0.5, size * 0.22), QPointF(size * 0.5, size * 0.78))
    painter.drawLine(QPointF(size * 0.22, size * 0.5), QPointF(size * 0.78, size * 0.5))


def _paint_minus(painter: QPainter, size: float, color: QColor) -> None:
    """Minus sign — 'nudge this axis in the negative direction'."""
    pen = QPen(
        color, max(1.5, size * 0.11), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap
    )
    painter.setPen(pen)
    painter.drawLine(QPointF(size * 0.22, size * 0.5), QPointF(size * 0.78, size * 0.5))


def _paint_home(painter: QPainter, size: float, color: QColor) -> None:
    """Outlined house — 'send this camera back to its home position'."""
    painter.setPen(
        QPen(
            color,
            max(1.5, size * 0.095),
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
    )
    painter.drawPolyline(
        [  # roof
            QPointF(size * 0.16, size * 0.50),
            QPointF(size * 0.50, size * 0.21),
            QPointF(size * 0.84, size * 0.50),
        ]
    )
    painter.drawPolyline(
        [  # walls
            QPointF(size * 0.28, size * 0.47),
            QPointF(size * 0.28, size * 0.79),
            QPointF(size * 0.72, size * 0.79),
            QPointF(size * 0.72, size * 0.47),
        ]
    )


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


def home_icon(
    color: str = _DEFAULT_COLOR,
    disabled_color: str = _DEFAULT_DISABLED,
    size: int = ICON_SIZE,
) -> QIcon:
    """House 'go home' icon for the per-camera home button."""
    return _icon(_paint_home, color, disabled_color, size)


def plus_icon(
    color: str = _DEFAULT_COLOR,
    disabled_color: str = _DEFAULT_DISABLED,
    size: int = ICON_SIZE,
) -> QIcon:
    """Plus icon for a jog axis's positive-direction button."""
    return _icon(_paint_plus, color, disabled_color, size)


def minus_icon(
    color: str = _DEFAULT_COLOR,
    disabled_color: str = _DEFAULT_DISABLED,
    size: int = ICON_SIZE,
) -> QIcon:
    """Minus icon for a jog axis's negative-direction button."""
    return _icon(_paint_minus, color, disabled_color, size)

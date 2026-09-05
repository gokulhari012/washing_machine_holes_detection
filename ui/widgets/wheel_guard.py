"""Stop spin boxes, combo boxes and sliders from eating the mouse wheel.

Every page is wrapped in a :class:`QScrollArea` (:meth:`MainWindow.add_page`),
and Qt's default is that a spin box / combo box / slider under the pointer
consumes a wheel event whether or not it has keyboard focus — those widgets
even default to ``Qt.FocusPolicy.WheelFocus``, so the first notch focuses the
widget and every notch after that keeps editing it. Scrolling a tall page
(the PLC register map, Detection's parameter forms) therefore silently
retypes whatever value happens to pass under the cursor.

On this HMI that is not cosmetic: a nudged exposure, register address or
tolerance is a live configuration change nobody asked for and nobody sees.
So the guard swallows wheel events on those widgets unless the widget has
focus, and re-sends the event to the enclosing scroll area instead, which
keeps the page scrolling normally under the pointer. A deliberate edit
(click or tab into the field, then scroll) still works.
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QComboBox,
    QSlider,
    QWidget,
)

# QAbstractSpinBox covers QSpinBox/QDoubleSpinBox/QDateEdit/QDateTimeEdit.
_GUARDED = (QAbstractSpinBox, QComboBox, QSlider)


def _enclosing_scroll_area(widget: QWidget) -> QAbstractScrollArea | None:
    """Nearest scrolling ancestor — the thing the user meant to scroll."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea):
            return parent
        parent = parent.parentWidget()
    return None


class _WheelGuard(QObject):
    """Event filter; one instance is parented to each guarded page."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() != QEvent.Type.Wheel:
            return False
        if isinstance(watched, QWidget) and watched.hasFocus():
            return False  # focused: the user is deliberately editing this field
        area = _enclosing_scroll_area(watched)  # type: ignore[arg-type]
        if area is not None:
            # Hand the notch to the page instead. The viewport is not itself
            # guarded, so this cannot recurse.
            event.ignore()
            QCoreApplication.sendEvent(area.viewport(), event)
        return True  # either way, the value must not change


def install_wheel_guard(root: QWidget) -> None:
    """Guard every spin box / combo box / slider already inside ``root``.

    Call once the widget tree is fully built — it walks the existing children
    and does not watch for later additions.
    """
    guard = _WheelGuard(root)  # parented: dies with the page
    for widget_type in _GUARDED:
        for widget in root.findChildren(widget_type):
            widget.installEventFilter(guard)
            # WheelFocus would let a stray notch focus the widget, after which
            # hasFocus() above would wave every following notch through.
            if widget.focusPolicy() == Qt.FocusPolicy.WheelFocus:
                widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

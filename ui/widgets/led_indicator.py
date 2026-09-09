"""LED status indicators for connection/health display."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QRadialGradient
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from core.utilities.enums import ConnectionState
from ui import theme

# Palette tokens, not hex: the light scheme darkens the verdict colours so a
# green LED still clears 3:1 against a white panel, and an LED nobody can
# read is worse than no LED at all.
STATE_TOKENS: dict[str, str] = {
    ConnectionState.CONNECTED.value: "good",
    ConnectionState.DISCONNECTED.value: "text-disabled",
    ConnectionState.CONNECTING.value: "warn",
    ConnectionState.ERROR.value: "ng",
}
_UNKNOWN_TOKEN = "text-disabled"


class LedIndicator(QWidget):
    """Small round LED painted with a radial gradient."""

    def __init__(self, diameter: int = 14, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._diameter = diameter
        # The last state set, kept so a theme change can re-resolve its token.
        # A raw colour passed to set_color clears it: that caller chose a
        # specific colour and gets to keep it.
        self._token: str | None = _UNKNOWN_TOKEN
        self._color = theme.qcolor(_UNKNOWN_TOKEN)
        self.setFixedSize(diameter, diameter)
        theme.subscribe(self._on_theme_changed)

    def set_state(self, state: ConnectionState | str) -> None:
        """Colour from a ConnectionState (or its string value)."""
        key = state.value if isinstance(state, ConnectionState) else str(state)
        self._token = STATE_TOKENS.get(key, _UNKNOWN_TOKEN)
        self._color = theme.qcolor(self._token)
        self.update()

    def set_color(self, color: str) -> None:
        self._token = None
        self._color = QColor(color)
        self.update()

    def _on_theme_changed(self, _theme) -> None:
        if self._token is not None:
            self._color = theme.qcolor(self._token)
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)

        gradient = QRadialGradient(
            rect.center().x() - rect.width() * 0.15,
            rect.center().y() - rect.height() * 0.15,
            rect.width() * 0.75,
        )
        gradient.setColorAt(0.0, self._color.lighter(150))
        gradient.setColorAt(1.0, self._color.darker(120))

        painter.setPen(self._color.darker(160))
        painter.setBrush(gradient)
        painter.drawEllipse(rect)
        painter.end()


class LabeledLed(QWidget):
    """LED + text, e.g. ``● PLC  connected`` for status bars and panels."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.led = LedIndicator()
        self._label = QLabel(text)
        layout.addWidget(self.led)
        layout.addWidget(self._label)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

    def set_state(self, state: ConnectionState | str, text: str | None = None) -> None:
        self.led.set_state(state)
        if text is not None:
            self._label.setText(text)

    def set_text(self, text: str) -> None:
        self._label.setText(text)

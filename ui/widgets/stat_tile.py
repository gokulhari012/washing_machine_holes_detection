"""Dashboard stat tile: small caption + large value, optional accent colour."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class StatTile(QFrame):
    """Styled via ``QFrame[class="tile"]`` / ``#tileTitle`` / ``#tileValue``."""

    def __init__(
        self,
        title: str,
        value: str = "—",
        accent: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("class", "tile")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 12)
        layout.setSpacing(2)

        self._title = QLabel(title.upper())
        self._title.setObjectName("tileTitle")
        self._value = QLabel(value)
        self._value.setObjectName("tileValue")
        layout.addWidget(self._title)
        layout.addWidget(self._value)

        if accent:
            self.set_accent(accent)

    def set_value(self, value: str | int | float) -> None:
        self._value.setText(str(value))

    def set_accent(self, color: str) -> None:
        """Colour only the value text (e.g. good=green, NG=red)."""
        self._value.setStyleSheet(f"color: {color};")

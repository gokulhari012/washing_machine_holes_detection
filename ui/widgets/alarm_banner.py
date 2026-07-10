"""Dismissable alarm banner shown above the page area.

Displays the most recent alarm; while visible, further alarms increment a
"+N more" counter instead of flickering the text. Styled per severity via the
``QFrame#alarmBanner[severity=...]`` QSS rules.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QWidget


class AlarmBanner(QFrame):
    """Hidden until :meth:`show_alarm` is called."""

    cleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("alarmBanner")
        self._count = 0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 8, 6)
        layout.setSpacing(10)

        self._icon = QLabel("⚠")  # ⚠
        self._message = QLabel("")
        self._more = QLabel("")
        self._more.setProperty("class", "dim")
        close_button = QToolButton()
        close_button.setText("✕")  # ✕
        close_button.setToolTip("Dismiss alarm")
        close_button.clicked.connect(self.clear)

        layout.addWidget(self._icon)
        layout.addWidget(self._message, stretch=1)
        layout.addWidget(self._more)
        layout.addWidget(close_button)

        self.hide()

    # ------------------------------------------------------------------ api
    def show_alarm(self, severity: str, message: str) -> None:
        """severity: ``"error"`` or ``"warning"`` (anything else = warning)."""
        if self.isVisible():
            self._count += 1
            self._more.setText(f"+{self._count - 1} more")
        else:
            self._count = 1
            self._more.setText("")

        self._message.setText(message)
        self.setProperty("severity", "error" if severity == "error" else "warning")
        self._repolish()
        self.show()

    def clear(self) -> None:
        self._count = 0
        self._more.setText("")
        self.hide()
        self.cleared.emit()

    # ------------------------------------------------------------- internal
    def _repolish(self) -> None:
        """Re-evaluate QSS after a dynamic property change."""
        style = self.style()
        style.unpolish(self)
        style.polish(self)
        for child in (self._icon, self._message, self._more):
            style.unpolish(child)
            style.polish(child)

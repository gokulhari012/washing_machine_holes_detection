"""Application theming: Fusion style + dark QSS + matching QPalette.

The QSS covers the widgets we style explicitly; the palette keeps the rest
(native dialogs, item-view branches, disabled text) consistent with it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from core.logging import get_logger
from core.utilities.enums import LogSource

logger = get_logger(LogSource.UI)

QSS_PATH = Path(__file__).resolve().parent.parent / "resources" / "styles" / "dark_theme.qss"

# central colour constants for code that draws (LEDs, overlays, charts)
COLOR_BG = "#14181d"
COLOR_SURFACE = "#1c2128"
COLOR_BORDER = "#2e3642"
COLOR_TEXT = "#d6dbe3"
COLOR_DIM = "#8b95a3"
COLOR_ACCENT = "#2f81f7"
COLOR_GOOD = "#3fb950"
COLOR_NG = "#f85149"
COLOR_WARN = "#d29922"


def apply_dark_theme(app: QApplication) -> None:
    """Install the dark industrial theme on the whole application."""
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLOR_BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLOR_SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1a1f26"))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLOR_TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor("#232933"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#232933"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(COLOR_TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLOR_ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Link, QColor(COLOR_ACCENT))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#57606a"))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#57606a")
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#57606a")
    )
    app.setPalette(palette)

    try:
        app.setStyleSheet(QSS_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.error("Could not load stylesheet %s: %s — using palette only", QSS_PATH, exc)

"""Application theming: Fusion style + a QSS template painted from a palette.

Two colour schemes, one stylesheet. ``resources/styles/theme.qss`` is a
*template* whose every colour is an ``@token``; this module holds one palette
per :class:`~core.utilities.enums.AppTheme` and substitutes the selected one
before handing the result to ``QApplication.setStyleSheet``. A second,
hand-maintained ``light_theme.qss`` was the obvious alternative and was
rejected: 300 lines duplicated across two files diverge the first time
somebody styles a new widget in only one of them.

The QSS covers the widgets we style through the stylesheet; the QPalette keeps
the rest (native dialogs, item-view branches, disabled text) consistent with
it, and :func:`color` serves the handful of places that *draw* rather than
style — table cell backgrounds, log-level foregrounds, the image viewer's
backdrop.

Switching theme is live: :func:`apply_theme` re-paints the whole application
and then fires the :func:`subscribe` observers, which is how those code-drawn
widgets get told to re-read their colours. Anything styled purely through the
QSS needs no subscription at all.

Adding a colour: add the token to **both** ``_DARK`` and ``_LIGHT``. A token
used in the QSS but missing from a palette raises at load time
(:class:`ConfigurationError`) rather than reaching the screen as a mangled
rule, and the mismatch is caught by ``tests/test_theme.py``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from core.logging import get_logger
from core.utilities.enums import AppTheme, LogSource
from core.utilities.exceptions import ConfigurationError

logger = get_logger(LogSource.UI)

# ``sys._MEIPASS`` is the PyInstaller extraction root; in a normal checkout the
# stylesheet simply lives two levels up from this module.
_RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
QSS_PATH = _RESOURCE_ROOT / "resources" / "styles" / "theme.qss"

#: placeholder syntax in the template; ``@`` is not legal QSS, so a token that
#: survives substitution is visible as a broken rule rather than a silent one.
_TOKEN_RE = re.compile(r"@([a-z0-9-]+)")


# ---------------------------------------------------------------- palettes
# Semantic names, not colour names: "surface" and "text-dim" mean the same
# thing in both schemes, which is what lets one QSS serve both. The dark
# values are the station's original industrial palette, unchanged.
_DARK: dict[str, str] = {
    # structure
    "bg": "#14181d",
    "bg-deep": "#10141a",
    "surface": "#161a20",
    "alt-bg": "#1a1f26",
    "border": "#2e3642",
    "border-strong": "#3a4552",
    "grid-line": "#242b35",
    # text
    "text": "#d6dbe3",
    "text-strong": "#e8ecf2",
    "text-dim": "#8b95a3",
    "text-muted": "#9aa5b3",
    "text-disabled": "#57606a",
    # buttons
    "button-bg": "#232933",
    "button-hover-bg": "#2a3240",
    "button-hover-border": "#465364",
    "button-pressed-bg": "#1d232c",
    "button-disabled-border": "#2a313b",
    # inputs
    "input-bg": "#1c2128",
    "input-disabled-bg": "#171c22",
    "header-bg": "#1c2128",
    # nav rail
    "nav-hover-bg": "#1a212b",
    "nav-hover-fg": "#c4ccd6",
    "nav-selected-bg": "#1d2634",
    # accent
    "accent": "#2f81f7",
    "accent-hover": "#4694ff",
    "accent-pressed": "#2469cc",
    "accent-disabled-bg": "#24405f",
    "accent-disabled-fg": "#7d8794",
    "on-accent": "#ffffff",
    "selection-bg": "#25436b",
    # verdict / severity
    "good": "#3fb950",
    "ng": "#f85149",
    "warn": "#d29922",
    "danger-bg": "#3d1418",
    "danger-fg": "#ffb3ae",
    "warn-bg": "#3a2d10",
    "warn-fg": "#e8c88a",
    # widgets
    "scroll-hover": "#46536a",
    "tab-hover-bg": "#1f252d",
    "slider-handle": "#d6dbe3",
    "slider-handle-hover": "#ffffff",
    "panel-icon-bg": "#2b333f",
    "panel-icon-border": "#46525f",
    "panel-icon-hover-bg": "#38424f",
    "panel-icon-hover-border": "#5b6a7c",
    "panel-icon-pressed-bg": "#232a34",
    # dashboard tiles / summary cards
    "tile-bg": "#1a2028",
    "tile-border": "#3a4553",
    "cell-bg": "#212832",
    "cell-border": "#38424f",
    "cell-title-fg": "#9aa4b2",
    "cell-value-fg": "#f2f5f9",
    "cell-value-strong-fg": "#ffffff",
    # code-drawn only (no QSS rule): coordinates table + image viewer
    "axis-x": "#3fb950",
    "axis-y": "#e3c53d",
    "table-camera-header-bg": "#262e39",
    "table-axis-header-bg": "#20262f",
    "table-value-bg": "#1a2028",
    "viewer-bg": "#0d1117",
    "icon-fg": "#d6dbe3",
    "icon-disabled-fg": "#57606a",
}

# The light scheme is the same structure inverted, not a different design: the
# roles keep their relationships (surface sits above bg, border-strong is more
# contrast than border) so every rule in the template still reads correctly.
# Verdict colours are darkened rather than reused — #3fb950 on white is under
# 3:1 and a GOOD/NG cell has to be legible from the line, not just present.
_LIGHT: dict[str, str] = {
    # structure
    "bg": "#f4f6f9",
    "bg-deep": "#e9edf2",
    "surface": "#ffffff",
    "alt-bg": "#f0f2f5",
    "border": "#d0d7e0",
    "border-strong": "#b6c0cc",
    "grid-line": "#e2e7ee",
    # text
    "text": "#1f2733",
    "text-strong": "#0f1621",
    "text-dim": "#5c6773",
    "text-muted": "#57626f",
    "text-disabled": "#9aa4b0",
    # buttons
    "button-bg": "#e8ebef",
    "button-hover-bg": "#dfe4ea",
    "button-hover-border": "#9aa6b4",
    "button-pressed-bg": "#cdd4dc",
    "button-disabled-border": "#e0e4ea",
    # inputs
    "input-bg": "#ffffff",
    "input-disabled-bg": "#eef0f3",
    "header-bg": "#eef1f5",
    # nav rail
    "nav-hover-bg": "#dde4ec",
    "nav-hover-fg": "#1f2733",
    "nav-selected-bg": "#d6e4fb",
    # accent
    "accent": "#1f6feb",
    "accent-hover": "#3b82f6",
    "accent-pressed": "#1a5ec4",
    "accent-disabled-bg": "#a9c7f5",
    "accent-disabled-fg": "#eef4fd",
    "on-accent": "#ffffff",
    "selection-bg": "#cfe0fb",
    # verdict / severity
    "good": "#1a7f37",
    "ng": "#d1242f",
    "warn": "#9a6700",
    "danger-bg": "#ffebe9",
    "danger-fg": "#a40e26",
    "warn-bg": "#fff8c5",
    "warn-fg": "#7d4e00",
    # widgets
    "scroll-hover": "#a8b2be",
    "tab-hover-bg": "#e6eaf0",
    "slider-handle": "#4a5563",
    "slider-handle-hover": "#1f2733",
    "panel-icon-bg": "#e4e9ef",
    "panel-icon-border": "#aeb8c4",
    "panel-icon-hover-bg": "#d3dae2",
    "panel-icon-hover-border": "#93a0af",
    "panel-icon-pressed-bg": "#c2cad3",
    # dashboard tiles / summary cards
    "tile-bg": "#ffffff",
    "tile-border": "#ccd4de",
    "cell-bg": "#f7f9fc",
    "cell-border": "#c7cfd9",
    "cell-title-fg": "#4a5563",
    "cell-value-fg": "#0f1621",
    "cell-value-strong-fg": "#0b1220",
    # code-drawn only (no QSS rule): coordinates table + image viewer
    "axis-x": "#1a7f37",
    "axis-y": "#8a6d00",
    "table-camera-header-bg": "#dfe6ef",
    "table-axis-header-bg": "#eaeff5",
    "table-value-bg": "#ffffff",
    # The viewer keeps a near-black backdrop in both schemes on purpose: it is
    # a letterbox around a photograph, and a white one washes out a dark part.
    "viewer-bg": "#22272e",
    "icon-fg": "#1f2733",
    "icon-disabled-fg": "#9aa4b0",
}

_PALETTES: dict[AppTheme, dict[str, str]] = {
    AppTheme.DARK: _DARK,
    AppTheme.LIGHT: _LIGHT,
}

# The theme in force. Module state rather than a constructor argument because
# the things that read it are leaf widgets that paint pixels (an LED, a table
# item, an icon pixmap); threading a palette through every one of them would
# buy nothing — the UI is single-threaded and there is exactly one QApplication.
_active: AppTheme = AppTheme.DARK
_observers: list[Callable[[AppTheme], None]] = []


# ------------------------------------------------------------------- api
def current_theme() -> AppTheme:
    """Which scheme is painted right now."""
    return _active


def color(token: str) -> str:
    """Hex value of *token* in the active theme.

    For code that draws instead of styling. Raises :class:`KeyError` on an
    unknown token — a typo here is a programming error, not a config fault.
    """
    return _PALETTES[_active][token]


def qcolor(token: str) -> QColor:
    """:func:`color` as a ``QColor``."""
    return QColor(color(token))


def subscribe(callback: Callable[[AppTheme], None]) -> None:
    """Register *callback* to run after every theme change.

    Only widgets that paint their own colours need this — anything styled by
    the QSS is re-painted by ``setStyleSheet`` alone. Callbacks run on the GUI
    thread, in registration order, and a raising callback never breaks the
    switch for the ones after it.

    Every subscriber today is built once in a page constructor and lives as
    long as the window, so nothing unsubscribes. A widget that is genuinely
    created and destroyed at runtime must call :func:`unsubscribe` when it
    goes — otherwise its callback keeps a deleted C++ object alive in this
    list and raises (harmlessly, but noisily) on every subsequent switch.
    """
    _observers.append(callback)


def unsubscribe(callback: Callable[[AppTheme], None]) -> None:
    """Remove a previously registered callback (no-op if absent)."""
    try:
        _observers.remove(callback)
    except ValueError:
        pass


def build_stylesheet(theme: AppTheme) -> str:
    """Render the QSS template with *theme*'s palette.

    Raises :class:`ConfigurationError` if the template names a token the
    palette does not define — that is a stylesheet Qt would silently drop
    rules from, so it is worth failing loudly at the one place that can.
    """
    palette = _PALETTES[theme]
    try:
        template = QSS_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Could not read stylesheet {QSS_PATH}: {exc}") from exc

    missing: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        value = palette.get(token)
        if value is None:
            missing.add(token)
            return match.group(0)
        return value

    rendered = _TOKEN_RE.sub(replace, template)
    if missing:
        raise ConfigurationError(
            f"Stylesheet uses tokens missing from the {theme.value} palette: "
            + ", ".join(sorted(missing))
        )
    return rendered


def apply_theme(app: QApplication, theme: AppTheme = AppTheme.DARK) -> None:
    """Paint the whole application in *theme* and notify the observers.

    Safe to call repeatedly: it is how a theme change is applied at runtime,
    not only at startup. A stylesheet that cannot be read or rendered leaves
    the palette applied and is logged — the station stays usable in the
    Fusion default rather than refusing to start over a cosmetic fault.
    """
    global _active
    _active = theme
    app.setStyle("Fusion")
    app.setPalette(_build_qpalette(theme))

    try:
        app.setStyleSheet(build_stylesheet(theme))
    except ConfigurationError as exc:
        logger.error("%s - using palette only", exc)

    for callback in list(_observers):
        try:
            callback(theme)
        except Exception:  # a bad observer must never break the switch
            logger.exception("Theme observer raised")


def apply_dark_theme(app: QApplication) -> None:
    """Install the dark industrial theme. Kept for the standalone lab scripts,
    which have no config file to read a preference from."""
    apply_theme(app, AppTheme.DARK)


# --------------------------------------------------------------- internals
def _build_qpalette(theme: AppTheme) -> QPalette:
    """QPalette matching the QSS, for the widgets the stylesheet cannot reach."""
    tokens = _PALETTES[theme]

    def c(token: str) -> QColor:
        return QColor(tokens[token])

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, c("bg"))
    palette.setColor(QPalette.ColorRole.WindowText, c("text"))
    palette.setColor(QPalette.ColorRole.Base, c("input-bg"))
    palette.setColor(QPalette.ColorRole.AlternateBase, c("alt-bg"))
    palette.setColor(QPalette.ColorRole.Text, c("text"))
    palette.setColor(QPalette.ColorRole.Button, c("button-bg"))
    palette.setColor(QPalette.ColorRole.ButtonText, c("text"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, c("button-bg"))
    palette.setColor(QPalette.ColorRole.ToolTipText, c("text"))
    palette.setColor(QPalette.ColorRole.Highlight, c("accent"))
    palette.setColor(QPalette.ColorRole.HighlightedText, c("on-accent"))
    palette.setColor(QPalette.ColorRole.Link, c("accent"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, c("text-disabled"))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, c("text-disabled")
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, c("text-disabled")
    )
    return palette


# -------------------------------------------------- backwards-compatible names
# Historical module constants, still imported by code that only ever needs the
# *verdict* colours. They are the dark values and do not follow a theme change
# - new code should call color()/qcolor() instead, which do.
COLOR_BG = _DARK["bg"]
COLOR_SURFACE = _DARK["input-bg"]
COLOR_BORDER = _DARK["border"]
COLOR_TEXT = _DARK["text"]
COLOR_DIM = _DARK["text-dim"]
COLOR_ACCENT = _DARK["accent"]
COLOR_GOOD = _DARK["good"]
COLOR_NG = _DARK["ng"]
COLOR_WARN = _DARK["warn"]

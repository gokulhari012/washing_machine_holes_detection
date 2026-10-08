"""Theme system: the two palettes, the QSS template, and the developer gate.

The dark scheme is what the station was commissioned against, so the first
test pins a sample of it against the hexes that used to be written straight
into ``dark_theme.qss`` — tokenising the stylesheet must not have moved a
single colour.

The rest guard the two ways this feature breaks quietly: a rule added to the
template naming a token only one palette defines (Qt drops the rule and the
widget silently loses its colour), and the Appearance group escaping its
developer-only gate.
"""

from __future__ import annotations

import re

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from core.utilities.enums import AppTheme, UserRole
from ui import theme


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------- palettes
def test_the_dark_palette_is_unchanged_by_tokenising_the_stylesheet():
    """The colours the station was commissioned with, spot-checked."""
    rendered = theme.build_stylesheet(AppTheme.DARK)
    assert "background-color: #14181d;" in rendered  # page background
    assert 'QLabel[class="dim"]       { color: #8b95a3; }' in rendered
    assert 'QLabel[result="GOOD"]  { color: #3fb950; font-weight: 700; }' in rendered
    assert "QListWidget#navRail::item:hover    { background-color: #1a212b;" in rendered


def test_both_palettes_define_every_token_the_template_uses():
    """A token missing from one palette is a rule Qt drops in silence."""
    template = theme.QSS_PATH.read_text(encoding="utf-8")
    used = set(re.findall(r"@([a-z0-9-]+)", template))
    assert used, "the template should still be tokenised"
    for scheme in AppTheme:
        rendered = theme.build_stylesheet(scheme)
        assert not re.search(r"@[a-z0-9-]+", rendered)


def test_the_two_palettes_hold_exactly_the_same_tokens():
    """Adding a colour to one scheme only is the way this drifts apart."""
    assert set(theme._DARK) == set(theme._LIGHT)


def test_the_light_scheme_is_actually_light_and_the_dark_one_dark():
    """Guards a copy-paste that leaves one scheme pointing at the other's
    values — every rule would still render, so nothing else would catch it."""
    from PySide6.QtGui import QColor

    dark_bg = QColor(theme._DARK["bg"]).lightnessF()
    light_bg = QColor(theme._LIGHT["bg"]).lightnessF()
    assert dark_bg < 0.25
    assert light_bg > 0.85
    # ...and the text has to go the other way, or the scheme is unreadable.
    assert QColor(theme._DARK["text"]).lightnessF() > 0.7
    assert QColor(theme._LIGHT["text"]).lightnessF() < 0.3


def test_an_unknown_token_is_refused_rather_than_left_on_screen(monkeypatch, tmp_path):
    from core.utilities.exceptions import ConfigurationError

    broken = tmp_path / "theme.qss"
    broken.write_text("QWidget { color: @not-a-real-token; }", encoding="utf-8")
    monkeypatch.setattr(theme, "QSS_PATH", broken)
    with pytest.raises(ConfigurationError, match="not-a-real-token"):
        theme.build_stylesheet(AppTheme.DARK)


# ------------------------------------------------------------ applying it
def test_applying_a_theme_updates_current_theme_and_the_color_accessor(qt_app):
    try:
        theme.apply_theme(qt_app, AppTheme.LIGHT)
        assert theme.current_theme() is AppTheme.LIGHT
        assert theme.color("bg") == theme._LIGHT["bg"]
        theme.apply_theme(qt_app, AppTheme.DARK)
        assert theme.color("bg") == theme._DARK["bg"]
    finally:
        theme.apply_theme(qt_app, AppTheme.DARK)


def test_observers_are_notified_and_a_raising_one_does_not_break_the_switch(qt_app):
    seen: list[AppTheme] = []

    def bad(_scheme):
        raise RuntimeError("boom")

    theme.subscribe(bad)
    theme.subscribe(seen.append)
    try:
        theme.apply_theme(qt_app, AppTheme.LIGHT)
        assert seen == [AppTheme.LIGHT]
    finally:
        theme.unsubscribe(bad)
        theme.unsubscribe(seen.append)
        theme.apply_theme(qt_app, AppTheme.DARK)


def test_an_unrecognised_config_value_degrades_to_dark():
    assert AppTheme.from_value("light") is AppTheme.LIGHT
    assert AppTheme.from_value("LIGHT") is AppTheme.LIGHT
    assert AppTheme.from_value("solarized") is AppTheme.DARK
    assert AppTheme.from_value(None) is AppTheme.DARK


# ------------------------------------------------------- the settings gate
class _FakeAuth:
    """Just the slice of AuthService the Settings page's access gate uses."""

    def __init__(self, role: UserRole | None = None) -> None:
        self.role = role
        self.callbacks: list = []

    def subscribe(self, callback) -> None:
        self.callbacks.append(callback)

    def has_role(self, required: UserRole) -> bool:
        return self.role is not None and self.role.covers(required)

    @property
    def is_admin(self) -> bool:
        return self.has_role(UserRole.ADMIN)

    @property
    def current_user(self):
        return None

    def login(self, role: UserRole | None) -> None:
        self.role = role
        for callback in self.callbacks:
            callback()


_CONFIG_NAMES = ("app_config", "plc", "camera", "detection", "machine_models")


def _config_manager(tmp_path):
    """A ConfigManager over a throwaway copy of the shipped defaults."""
    import shutil
    from pathlib import Path

    from core.utilities import ConfigManager

    source = Path(__file__).resolve().parent.parent / "config" / "defaults"
    config_dir = tmp_path / "config"
    (config_dir / "defaults").mkdir(parents=True)
    for name in _CONFIG_NAMES:
        shutil.copy(source / f"{name}.json", config_dir / f"{name}.json")
        shutil.copy(source / f"{name}.json", config_dir / "defaults" / f"{name}.json")
    return ConfigManager(config_dir)


@pytest.fixture(autouse=True)
def _no_modals(monkeypatch):
    """Save pops a confirmation; nothing here is driving an event loop."""
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))


def _settings_page(config, auth):
    """A real SettingsPage, built the way a fresh login would build it."""
    from services.shift_service import ShiftService
    from ui.settings.settings_page import SettingsPage

    return SettingsPage(
        config, auth, backup_service=None, shift_service=ShiftService(config)
    )


def test_the_theme_picker_is_developer_only(qt_app, tmp_path):
    auth = _FakeAuth(UserRole.ADMIN)
    page = _settings_page(_config_manager(tmp_path), auth)
    assert not page._appearance_box.isVisibleTo(page), "admins must not see it"

    auth.login(UserRole.DEVELOPER)
    assert page._appearance_box.isVisibleTo(page)

    auth.login(UserRole.ADMIN)
    assert not page._appearance_box.isVisibleTo(page)


def test_saving_records_the_chosen_theme(qt_app, tmp_path):
    auth = _FakeAuth(UserRole.DEVELOPER)
    config = _config_manager(tmp_path)
    page = _settings_page(config, auth)
    page._theme.setCurrentIndex(page._theme.findData(AppTheme.LIGHT.value))
    page._on_save()
    assert config.get_value("app_config", "application.theme") == "light"


def test_an_admin_save_preserves_a_developers_choice(qt_app, tmp_path):
    """The picker is hidden for an admin, not absent — saving Settings as an
    admin must not quietly reset the scheme to the first item in the list."""
    auth = _FakeAuth(UserRole.DEVELOPER)
    config = _config_manager(tmp_path)
    page = _settings_page(config, auth)
    page._theme.setCurrentIndex(page._theme.findData(AppTheme.LIGHT.value))
    page._on_save()

    auth.login(UserRole.ADMIN)
    _settings_page(config, auth)._on_save()
    assert config.get_value("app_config", "application.theme") == "light"


# ------------------------------------------------- LED page's drawn log
# The LED page appends inline-styled HTML to a QTextEdit rather than letting
# the QSS colour it — a text document's spans are not widgets — so it is one
# of the places that must look its colours up per theme and re-render on a
# switch. These pin the wiring without building the whole page.
def test_every_led_log_kind_maps_to_a_token_both_palettes_define():
    """A new log kind whose token exists in only one scheme is the drift."""
    from ui.led.led_page import _LOG_TOKENS

    tokens = set(_LOG_TOKENS.values()) | {"led-log-default"}
    for scheme in AppTheme:
        for token in tokens:
            assert token in theme._PALETTES[scheme], f"{token} missing from {scheme}"


def test_the_led_log_colours_differ_between_the_schemes():
    """The dark scheme's pastels are unreadable on the light QTextEdit, so
    they must not simply be reused."""
    from ui.led.led_page import _LOG_TOKENS

    for token in _LOG_TOKENS.values():
        assert theme._DARK[token] != theme._LIGHT[token], token


def test_settings_page_saves_the_full_view_checkbox(qt_app, tmp_path) -> None:
    config = _config_manager(tmp_path)
    page = _settings_page(config, _FakeAuth(UserRole.ADMIN))
    assert page._full_frame.isChecked()  # shipped default: full image

    page._full_frame.setChecked(False)
    page._on_save()
    _dispose(page)

    assert config.get_value("app_config", "ui.dashboard_full_frame") is False


def test_settings_page_saves_the_dashboard_screenshot_checkbox(qt_app, tmp_path) -> None:
    config = _config_manager(tmp_path)
    page = _settings_page(config, _FakeAuth(UserRole.ADMIN))
    assert not page._dashboard_shot.isChecked()  # shipped default: off

    page._dashboard_shot.setChecked(True)
    page._on_save()
    _dispose(page)

    assert config.get_value("app_config", "storage.save_dashboard_screenshot") is True


def _dispose(widget) -> None:
    """Delete a parentless widget now, while the QApplication is alive.
    Left to Python, it is collected in pytest's final gc pass, after Qt has
    started tearing down, which crashes the interpreter on exit."""
    import shiboken6

    shiboken6.delete(widget)


def _wait_for(predicate, timeout_s: float = 3.0) -> bool:
    import time

    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.mark.parametrize(
    ("enabled", "partial", "saved"),
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_dashboard_screenshot_after_a_full_cycle_only(
    qt_app, tmp_path, enabled, partial, saved
) -> None:
    """A full cycle with the setting on leaves a PNG beside its camera
    images; a single-camera cycle, or the setting off, leaves nothing."""
    from datetime import datetime
    from types import SimpleNamespace

    from PySide6.QtWidgets import QLabel

    from ui.dashboard import dashboard_snapshot
    from ui.dashboard.dashboard_snapshot import DashboardSnapshotter

    config = _config_manager(tmp_path)
    cfg = config.load("app_config")
    cfg["storage"]["image_directory"] = str(tmp_path / "images")
    cfg["storage"]["save_dashboard_screenshot"] = enabled
    config.save("app_config", cfg)

    widget = QLabel("dashboard")
    widget.resize(200, 100)
    cycle = SimpleNamespace(
        partial=partial,
        started_at=datetime(2026, 10, 1, 9, 30, 15),
        machine_number=42,
        overall_result=SimpleNamespace(value="GOOD"),
    )
    DashboardSnapshotter(widget, config).on_inspection_finished(cycle)

    expected = tmp_path / "images" / "2026-10-01" / "093015_42_dashboard_GOOD.png"
    try:
        if saved:
            assert _wait_for(expected.exists)
        else:
            _wait_for(lambda: False, timeout_s=(dashboard_snapshot.SETTLE_MS + 200) / 1000)
            assert not (tmp_path / "images").exists()
    finally:
        _dispose(widget)


@pytest.mark.parametrize("align", [True, False])
def test_dashboard_screenshot_auto_align_resets_the_views_first(
    qt_app, tmp_path, align
) -> None:
    """With auto-align on, every camera picture is reset to its default view
    before the grab; with it off, an operator's zoom/pan is left alone."""
    from datetime import datetime
    from types import SimpleNamespace

    from PySide6.QtWidgets import QLabel

    from ui.dashboard.dashboard_snapshot import DashboardSnapshotter

    config = _config_manager(tmp_path)
    cfg = config.load("app_config")
    cfg["storage"]["image_directory"] = str(tmp_path / "images")
    cfg["storage"]["save_dashboard_screenshot"] = True
    cfg["storage"]["auto_align_dashboard_screenshot"] = align
    config.save("app_config", cfg)

    calls: list[str] = []

    class _Dashboard(QLabel):
        def reset_image_views(self) -> None:
            calls.append("reset")

        def grab(self, *args):
            calls.append("grab")
            return super().grab(*args)

    widget = _Dashboard("dashboard")
    widget.resize(200, 100)
    cycle = SimpleNamespace(
        partial=False,
        started_at=datetime(2026, 10, 1, 9, 30, 15),
        machine_number=42,
        overall_result=SimpleNamespace(value="GOOD"),
    )
    DashboardSnapshotter(widget, config).on_inspection_finished(cycle)

    expected = tmp_path / "images" / "2026-10-01" / "093015_42_dashboard_GOOD.png"
    try:
        assert _wait_for(expected.exists)
        assert calls == (["reset", "grab"] if align else ["grab"])
    finally:
        _dispose(widget)


def test_dashboard_reset_image_views_refits_a_zoomed_panel(qt_app) -> None:
    import numpy as np

    from ui.dashboard.camera_panel import CameraPanel

    panel = CameraPanel(1, "Camera 1")
    panel.resize(400, 300)
    view = panel._view
    try:
        view.set_frame(np.zeros((120, 160, 3), dtype=np.uint8))
        fitted = view.transform().m11()
        view.scale(3.0, 3.0)
        view._auto_fit = False
        panel.reset_view()
        assert view._auto_fit is True
        assert view.transform().m11() == pytest.approx(fitted)
    finally:
        _dispose(panel)

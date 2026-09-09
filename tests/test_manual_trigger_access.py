"""The manual trigger controls are developer-only (offscreen Qt).

Two widgets carry them: the toolbar's "Simulate trigger for all camera at a
time" button and the Dashboard's trigger bar (delay between cameras + its own
Simulate Trigger). Firing the station by hand and re-timing its capture
sequence are commissioning acts, so both are *hidden* — not merely disabled —
for the logged-out operator and for admins, and both have to notice a session
change after they were built.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from core.database import DatabaseEngine, User
from core.utilities.enums import UserRole
from models.app_state import AppState
from services.auth_service import AuthService
from services.database_service import DatabaseService
from ui.dashboard.dashboard_page import DashboardPage
from ui.main_window import MainWindow

CAMERA_CONFIGS = [
    {"index": 1, "name": "Camera 1", "enabled": True},
    {"index": 2, "name": "Camera 2", "enabled": True},
]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def auth(tmp_path):
    engine = DatabaseEngine(tmp_path / "trigger.db")
    engine.create_schema()
    service = AuthService(DatabaseService(engine))
    service.ensure_default_accounts()
    yield service
    engine.dispose()


def login_as(service: AuthService, role: UserRole) -> None:
    """Bypass the dialog, then fan out as a real login does."""
    service._current = User(username=role.value, password_hash="", role=role.value)
    service._notify()


# ------------------------------------------------------------------- toolbar
@pytest.fixture()
def window(qapp, auth):
    win = MainWindow(
        AppState(),
        auth,
        factory_name="test",
        camera_indexes=[1],
        on_simulate_all_cameras=lambda: None,
    )
    win.add_page("Dashboard", "•", QWidget())
    yield win
    win.close()


# The window is never shown in these tests, so isVisible() is False for
# everything — isHidden() is the flag setVisible() actually toggles.
def test_toolbar_trigger_is_hidden_when_logged_out(window) -> None:
    assert window._simulate_button.isHidden() is True


def test_toolbar_trigger_stays_hidden_for_an_admin(window, auth) -> None:
    login_as(auth, UserRole.ADMIN)
    window._refresh_nav_visibility()
    assert window._simulate_button.isHidden() is True


def test_toolbar_trigger_appears_for_a_developer(window, auth) -> None:
    login_as(auth, UserRole.DEVELOPER)
    window._refresh_nav_visibility()
    assert window._simulate_button.isHidden() is False
    assert window._simulate_button.text() == "Simulate trigger for all camera at a time"


# ----------------------------------------------------------------- dashboard
@pytest.fixture()
def dashboard(qapp, auth):
    page = DashboardPage(
        AppState(),
        CAMERA_CONFIGS,
        on_simulate_trigger=lambda: None,
        on_camera_trigger=lambda index: None,
        auth_service=auth,
    )
    yield page
    page.deleteLater()


def test_dashboard_trigger_bar_is_hidden_when_logged_out(dashboard) -> None:
    assert dashboard._trigger_bar.isVisibleTo(dashboard) is False


def test_dashboard_trigger_bar_stays_hidden_for_an_admin(dashboard, auth) -> None:
    login_as(auth, UserRole.ADMIN)
    assert dashboard._trigger_bar.isVisibleTo(dashboard) is False


def test_dashboard_trigger_bar_follows_the_session(dashboard, auth) -> None:
    """No explicit refresh call: the page subscribed to the auth service, so a
    developer logging in after it was built still gets the controls."""
    login_as(auth, UserRole.DEVELOPER)
    assert dashboard._trigger_bar.isVisibleTo(dashboard) is True

    auth.logout()
    assert dashboard._trigger_bar.isVisibleTo(dashboard) is False


def test_per_camera_triggers_are_not_gated(dashboard) -> None:
    """The panel ▶ buttons are the operator's own control and stay available."""
    assert dashboard._panels
    for panel in dashboard._panels.values():
        assert panel.isVisibleTo(dashboard) is True


def test_no_auth_service_leaves_the_bar_visible(qapp) -> None:
    page = DashboardPage(AppState(), CAMERA_CONFIGS, on_simulate_trigger=lambda: None)
    assert page._trigger_bar.isVisibleTo(page) is True
    page.deleteLater()

"""Nav-rail role gating in ``MainWindow`` (offscreen Qt).

The one behaviour worth pinning down at the widget level: which nav entries a
session can see, and that the selection never strands on a row that just
became hidden. Runs on the offscreen platform plugin, so it needs no display.
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
from ui.main_window import MainWindow

# Same table as main.py's page registration, least privileged first.
PAGES = [
    ("Dashboard", None),
    ("Cameras", UserRole.ADMIN),
    ("PLC", UserRole.ADMIN),
    ("Detection", UserRole.ADMIN),
    ("Calibration", UserRole.DEVELOPER),
    ("Machine Models", UserRole.DEVELOPER),
    ("Database", None),
    ("Logs", None),
    ("Settings", UserRole.ADMIN),
]
OPERATOR_PAGES = {"Dashboard", "Database", "Logs"}
ADMIN_PAGES = OPERATOR_PAGES | {"Cameras", "PLC", "Detection", "Settings"}
DEVELOPER_PAGES = ADMIN_PAGES | {"Calibration", "Machine Models"}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(qapp, tmp_path):
    engine = DatabaseEngine(tmp_path / "nav.db")
    engine.create_schema()
    auth = AuthService(DatabaseService(engine))
    auth.ensure_default_accounts()
    win = MainWindow(AppState(), auth, factory_name="test", camera_indexes=[1])
    for title, min_role in PAGES:
        win.add_page(title, "•", QWidget(), min_role=min_role)
    yield win, auth
    win.close()
    engine.dispose()


def visible_titles(win: MainWindow) -> set[str]:
    nav = win._nav
    return {
        nav.item(row).text().split("  ", 1)[1]
        for row in range(nav.count())
        if not nav.item(row).isHidden()
    }


def login_as(auth: AuthService, role: UserRole) -> None:
    """Bypass the dialog — the password is not what is under test here."""
    auth._current = User(username=role.value, password_hash="", role=role.value)


def test_logged_out_sees_the_operator_view(window) -> None:
    win, _auth = window
    assert visible_titles(win) == OPERATOR_PAGES


def test_admin_sees_the_engineering_consoles_but_not_commissioning(window) -> None:
    win, auth = window
    login_as(auth, UserRole.ADMIN)
    win._refresh_nav_visibility()
    assert visible_titles(win) == ADMIN_PAGES


def test_developer_sees_every_page(window) -> None:
    win, auth = window
    login_as(auth, UserRole.DEVELOPER)
    win._refresh_nav_visibility()
    assert visible_titles(win) == DEVELOPER_PAGES
    assert len(DEVELOPER_PAGES) == len(PAGES)


def test_selection_falls_back_when_the_current_page_is_locked(window) -> None:
    win, auth = window
    login_as(auth, UserRole.DEVELOPER)
    win._refresh_nav_visibility()
    win.show_page(PAGES.index(("Calibration", UserRole.DEVELOPER)))

    login_as(auth, UserRole.ADMIN)  # developer session replaced by an admin one
    win._refresh_nav_visibility()
    assert win._nav.currentRow() == 0  # Dashboard — always visible


def test_selection_survives_a_page_that_stays_visible(window) -> None:
    win, auth = window
    login_as(auth, UserRole.ADMIN)
    win._refresh_nav_visibility()
    database_row = PAGES.index(("Database", None))
    win.show_page(database_row)

    auth.logout()
    win._refresh_nav_visibility()
    assert win._nav.currentRow() == database_row

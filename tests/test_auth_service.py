"""Role hierarchy and default-account bootstrap.

The three access tiers nest (operator < admin < developer), so most of what
matters here is that a *higher* role satisfies a *lower* gate — the mistake
these tests exist to catch is an ``== ADMIN`` check reappearing somewhere and
locking developers out of admin pages.
"""

from __future__ import annotations

import pytest

from core.database import DatabaseEngine
from core.utilities.enums import UserRole
from core.utilities.exceptions import AuthenticationError
from services.auth_service import (
    DEFAULT_ADMIN_PASSWORD,
    DEFAULT_ADMIN_USER,
    DEFAULT_DEVELOPER_PASSWORD,
    DEFAULT_DEVELOPER_USER,
    AuthService,
)
from services.database_service import DatabaseService


@pytest.fixture()
def auth(tmp_path) -> AuthService:
    engine = DatabaseEngine(tmp_path / "auth.db")
    engine.create_schema()
    service = AuthService(DatabaseService(engine))
    service.ensure_default_accounts()
    yield service
    engine.dispose()


def test_roles_nest() -> None:
    assert UserRole.DEVELOPER.covers(UserRole.ADMIN)
    assert UserRole.DEVELOPER.covers(UserRole.OPERATOR)
    assert UserRole.ADMIN.covers(UserRole.OPERATOR)
    assert not UserRole.ADMIN.covers(UserRole.DEVELOPER)
    assert not UserRole.OPERATOR.covers(UserRole.ADMIN)
    assert UserRole.ADMIN.covers(UserRole.ADMIN)


def test_logged_out_has_no_role(auth: AuthService) -> None:
    assert auth.current_role is None
    assert not auth.is_admin and not auth.is_developer
    assert not auth.has_role(UserRole.OPERATOR)  # nav gating: logged out sees no gated page
    with pytest.raises(AuthenticationError):
        auth.require_admin()


def test_admin_login_stops_below_developer(auth: AuthService) -> None:
    auth.login(DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASSWORD)
    assert auth.current_role is UserRole.ADMIN
    assert auth.is_admin and not auth.is_developer
    auth.require_admin()


def test_developer_satisfies_admin_gates(auth: AuthService) -> None:
    auth.login(DEFAULT_DEVELOPER_USER, DEFAULT_DEVELOPER_PASSWORD)
    assert auth.current_role is UserRole.DEVELOPER
    assert auth.is_developer and auth.is_admin  # manual PLC writes, Settings edits
    auth.require_admin()


def test_logout_clears_the_session(auth: AuthService) -> None:
    auth.login(DEFAULT_DEVELOPER_USER, DEFAULT_DEVELOPER_PASSWORD)
    auth.logout()
    assert auth.current_role is None and not auth.is_admin


def test_unknown_role_fails_closed(auth: AuthService, tmp_path) -> None:
    """A role string outside UserRole degrades to operator, never raises."""
    auth._users.create("stranger", auth._users.get_by_username("admin").password_hash, "wizard")
    auth.login("stranger", DEFAULT_ADMIN_PASSWORD)
    assert auth.current_role is UserRole.OPERATOR
    assert not auth.is_admin


def test_bootstrap_adds_a_missing_account_to_an_existing_station(tmp_path) -> None:
    """The developer account appears on a database that only had admin —
    the check is per account, not "is the users table empty"."""
    engine = DatabaseEngine(tmp_path / "legacy.db")
    engine.create_schema()
    database = DatabaseService(engine)
    service = AuthService(database)
    service.ensure_default_accounts()
    # simulate a pre-developer station: drop the developer row
    with engine.session_scope() as session:
        session.delete(database.users.get_by_username(DEFAULT_DEVELOPER_USER))
    assert database.users.get_by_username(DEFAULT_DEVELOPER_USER) is None

    service.ensure_default_accounts()
    developer = database.users.get_by_username(DEFAULT_DEVELOPER_USER)
    assert developer is not None and developer.role == UserRole.DEVELOPER.value
    engine.dispose()


def test_bootstrap_never_resets_a_changed_password(auth: AuthService) -> None:
    auth.login(DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASSWORD)
    auth.change_password(DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASSWORD, "hunter2")
    auth.logout()

    auth.ensure_default_accounts()
    with pytest.raises(AuthenticationError):
        auth.login(DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASSWORD)
    assert auth.login(DEFAULT_ADMIN_USER, "hunter2").role == UserRole.ADMIN.value

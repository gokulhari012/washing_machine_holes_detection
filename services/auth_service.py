"""Authentication for password-protected areas (Settings, manual PLC writes)
and for the role-gated pages in the nav rail.

Passwords are stored as salted PBKDF2-HMAC-SHA256 in the format

    pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>

Three access levels, nested rather than parallel — see :class:`UserRole`.
Logged out is the operator view (Dashboard, Database, Logs); ``admin`` adds
the engineering consoles; ``developer`` adds the commissioning consoles on
top of everything an admin sees. Every gate in the app therefore asks
:meth:`AuthService.has_role` (or ``is_admin``, which is "admin **or above**")
instead of comparing the role for equality.

Default accounts (``admin``/``admin`` and ``developer``/``developer``) are
created on demand by :meth:`AuthService.ensure_default_accounts` with a
prominent warning logged — the Settings page prompts for a password change on
first login. That check is per account, not "is the table empty", so a station
commissioned before the developer role existed gains the account on its next
start instead of staying stuck with only ``admin``.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from core.database import User
from core.logging import get_logger
from core.utilities.enums import LogSource, UserRole
from core.utilities.exceptions import AuthenticationError
from services.database_service import DatabaseService

logger = get_logger(LogSource.SYSTEM)

_ALGORITHM = "pbkdf2_sha256"
_ITERATIONS = 200_000
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASSWORD = "admin"
DEFAULT_DEVELOPER_USER = "developer"
DEFAULT_DEVELOPER_PASSWORD = "developer"

_DEFAULT_ACCOUNTS: tuple[tuple[str, str, UserRole], ...] = (
    (DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASSWORD, UserRole.ADMIN),
    (DEFAULT_DEVELOPER_USER, DEFAULT_DEVELOPER_PASSWORD, UserRole.DEVELOPER),
)


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"{_ALGORITHM}${_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_hex, hash_hex = stored.split("$")
        if algorithm != _ALGORITHM:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(digest.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


class AuthService:
    """Login session + account management over the user repository."""

    def __init__(self, database_service: DatabaseService) -> None:
        self._users = database_service.users
        self._current: User | None = None

    # -------------------------------------------------------------- session
    @property
    def current_user(self) -> User | None:
        return self._current

    @property
    def current_role(self) -> UserRole | None:
        """The logged-in account's role, or ``None`` when nobody is logged in.

        A stored role string that is not a known :class:`UserRole` degrades to
        ``OPERATOR`` — the least privilege — rather than raising: a typo or a
        role removed from a future build must fail closed, never unlock a page.
        """
        if self._current is None:
            return None
        try:
            return UserRole(self._current.role)
        except ValueError:
            logger.warning(
                "User %s has unknown role %r — treating as %s",
                self._current.username,
                self._current.role,
                UserRole.OPERATOR.value,
            )
            return UserRole.OPERATOR

    def has_role(self, required: UserRole) -> bool:
        """True when the session's role covers *required* (see UserRole)."""
        role = self.current_role
        return role is not None and role.covers(required)

    @property
    def is_admin(self) -> bool:
        """Admin **or above** — a developer satisfies every admin gate."""
        return self.has_role(UserRole.ADMIN)

    @property
    def is_developer(self) -> bool:
        return self.has_role(UserRole.DEVELOPER)

    def login(self, username: str, password: str) -> User:
        """Raises AuthenticationError on unknown user / wrong password."""
        user = self._users.get_by_username(username.strip())
        if user is None or not _verify_password(password, user.password_hash):
            logger.warning("Failed login attempt for %r", username)
            raise AuthenticationError("Invalid username or password")
        self._users.touch_last_login(user.username)
        self._current = user
        logger.info("User %s logged in (%s)", user.username, user.role)
        return user

    def logout(self) -> None:
        if self._current is not None:
            logger.info("User %s logged out", self._current.username)
        self._current = None

    def require_admin(self) -> None:
        """Raises AuthenticationError unless an admin is logged in."""
        if not self.is_admin:
            raise AuthenticationError("Administrator login required")

    # ------------------------------------------------------------- accounts
    def ensure_default_accounts(self) -> None:
        """Create any missing default account (admin/admin, developer/developer).

        Checked per account rather than "is the users table empty", so an
        existing station that only ever had ``admin`` picks the developer
        login up on its next start. An account an operator has since renamed
        or re-passworded is left alone — only a *missing* username is created.
        """
        for username, password, role in _DEFAULT_ACCOUNTS:
            if self._users.get_by_username(username) is not None:
                continue
            self._users.create(username, _hash_password(password), role.value)
            logger.warning(
                "Default %s account created (%s/%s) — "
                "CHANGE THE PASSWORD in Settings before production use",
                role.value,
                username,
                password,
            )

    def create_user(self, username: str, password: str, role: UserRole) -> None:
        """Admin-only. Raises AuthenticationError / DatabaseError."""
        self.require_admin()
        if self._users.get_by_username(username.strip()) is not None:
            raise AuthenticationError(f"User {username!r} already exists")
        self._users.create(username.strip(), _hash_password(password), role.value)
        logger.info("User %s created (%s)", username, role.value)

    def change_password(self, username: str, old_password: str, new_password: str) -> None:
        """Users change their own password; admins may skip the old-password
        check for other accounts. Raises AuthenticationError."""
        user = self._users.get_by_username(username.strip())
        if user is None:
            raise AuthenticationError(f"Unknown user {username!r}")
        acting_on_self = self._current is not None and self._current.username == username
        if acting_on_self or not self.is_admin:  # admins may reset other accounts
            if not _verify_password(old_password, user.password_hash):
                raise AuthenticationError("Current password is incorrect")
        if len(new_password) < 4:
            raise AuthenticationError("New password too short (minimum 4 characters)")
        self._users.update_password(username.strip(), _hash_password(new_password))
        logger.info("Password changed for %s", username)

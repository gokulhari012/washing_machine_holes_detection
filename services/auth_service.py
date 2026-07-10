"""Authentication for password-protected areas (Settings, manual PLC writes).

Passwords are stored as salted PBKDF2-HMAC-SHA256 in the format

    pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>

On first run (empty users table) a default ``admin``/``admin`` account is
created and a prominent warning is logged — the Settings page prompts for a
password change on first login.
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
    def is_admin(self) -> bool:
        return self._current is not None and self._current.role == UserRole.ADMIN.value

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
    def ensure_default_admin(self) -> None:
        """First-run bootstrap: create admin/admin when no users exist."""
        if self._users.count() == 0:
            self._users.create(
                DEFAULT_ADMIN_USER,
                _hash_password(DEFAULT_ADMIN_PASSWORD),
                UserRole.ADMIN.value,
            )
            logger.warning(
                "Default administrator account created (admin/admin) — "
                "CHANGE THE PASSWORD in Settings before production use"
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

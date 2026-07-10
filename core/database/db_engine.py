"""SQLite engine and session management.

Why WAL mode
------------
The DatabaseWorker thread writes inspections/logs while the UI thread reads
history and counters. SQLite's default rollback journal serialises readers
behind the writer; WAL (write-ahead logging) lets readers proceed
concurrently, which keeps the Database Viewer responsive during production.

Sessions
--------
``session_scope()`` yields a short-lived session that commits on success and
rolls back on error, translating SQLAlchemy failures into the domain's
:class:`~core.utilities.exceptions.DatabaseError`. ``expire_on_commit=False``
keeps returned ORM objects readable after the session closes.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from core.logging import get_logger
from core.utilities.enums import LogSource
from core.utilities.exceptions import DatabaseError

from core.database.models import Base

logger = get_logger(LogSource.DATABASE)


class DatabaseEngine:
    """Owns the SQLAlchemy engine and hands out transactional sessions."""

    def __init__(self, db_path: str | Path, *, echo: bool = False) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_engine(
            f"sqlite:///{self._db_path.as_posix()}",
            echo=echo,
            # Sessions are confined to one thread at a time by design, but the
            # pool may hand a connection to a different thread later.
            connect_args={"check_same_thread": False, "timeout": 30},
        )

        @event.listens_for(self._engine, "connect")
        def _set_pragmas(dbapi_conn, _record) -> None:  # noqa: ANN001
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")  # required for cascade deletes
            cursor.close()

        self._session_factory = sessionmaker(
            bind=self._engine, expire_on_commit=False, autoflush=False
        )

    # ------------------------------------------------------------------ setup
    @property
    def db_path(self) -> Path:
        return self._db_path

    def create_schema(self) -> None:
        """Create any missing tables (idempotent, called once at startup)."""
        try:
            Base.metadata.create_all(self._engine)
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Schema creation failed: {exc}") from exc
        logger.info("Database schema verified at %s", self._db_path)

    # --------------------------------------------------------------- sessions
    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Transactional scope: commit on success, rollback + DatabaseError on failure."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            raise DatabaseError(str(exc)) from exc
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ---------------------------------------------------------------- backup
    def backup_to(self, dest_path: str | Path) -> Path:
        """Consistent online backup using SQLite's native backup API (WAL-safe).

        Returns the destination path.

        Raises:
            DatabaseError: backup failed.
        """
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw = self._engine.raw_connection()
            try:
                source_conn: sqlite3.Connection = raw.driver_connection  # type: ignore[assignment]
                with sqlite3.connect(dest) as target_conn:
                    source_conn.backup(target_conn)
            finally:
                raw.close()
        except (sqlite3.Error, SQLAlchemyError) as exc:
            raise DatabaseError(f"Backup to {dest} failed: {exc}") from exc
        logger.info("Database backed up to %s", dest)
        return dest

    def dispose(self) -> None:
        """Close all pooled connections (application shutdown)."""
        self._engine.dispose()
        logger.info("Database engine disposed")

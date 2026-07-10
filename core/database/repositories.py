"""Repository layer — the only place SQL/ORM queries are written.

Each repository receives the shared :class:`DatabaseEngine` and opens a
short-lived session per operation, so repositories are safe to call from any
thread (the DatabaseWorker, the UI thread for reads, the backup timer).
Returned ORM objects are detached but fully loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.orm import selectinload

from core.database.db_engine import DatabaseEngine
from core.database.models import (
    Calibration,
    CameraConfiguration,
    Inspection,
    LogEntry,
    PlcConfiguration,
    SystemConfiguration,
    User,
    now_local,
)
from core.logging import get_logger
from core.utilities.enums import InspectionResult, LogSource

logger = get_logger(LogSource.DATABASE)


class BaseRepository:
    """Common constructor: repositories share one engine, never one session."""

    def __init__(self, db: DatabaseEngine) -> None:
        self._db = db


# --------------------------------------------------------------------------- #
# Inspections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InspectionFilter:
    """Search criteria for the Database Viewer. ``None`` means 'no filter'."""

    date_from: datetime | None = None
    date_to: datetime | None = None
    machine_number: int | None = None
    result: str | None = None
    serial_number: str | None = None


@dataclass(frozen=True)
class DailyCounts:
    """Dashboard counters for one production day."""

    total: int = 0
    good: int = 0
    ng: int = 0


class InspectionRepository(BaseRepository):
    """CRUD and analytics for inspections + their per-camera details."""

    def add(self, inspection: Inspection) -> int:
        """Persist an inspection (with its details) and return its new id."""
        with self._db.session_scope() as session:
            session.add(inspection)
            session.flush()  # assign primary key before scope closes
            return inspection.id

    def get_recent(self, limit: int = 50) -> list[Inspection]:
        """Latest inspections, newest first, details eagerly loaded."""
        with self._db.session_scope() as session:
            stmt = (
                select(Inspection)
                .options(selectinload(Inspection.details))
                .order_by(Inspection.created_at.desc(), Inspection.id.desc())
                .limit(limit)
            )
            return list(session.scalars(stmt).all())

    def search(
        self,
        criteria: InspectionFilter,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[Inspection], int]:
        """Filtered, paginated search. Returns ``(rows, total_match_count)``."""
        conditions = []
        if criteria.date_from is not None:
            conditions.append(Inspection.created_at >= criteria.date_from)
        if criteria.date_to is not None:
            conditions.append(Inspection.created_at < criteria.date_to)
        if criteria.machine_number is not None:
            conditions.append(Inspection.machine_number == criteria.machine_number)
        if criteria.result:
            conditions.append(Inspection.overall_result == criteria.result)
        if criteria.serial_number:
            conditions.append(Inspection.serial_number.contains(criteria.serial_number))

        with self._db.session_scope() as session:
            total = session.scalar(
                select(func.count()).select_from(Inspection).where(*conditions)
            ) or 0
            stmt = (
                select(Inspection)
                .options(selectinload(Inspection.details))
                .where(*conditions)
                .order_by(Inspection.created_at.desc(), Inspection.id.desc())
                .offset(offset)
                .limit(limit)
            )
            return list(session.scalars(stmt).all()), total

    def daily_counts(self, day: date | None = None) -> DailyCounts:
        """GOOD/NG/total counters for *day* (default: today) — dashboard tiles."""
        day = day or date.today()
        start = datetime.combine(day, time.min)
        end = start + timedelta(days=1)
        with self._db.session_scope() as session:
            rows = session.execute(
                select(Inspection.overall_result, func.count())
                .where(Inspection.created_at >= start, Inspection.created_at < end)
                .group_by(Inspection.overall_result)
            ).all()
        counts = {result: count for result, count in rows}
        return DailyCounts(
            total=sum(counts.values()),
            good=counts.get(InspectionResult.GOOD.value, 0),
            ng=counts.get(InspectionResult.NG.value, 0),
        )

    def purge_older_than(self, days: int) -> int:
        """Retention: delete inspections older than *days* (details cascade). Returns row count."""
        cutoff = now_local() - timedelta(days=days)
        with self._db.session_scope() as session:
            result = session.execute(delete(Inspection).where(Inspection.created_at < cutoff))
        deleted = result.rowcount or 0
        if deleted:
            logger.info("Retention purge removed %d inspections older than %d days", deleted, days)
        return deleted


# --------------------------------------------------------------------------- #
# Configuration mirrors
# --------------------------------------------------------------------------- #
class CameraConfigRepository(BaseRepository):
    """Audit mirror of camera settings (JSON files remain the boot source)."""

    def get_all(self) -> list[CameraConfiguration]:
        with self._db.session_scope() as session:
            stmt = select(CameraConfiguration).order_by(CameraConfiguration.camera_index)
            return list(session.scalars(stmt).all())

    def upsert(self, values: dict) -> None:
        """Insert or update by ``camera_index`` (must be present in *values*)."""
        camera_index = values["camera_index"]
        with self._db.session_scope() as session:
            row = session.scalar(
                select(CameraConfiguration).where(
                    CameraConfiguration.camera_index == camera_index
                )
            )
            if row is None:
                session.add(CameraConfiguration(**values))
            else:
                for key, value in values.items():
                    setattr(row, key, value)

    def delete_by_index(self, camera_index: int) -> None:
        with self._db.session_scope() as session:
            session.execute(
                delete(CameraConfiguration).where(
                    CameraConfiguration.camera_index == camera_index
                )
            )


class PlcConfigRepository(BaseRepository):
    """Audit mirror of the PLC settings (single row, id=1)."""

    ROW_ID = 1

    def get(self) -> PlcConfiguration | None:
        with self._db.session_scope() as session:
            return session.get(PlcConfiguration, self.ROW_ID)

    def save(self, values: dict) -> None:
        with self._db.session_scope() as session:
            row = session.get(PlcConfiguration, self.ROW_ID)
            if row is None:
                session.add(PlcConfiguration(id=self.ROW_ID, **values))
            else:
                for key, value in values.items():
                    setattr(row, key, value)


class SystemConfigRepository(BaseRepository):
    """Key/value store for application settings."""

    def get(self, key: str, default: str = "") -> str:
        with self._db.session_scope() as session:
            row = session.get(SystemConfiguration, key)
            return row.value if row is not None else default

    def set(self, key: str, value: str, description: str = "") -> None:
        with self._db.session_scope() as session:
            row = session.get(SystemConfiguration, key)
            if row is None:
                session.add(
                    SystemConfiguration(key=key, value=value, description=description)
                )
            else:
                row.value = value
                if description:
                    row.description = description


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
class CalibrationRepository(BaseRepository):
    """Calibration history; exactly one active row per camera."""

    def get_active(self, camera_index: int) -> Calibration | None:
        with self._db.session_scope() as session:
            stmt = (
                select(Calibration)
                .where(Calibration.camera_index == camera_index, Calibration.is_active)
                .order_by(Calibration.calibrated_at.desc())
                .limit(1)
            )
            return session.scalar(stmt)

    def get_all_active(self) -> dict[int, Calibration]:
        """Active calibration per camera index (missing cameras absent from dict)."""
        with self._db.session_scope() as session:
            stmt = (
                select(Calibration)
                .where(Calibration.is_active)
                .order_by(Calibration.camera_index, Calibration.calibrated_at.desc())
            )
            result: dict[int, Calibration] = {}
            for row in session.scalars(stmt):
                result.setdefault(row.camera_index, row)  # newest wins per camera
            return result

    def save(self, calibration: Calibration) -> int:
        """Deactivate the camera's previous calibration and store the new one as active."""
        with self._db.session_scope() as session:
            for old in session.scalars(
                select(Calibration).where(
                    Calibration.camera_index == calibration.camera_index,
                    Calibration.is_active,
                )
            ):
                old.is_active = False
            calibration.is_active = True
            session.add(calibration)
            session.flush()
            return calibration.id


# --------------------------------------------------------------------------- #
# Logs
# --------------------------------------------------------------------------- #
class LogRepository(BaseRepository):
    """Persisted log records for the Logs page."""

    def add_many(self, entries: list[LogEntry]) -> None:
        """Bulk insert (DatabaseWorker batches records to reduce commits)."""
        if not entries:
            return
        with self._db.session_scope() as session:
            session.add_all(entries)

    def query(
        self,
        *,
        level: str | None = None,
        source: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        text: str | None = None,
        offset: int = 0,
        limit: int = 500,
    ) -> tuple[list[LogEntry], int]:
        """Filtered, paginated log query. Returns ``(rows, total_match_count)``."""
        conditions = []
        if level:
            conditions.append(LogEntry.level == level)
        if source:
            conditions.append(LogEntry.source == source)
        if date_from is not None:
            conditions.append(LogEntry.created_at >= date_from)
        if date_to is not None:
            conditions.append(LogEntry.created_at < date_to)
        if text:
            conditions.append(LogEntry.message.contains(text))

        with self._db.session_scope() as session:
            total = session.scalar(
                select(func.count()).select_from(LogEntry).where(*conditions)
            ) or 0
            stmt = (
                select(LogEntry)
                .where(*conditions)
                .order_by(LogEntry.created_at.desc(), LogEntry.id.desc())
                .offset(offset)
                .limit(limit)
            )
            return list(session.scalars(stmt).all()), total

    def clear(self) -> int:
        """Delete all log rows (Logs page 'Clear Logs'). Returns row count."""
        with self._db.session_scope() as session:
            result = session.execute(delete(LogEntry))
        return result.rowcount or 0

    def purge_older_than(self, days: int) -> int:
        cutoff = now_local() - timedelta(days=days)
        with self._db.session_scope() as session:
            result = session.execute(delete(LogEntry).where(LogEntry.created_at < cutoff))
        return result.rowcount or 0


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
class UserRepository(BaseRepository):
    """Account storage. Hashing/verification is AuthService's job — only
    opaque hashes cross this boundary."""

    def get_by_username(self, username: str) -> User | None:
        with self._db.session_scope() as session:
            return session.scalar(select(User).where(User.username == username))

    def create(self, username: str, password_hash: str, role: str) -> int:
        with self._db.session_scope() as session:
            user = User(username=username, password_hash=password_hash, role=role)
            session.add(user)
            session.flush()
            return user.id

    def update_password(self, username: str, password_hash: str) -> bool:
        with self._db.session_scope() as session:
            user = session.scalar(select(User).where(User.username == username))
            if user is None:
                return False
            user.password_hash = password_hash
            return True

    def touch_last_login(self, username: str) -> None:
        with self._db.session_scope() as session:
            user = session.scalar(select(User).where(User.username == username))
            if user is not None:
                user.last_login = now_local()

    def count(self) -> int:
        with self._db.session_scope() as session:
            return session.scalar(select(func.count()).select_from(User)) or 0

"""Database backup and retention maintenance.

``run_maintenance()`` is invoked periodically (timer in the composition
root): it takes a timestamped online backup when auto-backup is enabled and
at most once per calendar day, purges inspections/logs beyond the retention
window, and prunes old backup files.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from core.database import DatabaseEngine
from core.logging import get_logger
from core.utilities import ConfigManager
from core.utilities.enums import LogSource
from core.utilities.exceptions import DatabaseError
from services.database_service import DatabaseService

logger = get_logger(LogSource.DATABASE)

LAST_BACKUP_KEY = "last_auto_backup_date"  # stored in system_configurations


class BackupService:
    """Backup files: ``<backup_directory>/inspection_YYYYMMDD_HHMMSS.db``."""

    def __init__(
        self,
        db_engine: DatabaseEngine,
        database_service: DatabaseService,
        config_manager: ConfigManager,
    ) -> None:
        self._engine = db_engine
        self._database = database_service
        self._config = config_manager

    # --------------------------------------------------------------- actions
    def backup_now(self) -> Path:
        """Immediate online backup (WAL-safe). Raises DatabaseError."""
        backup_dir = Path(self._db_config().get("backup_directory", "backups"))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._engine.backup_to(backup_dir / f"inspection_{stamp}.db")

    def run_retention(self) -> tuple[int, int]:
        """Purge old inspections and logs; returns (inspections, logs) deleted."""
        days = int(self._db_config().get("retention_days", 90))
        if days <= 0:
            return 0, 0
        return (
            self._database.inspections.purge_older_than(days),
            self._database.logs.purge_older_than(days),
        )

    def prune_backups(self, keep: int = 30) -> int:
        """Keep the newest *keep* backup files, delete the rest."""
        backup_dir = Path(self._db_config().get("backup_directory", "backups"))
        if not backup_dir.is_dir():
            return 0
        backups = sorted(backup_dir.glob("inspection_*.db"), reverse=True)
        removed = 0
        for stale in backups[keep:]:
            try:
                stale.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("Could not delete old backup %s: %s", stale, exc)
        return removed

    def run_maintenance(self) -> None:
        """Periodic entry point; never raises (logs + continues)."""
        try:
            if self._auto_backup_due():
                self.backup_now()
                self._database.system.set(
                    LAST_BACKUP_KEY, date.today().isoformat(), "Set by BackupService"
                )
                self.prune_backups()
            purged = self.run_retention()
            if any(purged):
                logger.info(
                    "Retention purge: %d inspections, %d logs removed", *purged
                )
        except DatabaseError as exc:
            logger.error("Maintenance failed: %s", exc)

    # -------------------------------------------------------------- internal
    def _db_config(self) -> dict:
        return self._config.load("app_config").get("database", {})

    def _auto_backup_due(self) -> bool:
        if not self._db_config().get("auto_backup", True):
            return False
        last = self._database.system.get(LAST_BACKUP_KEY, "")
        return last != date.today().isoformat()

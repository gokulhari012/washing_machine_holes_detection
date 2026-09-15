"""Retention covers the saved pictures, not only the database rows.

``BackupService.run_image_retention`` deletes ``images/<YYYY-MM-DD>/``
folders older than ``database.retention_days``. The rules worth pinning: a
folder exactly on the cutoff stays, ``0`` disables the purge entirely, and
anything not named as a plain ISO date is left alone (an operator may have
parked a folder of reference shots in there).
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from core.utilities.config_manager import ConfigManager
from services.backup_service import BackupService


def _service(tmp_path: Path, retention_days: int) -> BackupService:
    config_dir = tmp_path / "config"
    (config_dir / "defaults").mkdir(parents=True)
    doc = {
        "database": {"retention_days": retention_days, "auto_backup": False},
        "storage": {"image_directory": str(tmp_path / "images")},
    }
    (config_dir / "app_config.json").write_text(json.dumps(doc), encoding="utf-8")
    manager = ConfigManager(config_dir)
    return BackupService(db_engine=None, database_service=None, config_manager=manager)


def _day_dir(tmp_path: Path, name: str) -> Path:
    folder = tmp_path / "images" / name
    folder.mkdir(parents=True)
    (folder / "120000_1_cam1_good.png").write_bytes(b"x")
    return folder


def test_old_image_folders_are_deleted_and_recent_ones_kept(tmp_path):
    old = _day_dir(tmp_path, (date.today() - timedelta(days=91)).isoformat())
    edge = _day_dir(tmp_path, (date.today() - timedelta(days=90)).isoformat())
    recent = _day_dir(tmp_path, date.today().isoformat())

    removed = _service(tmp_path, 90).run_image_retention()

    assert removed == 1
    assert not old.exists()
    assert edge.exists()  # the cutoff day itself is inside the window
    assert recent.exists()


def test_zero_retention_keeps_every_image(tmp_path):
    ancient = _day_dir(tmp_path, "2000-01-01")

    assert _service(tmp_path, 0).run_image_retention() == 0
    assert ancient.exists()


def test_non_date_folders_are_never_touched(tmp_path):
    keep = _day_dir(tmp_path, "reference-shots")
    _day_dir(tmp_path, "2000-01-01")

    assert _service(tmp_path, 90).run_image_retention() == 1
    assert keep.exists()


def test_missing_image_directory_is_not_an_error(tmp_path):
    assert _service(tmp_path, 90).run_image_retention() == 0

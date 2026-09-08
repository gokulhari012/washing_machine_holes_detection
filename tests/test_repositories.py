"""Repository layer against a real temporary SQLite file (WAL, cascades)."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import text

from core.database import (
    Calibration,
    CalibrationRepository,
    DatabaseEngine,
    Inspection,
    InspectionDetail,
    InspectionFilter,
    InspectionRepository,
    UserRepository,
)


@pytest.fixture()
def db(tmp_path) -> DatabaseEngine:
    engine = DatabaseEngine(tmp_path / "test.db")
    engine.create_schema()
    yield engine
    engine.dispose()


def make_inspection(machine: int, result: str, when: datetime | None = None) -> Inspection:
    inspection = Inspection(
        machine_number=machine,
        serial_number=f"WM-{machine}",
        overall_result=result,
        created_at=when or datetime.now(),
    )
    for camera in range(1, 5):
        inspection.details.append(
            InspectionDetail(camera_index=camera, result=result, hole_found=True)
        )
    return inspection


def test_add_search_counts(db) -> None:
    repo = InspectionRepository(db)
    for machine, result in ((1, "GOOD"), (2, "NG"), (3, "GOOD")):
        repo.add(make_inspection(machine, result))

    rows, total = repo.search(InspectionFilter(), limit=10)
    assert total == 3 and len(rows[0].details) == 4

    rows, total = repo.search(InspectionFilter(result="NG"), limit=10)
    assert total == 1 and rows[0].machine_number == 2

    counts = repo.daily_counts()
    assert (counts.total, counts.good, counts.ng) == (3, 2, 1)


def test_purge_cascades(db) -> None:
    repo = InspectionRepository(db)
    repo.add(make_inspection(9, "NG", when=datetime.now() - timedelta(days=400)))
    repo.add(make_inspection(10, "GOOD"))
    assert repo.purge_older_than(90) == 1
    with db.session_scope() as session:
        orphans = session.execute(
            text(
                "SELECT COUNT(*) FROM inspection_details d "
                "LEFT JOIN inspections i ON d.inspection_id = i.id WHERE i.id IS NULL"
            )
        ).scalar()
    assert orphans == 0


def test_calibration_single_active(db) -> None:
    repo = CalibrationRepository(db)
    repo.save(Calibration(camera_index=1, pixels_per_mm_x=9.0, pixels_per_mm_y=9.0))
    repo.save(Calibration(camera_index=1, pixels_per_mm_x=10.0, pixels_per_mm_y=10.0))
    active = repo.get_active(1)
    assert active is not None and active.pixels_per_mm_x == 10.0
    assert list(repo.get_all_active().keys()) == [1]


def test_users(db) -> None:
    repo = UserRepository(db)
    assert repo.count() == 0
    repo.create("admin", "hash", "admin")
    user = repo.get_by_username("admin")
    assert user is not None and user.role == "admin"
    assert repo.update_password("admin", "hash2") is True
    assert repo.get_by_username("admin").password_hash == "hash2"


def test_search_filters_by_shift_exactly(db) -> None:
    """Shift names come from a fixed rota, so the filter matches whole names —
    'Night' must not also pull in 'Late Night'."""
    repo = InspectionRepository(db)
    for machine, shift in ((1, "Morning"), (2, "Night"), (3, "Late Night"), (4, "Night")):
        inspection = make_inspection(machine, "GOOD")
        inspection.shift = shift
        repo.add(inspection)

    _rows, total = repo.search(InspectionFilter(shift="Night"), limit=10)
    assert total == 2

    _rows, total = repo.search(InspectionFilter(shift="Morning"), limit=10)
    assert total == 1

    # no shift filter -> every row, including those stamped with nothing
    _rows, total = repo.search(InspectionFilter(), limit=10)
    assert total == 4


def test_shift_filter_combines_with_the_other_criteria(db) -> None:
    repo = InspectionRepository(db)
    for machine, shift, result in ((1, "Night", "GOOD"), (2, "Night", "NG"), (3, "Morning", "NG")):
        inspection = make_inspection(machine, result)
        inspection.shift = shift
        repo.add(inspection)

    rows, total = repo.search(InspectionFilter(shift="Night", result="NG"), limit=10)
    assert total == 1 and rows[0].machine_number == 2

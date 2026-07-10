"""Database facade: owns the repositories and the DTO↔ORM mapping.

Pages and workers reach persistence exclusively through this service (or the
repositories it exposes); nothing outside ``core/database`` writes SQL.
"""

from __future__ import annotations

from models.dto import InspectionCycleData, LogEvent

from core.database import (
    CalibrationRepository,
    CameraConfigRepository,
    DatabaseEngine,
    DailyCounts,
    Inspection,
    InspectionDetail,
    InspectionRepository,
    LogEntry,
    LogRepository,
    PlcConfigRepository,
    SystemConfigRepository,
    UserRepository,
)
from core.logging import get_logger
from core.utilities.enums import LogSource

logger = get_logger(LogSource.DATABASE)


class DatabaseService:
    """One engine, all repositories, and the inspection mapping."""

    def __init__(self, db_engine: DatabaseEngine) -> None:
        self._engine = db_engine
        self.inspections = InspectionRepository(db_engine)
        self.logs = LogRepository(db_engine)
        self.users = UserRepository(db_engine)
        self.camera_configs = CameraConfigRepository(db_engine)
        self.plc_config = PlcConfigRepository(db_engine)
        self.system = SystemConfigRepository(db_engine)
        self.calibrations = CalibrationRepository(db_engine)

    @property
    def engine(self) -> DatabaseEngine:
        return self._engine

    # ------------------------------------------------------------ inspections
    def save_inspection(self, cycle: InspectionCycleData) -> int:
        """Map a cycle DTO to ORM rows and persist; sets ``cycle.inspection_id``.

        Raises:
            DatabaseError
        """
        inspection = Inspection(
            created_at=cycle.started_at,
            machine_number=cycle.machine_number,
            serial_number=cycle.serial_number,
            overall_result=cycle.overall_result.value,
            plc_cycle_time_ms=cycle.plc_cycle_time_ms,
            detection_time_ms=cycle.detection_time_ms,
            operator=cycle.operator,
            shift=cycle.shift,
        )
        for index in sorted(cycle.cameras):
            data = cycle.cameras[index]
            inspection.details.append(
                InspectionDetail(
                    camera_index=data.camera_index,
                    camera_name=data.camera_name,
                    hole_found=data.hole_found,
                    x_px=data.x_px,
                    y_px=data.y_px,
                    x_mm=data.x_mm,
                    y_mm=data.y_mm,
                    deviation_mm=data.deviation_mm if data.deviation_mm is not None else 0.0,
                    confidence=data.confidence,
                    result=data.result.value,
                    image_path=data.image_path,
                )
            )
        cycle.inspection_id = self.inspections.add(inspection)
        return cycle.inspection_id

    def daily_counts(self) -> DailyCounts:
        return self.inspections.daily_counts()

    # ------------------------------------------------------------------ logs
    def save_log_events(self, events: list[LogEvent]) -> None:
        """Batch-persist log events (called by the database worker).

        Raises:
            DatabaseError
        """
        self.logs.add_many(
            [
                LogEntry(
                    created_at=event.created_at,
                    level=event.level,
                    source=event.source,
                    message=event.message,
                )
                for event in events
            ]
        )

"""Data-transfer objects that cross thread boundaries.

DTOs are plain dataclasses (no Qt, no ORM) so any layer can construct or read
them. They travel through queued Qt signals as opaque ``object`` payloads and
must be treated as immutable after emission — the ``frame`` arrays included
for display are never written to downstream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from core.logging import ROOT_LOGGER_NAME
from core.utilities.enums import InspectionResult, LogSource
from core.vision.detection_result import DetectionResult


@dataclass
class CameraInspectionData:
    """Per-camera outcome of one inspection cycle."""

    camera_index: int
    camera_name: str
    result: InspectionResult
    hole_found: bool = False
    x_px: float = 0.0
    y_px: float = 0.0
    x_mm: float = 0.0
    y_mm: float = 0.0
    deviation_mm: float | None = None  # None = camera not calibrated
    confidence: float = 0.0
    error: str = ""
    image_path: str | None = None
    detection: DetectionResult | None = field(default=None, repr=False)
    #: annotated frame for the dashboard panel (display-only, do not mutate)
    frame: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class InspectionCycleData:
    """One complete trigger cycle — the payload of ``inspection_completed``."""

    machine_number: int
    serial_number: str
    started_at: datetime
    overall_result: InspectionResult
    cameras: dict[int, CameraInspectionData] = field(default_factory=dict)
    plc_cycle_time_ms: float = 0.0   # trigger seen -> PLC output written
    detection_time_ms: float = 0.0   # parallel detection phase only
    operator: str = ""
    shift: str = ""
    plc_write_ok: bool = True
    inspection_id: int | None = None  # database id, set after persistence


@dataclass(frozen=True)
class LogEvent:
    """Lightweight log record for the live Logs page and the DB writer."""

    created_at: datetime
    level: str
    source: str   # LogSource value
    message: str

    @classmethod
    def from_record(cls, record: logging.LogRecord) -> "LogEvent":
        """Map a stdlib LogRecord; the subsystem comes from the logger name
        (``wmhd.plc`` -> ``PLC``), unknown names fall back to SYSTEM."""
        source = LogSource.SYSTEM.value
        prefix = f"{ROOT_LOGGER_NAME}."
        if record.name.startswith(prefix):
            candidate = record.name[len(prefix):].split(".", 1)[0].upper()
            if candidate in LogSource.__members__:
                source = LogSource[candidate].value
        return cls(
            created_at=datetime.fromtimestamp(record.created),
            level=record.levelname,
            source=source,
            message=record.getMessage(),
        )

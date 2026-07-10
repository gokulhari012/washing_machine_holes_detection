"""Holds the active calibration per camera and applies it in the hot path.

Loads active rows from the database once at startup; ``save`` writes through
the repository (which archives the previous calibration) and refreshes the
cache. When a camera has never been calibrated, ``evaluate`` falls back to
the **identity mapping (1 px = 1 mm)** and reports no deviation, so the
system runs out-of-the-box and the tolerance judgement stays disabled until
a real calibration exists.
"""

from __future__ import annotations

import threading

from core.calibration.calibration_model import CameraCalibration
from core.database.repositories import CalibrationRepository
from core.logging import get_logger
from core.utilities.enums import LogSource

logger = get_logger(LogSource.VISION)


class CalibrationManager:
    """Thread-safe cache of active calibrations over the repository."""

    def __init__(self, repository: CalibrationRepository) -> None:
        self._repository = repository
        self._lock = threading.Lock()
        self._calibrations: dict[int, CameraCalibration] = {}

    # -------------------------------------------------------------- loading
    def load_all(self) -> None:
        """Populate the cache from the database (startup / after restore)."""
        rows = self._repository.get_all_active()
        with self._lock:
            self._calibrations = {
                index: CameraCalibration.from_row(row) for index, row in rows.items()
            }
        logger.info(
            "Loaded calibrations for cameras: %s",
            sorted(self._calibrations) or "none",
        )

    # -------------------------------------------------------------- access
    def get(self, camera_index: int) -> CameraCalibration | None:
        with self._lock:
            return self._calibrations.get(camera_index)

    def has(self, camera_index: int) -> bool:
        return self.get(camera_index) is not None

    def save(self, calibration: CameraCalibration) -> int:
        """Persist as the camera's new active calibration; returns row id."""
        row_id = self._repository.save(calibration.to_row())
        with self._lock:
            self._calibrations[calibration.camera_index] = calibration
        logger.info(
            "Calibration saved for camera %d (rms %.3f mm)",
            calibration.camera_index,
            calibration.rms_error,
        )
        return row_id

    # ------------------------------------------------------------- hot path
    def evaluate(
        self, camera_index: int, x_px: float, y_px: float
    ) -> tuple[float, float, float | None]:
        """Convert a pixel position and judge it against the reference point.

        Returns:
            ``(x_mm, y_mm, deviation_mm)`` — ``deviation_mm`` is ``None`` when
            the camera is uncalibrated (identity fallback, tolerance check
            not applicable).
        """
        calibration = self.get(camera_index)
        if calibration is None:
            return float(x_px), float(y_px), None
        x_mm, y_mm = calibration.pixel_to_mm(x_px, y_px)
        return x_mm, y_mm, calibration.deviation_mm(x_mm, y_mm)

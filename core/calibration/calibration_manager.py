"""Holds the active calibration per camera and applies it in the hot path.

Loads active rows from the database once at startup; ``save`` writes through
the repository (which archives the previous calibration) and refreshes the
cache. When a camera has never been calibrated, ``evaluate`` falls back to
the **identity mapping (1 px = 1 mm)** and reports no deviation, so the
system runs out-of-the-box and the tolerance judgement stays disabled until
a real calibration exists.

``evaluate`` reports position relative to the analysed image's own centre
(pass ``image_width``/``image_height``) rather than the calibration's raw
coordinate frame — see its docstring.
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

    def apply_live(self, calibration: CameraCalibration) -> None:
        """Push *calibration* into the live cache without persisting or
        archiving history — mirrors ``CameraService.apply_live`` /
        ``VisionEngine.apply_config``'s "preview, don't persist" pattern.
        Used when a machine-model profile switches (see
        ``MachineModelService.apply_profile``), so the manually-tuned
        Calibration-page baseline in the database is never overwritten by an
        automatic model switch.
        """
        with self._lock:
            self._calibrations[calibration.camera_index] = calibration
        logger.info(
            "Calibration applied live for camera %d (not persisted)", calibration.camera_index
        )

    # ------------------------------------------------------------- hot path
    def evaluate(
        self,
        camera_index: int,
        x_px: float,
        y_px: float,
        image_width: float | None = None,
        image_height: float | None = None,
    ) -> tuple[float, float, float | None]:
        """Convert a pixel position and judge it against the reference point.

        When *image_width*/*image_height* are supplied (the dimensions of the
        analysed image — the ROI crop, or the full frame with no ROI), the
        returned ``(x_mm, y_mm)`` is re-based so the image's own centre reads
        as ``(0, 0)`` — the convention the PLC output, dashboard and database
        use, so a hole sitting exactly at the centre of frame always reports
        ``(0, 0)`` regardless of the camera's calibrated coordinate frame.
        The tolerance judgement (``deviation_mm``) is computed *before* that
        re-basing, against the reference point in the calibration's own
        (uncentred) frame, so re-basing never shifts the GOOD/NG verdict.

        Returns:
            ``(x_mm, y_mm, deviation_mm)`` — ``deviation_mm`` is ``None`` when
            the camera is uncalibrated (identity fallback, tolerance check
            not applicable).
        """
        calibration = self.get(camera_index)
        if calibration is None:
            x_mm, y_mm = float(x_px), float(y_px)
            if image_width and image_height:
                x_mm -= image_width / 2
                y_mm -= image_height / 2
            return x_mm, y_mm, None
        x_mm, y_mm = calibration.pixel_to_mm(x_px, y_px)
        deviation = calibration.deviation_mm(x_mm, y_mm)
        if image_width and image_height:
            center_x_mm, center_y_mm = calibration.pixel_to_mm(image_width / 2, image_height / 2)
            x_mm -= center_x_mm
            y_mm -= center_y_mm
        return x_mm, y_mm, deviation

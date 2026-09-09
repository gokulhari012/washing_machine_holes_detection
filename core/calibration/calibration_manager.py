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

``save`` also fans out to plain-callable observers (:meth:`subscribe`), the
same Qt-free notification ``ConfigManager`` uses for a settings file, so the
composition root can fold a freshly persisted calibration into the active
machine-model profile without ``core/`` knowing that services exist.
"""

from __future__ import annotations

import threading
from typing import Callable

from core.calibration.calibration_model import CameraCalibration
from core.database.repositories import CalibrationRepository
from core.logging import get_logger
from core.utilities.enums import LogSource

logger = get_logger(LogSource.VISION)

CalibrationCallback = Callable[[CameraCalibration], None]


class CalibrationManager:
    """Thread-safe cache of active calibrations over the repository."""

    def __init__(self, repository: CalibrationRepository) -> None:
        self._repository = repository
        self._lock = threading.Lock()
        self._calibrations: dict[int, CameraCalibration] = {}
        self._screw_compensation_enabled = False
        self._screw_offsets: dict[int, tuple[float, float]] = {}
        self._observers: list[CalibrationCallback] = []

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
        self._notify(calibration)
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

    # ----------------------------------------------------------- observers
    def subscribe(self, callback: CalibrationCallback) -> None:
        """Register *callback*, invoked with each calibration :meth:`save`
        persists.

        Persisted saves only — :meth:`apply_live` deliberately does not
        notify, because a machine-model switch pushing its own snapshot back
        into the cache is not the operator calibrating anything, and folding
        it into a profile would be circular.

        Callbacks run on the saving thread (in practice the GUI thread, which
        is where the Calibration page's Save button lives); keep them short.
        An exception in one is logged and never reaches the saver.
        """
        with self._lock:
            self._observers.append(callback)

    def unsubscribe(self, callback: CalibrationCallback) -> None:
        """Remove a previously registered callback (no-op if absent)."""
        with self._lock:
            try:
                self._observers.remove(callback)
            except ValueError:
                pass

    def _notify(self, calibration: CameraCalibration) -> None:
        with self._lock:
            observers = list(self._observers)
        for callback in observers:
            try:
                callback(calibration)
            except Exception:  # an observer must never break a save
                logger.exception("Calibration observer failed")

    # ---------------------------------------------- screw driver compensation
    def apply_screw_compensation(
        self, enabled: bool, positions: dict[int, tuple[float, float]]
    ) -> None:
        """Set the live per-camera screw-driver position offsets.

        Preview-only, like :meth:`apply_live`: pushed by
        ``MachineModelService.apply_profile`` from the selected profile's
        ``screw_driver_compensation`` block. When *enabled*, :meth:`screw_offset`
        returns the configured (x_mm, y_mm) for a camera instead of (0, 0) — see
        its docstring for where that offset is used.
        """
        with self._lock:
            self._screw_compensation_enabled = enabled
            self._screw_offsets = dict(positions)

    def screw_offset(self, camera_index: int) -> tuple[float, float]:
        """The (x_mm, y_mm) offset to add to *camera_index*'s PLC-bound
        position — never to the measured position reported to the dashboard
        or stored in the database, which must stay the true detected hole
        location. Returns (0.0, 0.0) when compensation is disabled or no
        offset is configured for that camera. See
        ``InspectionService._plc_position``, the only caller.
        """
        with self._lock:
            if not self._screw_compensation_enabled:
                return 0.0, 0.0
            return self._screw_offsets.get(camera_index, (0.0, 0.0))

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

        Re-basing also **flips the Y sign**: pixel rows grow downward, but the
        reported coordinate follows the machine/servo convention where up is
        positive, so a hole below the image centre reports a *negative* y_mm.
        X is untouched (rightward is positive in both conventions).

        The tolerance judgement (``deviation_mm``) is computed *before* that
        re-basing, against the reference point in the calibration's own
        (uncentred, image-oriented) frame, so re-basing never shifts the
        GOOD/NG verdict.

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
                y_mm = image_height / 2 - y_mm
            return x_mm, y_mm, None
        x_mm, y_mm = calibration.pixel_to_mm(x_px, y_px)
        deviation = calibration.deviation_mm(x_mm, y_mm)
        if image_width and image_height:
            center_x_mm, center_y_mm = calibration.pixel_to_mm(image_width / 2, image_height / 2)
            x_mm -= center_x_mm
            y_mm = center_y_mm - y_mm
        return x_mm, y_mm, deviation

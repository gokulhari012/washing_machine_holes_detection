"""Holds the active calibration per camera and applies it in the hot path.

Loads active rows from the database once at startup; ``save`` writes through
the repository (which archives the previous calibration) and refreshes the
cache. When a camera has never been calibrated, ``evaluate`` falls back to
the **identity mapping (1 px = 1 mm)** and reports no deviation, so the
system runs out-of-the-box and the tolerance judgement stays disabled until
a real calibration exists.

``evaluate`` reports position relative to the analysed image's own centre
(pass ``image_width``/``image_height``) rather than the calibration's raw
coordinate frame, with each camera's configured axis signs applied — see its
docstring.

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

        The camera's **axis signs** survive the swap: they say which corner
        that gantry homes at, which no machine model can change, and a profile
        snapshot does not carry them (``CameraCalibration.to_dict``). Taking
        *calibration*'s defaults instead would quietly un-invert an axis on
        every model switch and drive the servo the wrong way. *calibration* is
        updated in place, so callers that keep a reference see the same signs
        the cache does — ``apply_profile`` builds a fresh object per switch.

        The camera's **screw driver compensation** survives the swap for the
        same reason and by the same mechanism (see
        :meth:`save_screw_compensation`).
        """
        with self._lock:
            previous = self._calibrations.get(calibration.camera_index)
            if previous is not None:
                calibration.invert_x = previous.invert_x
                calibration.invert_y = previous.invert_y
                calibration.screw_compensation_enabled = previous.screw_compensation_enabled
                calibration.screw_offset_x_mm = previous.screw_offset_x_mm
                calibration.screw_offset_y_mm = previous.screw_offset_y_mm
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
    def save_screw_compensation(
        self, camera_index: int, enabled: bool, x_mm: float, y_mm: float
    ) -> bool:
        """Persist *camera_index*'s screw-driver offset and apply it at once.

        The cache is updated as part of the save, so the very next inspection
        encodes the new offset — this is a live machine setting, not a
        snapshot waiting on a machine-model switch, and an operator who
        presses Save expects the next part to move.

        Writes through :meth:`CalibrationRepository.update_screw_compensation`,
        which updates the camera's active row in place rather than archiving a
        new calibration: adjusting where the screw driver sits is not a
        recalibration. Returns ``False`` (and changes nothing) for a camera
        that has no calibration to attach the offset to.

        Observers are deliberately **not** notified: :meth:`subscribe` means
        "the operator calibrated this camera", which folds into the active
        machine-model profile — and the offsets are rig facts a profile does
        not carry (``CameraCalibration.to_dict``).
        """
        with self._lock:
            calibration = self._calibrations.get(camera_index)
        if calibration is None:
            return False
        if not self._repository.update_screw_compensation(
            camera_index, bool(enabled), float(x_mm), float(y_mm)
        ):
            return False
        with self._lock:
            # Re-read under the lock: apply_live may have swapped the object.
            live = self._calibrations.get(camera_index, calibration)
            live.screw_compensation_enabled = bool(enabled)
            live.screw_offset_x_mm = float(x_mm)
            live.screw_offset_y_mm = float(y_mm)
        logger.info(
            "Screw driver compensation for camera %d: %s (%.2f, %.2f) mm",
            camera_index,
            "on" if enabled else "off",
            x_mm,
            y_mm,
        )
        return True

    def screw_offset(self, camera_index: int) -> tuple[float, float]:
        """The (x_mm, y_mm) offset to add to *camera_index*'s PLC-bound
        position — never to the measured position reported to the dashboard
        or stored in the database, which must stay the true detected hole
        location. Returns (0.0, 0.0) when the camera is uncalibrated or its
        compensation is switched off. See ``InspectionService._plc_position``,
        the only caller.
        """
        with self._lock:
            calibration = self._calibrations.get(camera_index)
            if calibration is None or not calibration.screw_compensation_enabled:
                return 0.0, 0.0
            return calibration.screw_offset_x_mm, calibration.screw_offset_y_mm

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

        On top of that, the camera's own ``invert_x``/``invert_y`` flags flip
        each axis again, so a station whose four gantries home at four
        different corners can make "toward the part's centre" read positive on
        all of them (Calibration page → Axis direction; see
        :class:`~core.calibration.calibration_model.CameraCalibration`). Both
        flips belong to the *centre-relative report*, which is why they are
        applied here and not in ``pixel_to_mm``: a call without image
        dimensions asks for the calibration's own frame — it is the internal
        hole-ranking helper (``InspectionService._select_hole``), not something
        anyone reads a sign off — and is left alone.

        The tolerance judgement (``deviation_mm``) is computed *before* that
        re-basing, against the reference point in the calibration's own
        (uncentred, image-oriented) frame, so neither re-basing nor an axis
        sign ever shifts the GOOD/NG verdict.

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
            if calibration.invert_x:
                x_mm = -x_mm
            if calibration.invert_y:
                y_mm = -y_mm
        return x_mm, y_mm, deviation

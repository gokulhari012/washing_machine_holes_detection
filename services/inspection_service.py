"""The inspection pipeline — the single place the production workflow lives.

    trigger (machine number) ──► capture + detect per camera
                                 (sequential, one camera at a time, or
                                  all four in parallel — see capture_mode)
                                 ──► calibrate px→mm + judge per camera
                                 ──► overall result
                                 ──► write PLC outputs (positions/result/complete)
                                 ──► save annotated images (per storage policy)
                                 ──► persist to database
                                 ──► publish to AppState (dashboard)

The x_mm/y_mm reported to the PLC, dashboard and database are relative to
the analysed (ROI-cropped) image's own centre — a hole exactly centred in
frame always reports (0, 0) — and Y points *up*, so a hole below the centre
reports a negative y_mm. See ``CalibrationManager.evaluate``.

Capture modes (``app_config.inspection``):

- ``sequential`` (default) — camera 1 is grabbed, judged and shown on the
  dashboard, then ``camera_delay_ms`` passes, then camera 2, and so on. One
  camera is busy at a time, which is what a station with four high-resolution
  GigE cameras on a shared network link wants, and it lets the operator watch
  the pictures arrive one by one.
- ``parallel`` — all cameras grab and detect at once (fastest cycle, needs the
  bandwidth for it).

Every cycle is stamped with the operator and the shift it ran in. The shift
comes from ``ShiftService`` -- resolved from the configured rota against the
cycle's own ``started_at``, so a cycle that straddles a handover is filed
under the shift it *began* in, and a cycle replayed from a queue would be
filed correctly too.

Fault policy (a production line must keep moving):
- one dead camera            → that camera reports ERROR, others proceed
- detection algorithm error  → that camera reports ERROR
- no hole found              → NG (a legitimate result, not an error)
- PLC write failure          → cycle still persisted + alarm; PLC result code
                               would have been ERROR-safe on the PLC side via
                               its own vision-complete timeout
- database failure           → alarm; the PLC handshake is NOT rolled back

Runs on the inspection worker thread; never on the UI thread.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.calibration import CalibrationManager
from core.camera import CameraManager
from core.logging import get_logger
from core.plc import PlcManager
from core.utilities import ConfigManager
from core.utilities.enums import InspectionResult, LogSource, PlcResultCode
from core.utilities.exceptions import CameraError, DatabaseError, DetectionError, PlcError
from core.vision import DetectionResult, VisionEngine, draw_detection_overlay
from models.app_state import AppState
from models.dto import CameraInspectionData, InspectionCycleData
from services.database_service import DatabaseService
from services.shift_service import ShiftService

logger = get_logger(LogSource.VISION)

SEQUENTIAL_MODE = "sequential"
DEFAULT_CAMERA_DELAY_MS = 500

_RESULT_TO_PLC = {
    InspectionResult.GOOD: PlcResultCode.GOOD,
    InspectionResult.NG: PlcResultCode.NG,
    InspectionResult.ERROR: PlcResultCode.ERROR,
}


class InspectionService:
    """Orchestrates one complete inspection cycle per PLC trigger."""

    def __init__(
        self,
        camera_manager: CameraManager,
        vision_engine: VisionEngine,
        calibration_manager: CalibrationManager,
        plc_manager: PlcManager,
        database_service: DatabaseService,
        app_state: AppState,
        config_manager: ConfigManager,
        shift_service: ShiftService,
    ) -> None:
        self._cameras = camera_manager
        self._vision = vision_engine
        self._calibration = calibration_manager
        self._plc = plc_manager
        self._database = database_service
        self._app_state = app_state
        self._config = config_manager
        self._shifts = shift_service

    # -------------------------------------------------------------- pipeline
    def run_inspection(self, machine_number: int) -> InspectionCycleData:
        """Execute the full cycle. Never raises — faults degrade the result."""
        cycle_started = time.perf_counter()
        started_at = datetime.now()
        app_cfg = self._config.load("app_config")
        application = app_cfg.get("application", {})

        self._app_state.notify_trigger(machine_number)
        logger.info("Inspection started (machine %d)", machine_number)

        enabled = sorted(
            index
            for index, camera in self._cameras.cameras.items()
            if camera.settings.enabled
        )

        # 1.+2. capture and detect, one camera at a time or all at once
        inspection_cfg = app_cfg.get("inspection", {})
        if str(inspection_cfg.get("capture_mode", SEQUENTIAL_MODE)).lower() == SEQUENTIAL_MODE:
            camera_results, detection_ms = self._run_sequential(enabled, inspection_cfg)
        else:
            camera_results, detection_ms = self._run_parallel(enabled)

        # 3. overall judgement
        overall = self._overall_result(camera_results)

        # 4. PLC output — positions for found holes, sentinel otherwise, plus
        # each camera's own GOOD/NG/ERROR verdict alongside its position
        positions = {
            index: (data.x_mm, data.y_mm) if data.hole_found else None
            for index, data in camera_results.items()
        }
        plc_camera_results = {
            index: _RESULT_TO_PLC[data.result] for index, data in camera_results.items()
        }
        plc_write_ok = True
        try:
            self._plc.write_inspection_output(positions, plc_camera_results, _RESULT_TO_PLC[overall])
        except PlcError as exc:
            plc_write_ok = False
            logger.error("PLC output write failed: %s", exc)
            self._app_state.raise_alarm("error", f"PLC write failed: {exc}")

        cycle = InspectionCycleData(
            machine_number=machine_number,
            serial_number=self._serial_number(application, machine_number),
            started_at=started_at,
            overall_result=overall,
            cameras=camera_results,
            plc_cycle_time_ms=(time.perf_counter() - cycle_started) * 1000.0,
            detection_time_ms=detection_ms,
            operator=str(application.get("operator_name", "")),
            shift=self._shifts.current_name(started_at),
            plc_write_ok=plc_write_ok,
        )

        # 5. annotated image storage (policy from app_config.storage)
        self._save_images(cycle, app_cfg.get("storage", {}))

        # 6. persistence — a DB fault must not break the PLC handshake
        try:
            self._database.save_inspection(cycle)
        except DatabaseError as exc:
            logger.error("Inspection persistence failed: %s", exc)
            self._app_state.raise_alarm("error", f"Database write failed: {exc}")

        # 7. dashboard
        self._app_state.publish_inspection(cycle)
        logger.info(
            "Inspection finished (machine %d): %s in %.0f ms (detect %.0f ms)",
            machine_number,
            overall.value,
            cycle.plc_cycle_time_ms,
            detection_ms,
        )
        return cycle

    def run_camera_inspection(
        self, camera_index: int, machine_number: int
    ) -> InspectionCycleData:
        """Inspect **one** camera and publish only that camera's registers.

        Raised by a per-camera PLC trigger register, or by the dashboard's
        per-camera Trigger button. The other cameras are not captured, not
        judged and not written — their PLC registers keep the values from
        whichever cycle last set them.

        The resulting cycle is marked :attr:`InspectionCycleData.partial`, so
        it is stored and shown like any other inspection but does not count
        towards the product counters — one camera is not a finished product.

        Never raises — faults degrade the result, exactly as in the full cycle.
        """
        cycle_started = time.perf_counter()
        started_at = datetime.now()
        app_cfg = self._config.load("app_config")
        application = app_cfg.get("application", {})

        self._app_state.notify_trigger(machine_number)
        camera = self._cameras.get(camera_index)
        logger.info(
            "Single-camera inspection started (camera %d, machine %d)",
            camera_index,
            machine_number,
        )
        self._app_state.post_status(f"Capturing {camera.name} (single)")

        try:
            frame = self._cameras.capture(camera_index)
        except CameraError:
            frame = None  # already logged and recorded in health by capture()
        if frame is not None:
            self._app_state.publish_camera_capture(camera_index, frame)

        detect_started = time.perf_counter()
        data = self._inspect_one(camera_index, frame)
        detection_ms = (time.perf_counter() - detect_started) * 1000.0
        self._app_state.publish_camera_result(camera_index, data)

        plc_write_ok = True
        try:
            self._plc.write_camera_inspection_output(
                camera_index,
                (data.x_mm, data.y_mm) if data.hole_found else None,
                _RESULT_TO_PLC[data.result],
            )
        except PlcError as exc:
            plc_write_ok = False
            logger.error("PLC output write failed for camera %d: %s", camera_index, exc)
            self._app_state.raise_alarm("error", f"PLC write failed: {exc}")

        cycle = InspectionCycleData(
            machine_number=machine_number,
            serial_number=self._serial_number(application, machine_number),
            started_at=started_at,
            overall_result=data.result,
            cameras={camera_index: data},
            plc_cycle_time_ms=(time.perf_counter() - cycle_started) * 1000.0,
            detection_time_ms=detection_ms,
            operator=str(application.get("operator_name", "")),
            shift=self._shifts.current_name(started_at),
            plc_write_ok=plc_write_ok,
            partial=True,
        )

        self._save_images(cycle, app_cfg.get("storage", {}))
        try:
            self._database.save_inspection(cycle)
        except DatabaseError as exc:
            logger.error("Inspection persistence failed: %s", exc)
            self._app_state.raise_alarm("error", f"Database write failed: {exc}")

        self._app_state.publish_inspection(cycle)
        logger.info(
            "Single-camera inspection finished (camera %d): %s in %.0f ms",
            camera_index,
            data.result.value,
            cycle.plc_cycle_time_ms,
        )
        return cycle

    # --------------------------------------------------------- capture modes
    def _run_sequential(
        self, enabled: list[int], inspection_cfg: dict
    ) -> tuple[dict[int, CameraInspectionData], float]:
        """One camera at a time: grab, show, judge, wait, next camera.

        Each picture is published to the dashboard the moment it is taken, so
        the operator sees the cameras working through the part in order rather
        than four panels updating at the end.
        """
        delay_s = max(0, int(inspection_cfg.get("camera_delay_ms", DEFAULT_CAMERA_DELAY_MS))) / 1000.0
        camera_results: dict[int, CameraInspectionData] = {}
        detection_ms = 0.0

        for position, index in enumerate(enabled, start=1):
            if position > 1 and delay_s > 0:
                time.sleep(delay_s)  # inspection thread only; the PLC keeps polling

            camera = self._cameras.get(index)
            self._app_state.post_status(
                f"Capturing {camera.name} ({position}/{len(enabled)})"
            )
            try:
                frame = self._cameras.capture(index)
            except CameraError:
                frame = None  # already logged and recorded in health by capture()
            if frame is not None:
                self._app_state.publish_camera_capture(index, frame)

            detect_started = time.perf_counter()
            data = self._inspect_one(index, frame)
            detection_ms += (time.perf_counter() - detect_started) * 1000.0

            camera_results[index] = data
            self._app_state.publish_camera_result(index, data)

        return camera_results, detection_ms

    def _run_parallel(
        self, enabled: list[int]
    ) -> tuple[dict[int, CameraInspectionData], float]:
        """All cameras grab together, then detect together (shortest cycle)."""
        frames = self._cameras.capture_all()
        for index, frame in frames.items():
            if frame is not None:
                self._app_state.publish_camera_capture(index, frame)

        detect_started = time.perf_counter()
        camera_results: dict[int, CameraInspectionData] = {}
        with ThreadPoolExecutor(
            max_workers=max(1, len(enabled)), thread_name_prefix="detect"
        ) as pool:
            futures = {
                index: pool.submit(self._inspect_one, index, frames.get(index))
                for index in enabled
            }
            for index, future in futures.items():
                camera_results[index] = future.result()
        detection_ms = (time.perf_counter() - detect_started) * 1000.0

        for index, data in camera_results.items():
            self._app_state.publish_camera_result(index, data)
        return camera_results, detection_ms

    # ------------------------------------------------------------ per camera
    def _inspect_one(
        self, camera_index: int, frame: np.ndarray | None
    ) -> CameraInspectionData:
        """Capture-to-judgement for a single camera. Never raises."""
        camera_name = self._cameras.get(camera_index).name

        if frame is None:
            health = self._cameras.health(camera_index)
            message = health.last_error or "capture failed"
            self._app_state.raise_alarm(
                "error", f"{camera_name}: {message}"
            )
            return CameraInspectionData(
                camera_index=camera_index,
                camera_name=camera_name,
                result=InspectionResult.ERROR,
                error=message,
            )

        try:
            detection = self._vision.detect(frame, camera_index)
        except DetectionError as exc:
            logger.error("%s: detection error: %s", camera_name, exc)
            self._app_state.raise_alarm("error", f"{camera_name}: {exc}")
            return CameraInspectionData(
                camera_index=camera_index,
                camera_name=camera_name,
                result=InspectionResult.ERROR,
                error=str(exc),
                frame=draw_detection_overlay(frame, None, label=camera_name),
            )

        self._select_hole(camera_index, detection)
        annotated = draw_detection_overlay(frame, detection, label=camera_name)
        best = detection.best
        if best is None or len(detection.holes) < self._vision.expected_hole_count(camera_index):
            return CameraInspectionData(
                camera_index=camera_index,
                camera_name=camera_name,
                result=InspectionResult.NG,
                hole_found=False,
                detection=detection,
                frame=annotated,
            )

        image_height, image_width = frame.shape[:2]
        x_mm, y_mm, deviation = self._calibration.evaluate(
            camera_index, best.x_px, best.y_px, image_width, image_height
        )
        tolerance = self._vision.position_tolerance_mm(camera_index)
        out_of_tolerance = (
            deviation is not None and tolerance > 0 and deviation > tolerance
        )

        return CameraInspectionData(
            camera_index=camera_index,
            camera_name=camera_name,
            result=InspectionResult.NG if out_of_tolerance else InspectionResult.GOOD,
            hole_found=True,
            x_px=best.x_px,
            y_px=best.y_px,
            x_mm=x_mm,
            y_mm=y_mm,
            deviation_mm=deviation,
            confidence=best.confidence,
            detection=detection,
            frame=annotated,
        )

    def _select_hole(self, camera_index: int, detection: DetectionResult) -> None:
        """Reorder ``detection.holes`` so the candidate the pipeline should
        judge is first (:attr:`DetectionResult.best`), so it is also the one
        drawn as the primary (green) circle by ``draw_detection_overlay``.

        A station may legitimately have more than one real hole in frame —
        without a calibrated reference point there is no way to tell them
        apart, so an uncalibrated camera keeps the previous behaviour
        (highest confidence, already ``holes[0]``). A calibrated camera picks
        whichever accepted candidate lands closest to the reference point
        instead: the highest-confidence candidate is not necessarily the
        *right* hole for that station, and letting it win anyway is what
        produces jumpy X/Y readings and spurious NG results when several real
        holes score similarly.
        """
        holes = detection.holes
        if len(holes) < 2 or not self._calibration.has(camera_index):
            return
        nearest = min(
            holes,
            key=lambda hole: self._calibration.evaluate(camera_index, hole.x_px, hole.y_px)[2],
        )
        if nearest is not holes[0]:
            holes.remove(nearest)
            holes.insert(0, nearest)

    # -------------------------------------------------------------- internal
    @staticmethod
    def _overall_result(
        camera_results: dict[int, CameraInspectionData]
    ) -> InspectionResult:
        if not camera_results:
            return InspectionResult.ERROR
        results = {data.result for data in camera_results.values()}
        if InspectionResult.ERROR in results:
            return InspectionResult.ERROR
        if InspectionResult.NG in results:
            return InspectionResult.NG
        return InspectionResult.GOOD

    def _serial_number(self, application_cfg: dict, machine_number: int) -> str:
        """``serial_prefix`` + the serial the PLC published for this machine.

        The number comes from the PLC's serial-number register, read on this
        (inspection) thread the same way servo homes are. The prefix stays a
        PC-side setting — the PLC publishes a number, never the text. When
        the register is not configured, or the read fails, the machine number
        stands in for it, which is exactly what the serial was before the
        register existed; a cycle is never failed over a serial.
        """
        prefix = str(application_cfg.get("serial_prefix", ""))
        number: int | None = None
        try:
            number = self._plc.read_serial_number()
        except PlcError as exc:
            logger.warning("Serial number read failed, using the machine number: %s", exc)
        if number is None:
            number = machine_number
        return f"{prefix}{number:06d}"

    def _save_images(self, cycle: InspectionCycleData, storage_cfg: dict) -> None:
        if not storage_cfg.get("save_images", True):
            return
        ng_only = bool(storage_cfg.get("save_ng_only", False))
        base_dir = Path(storage_cfg.get("image_directory", "images"))
        day_dir = base_dir / cycle.started_at.strftime("%Y-%m-%d")

        for index, data in cycle.cameras.items():
            if data.frame is None:
                continue
            if ng_only and data.result is InspectionResult.GOOD:
                continue
            try:
                day_dir.mkdir(parents=True, exist_ok=True)
                filename = (
                    f"{cycle.started_at:%H%M%S}_{cycle.machine_number}"
                    f"_cam{index}_{data.result.value}.png"
                )
                path = day_dir / filename
                if cv2.imwrite(str(path), data.frame):
                    data.image_path = str(path)
                else:
                    logger.warning("Image save failed (cv2 refused): %s", path)
            except OSError as exc:
                logger.warning("Image save failed for camera %d: %s", index, exc)

"""The inspection pipeline — the single place the production workflow lives.

    trigger (machine number) ──► capture ×4 (parallel)
                                 ──► detect ×4 (parallel)
                                 ──► calibrate px→mm + judge per camera
                                 ──► overall result
                                 ──► write PLC outputs (positions/result/complete)
                                 ──► save annotated images (per storage policy)
                                 ──► persist to database
                                 ──► publish to AppState (dashboard)

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
from core.utilities.exceptions import DatabaseError, DetectionError, PlcError
from core.vision import VisionEngine, draw_detection_overlay
from models.app_state import AppState
from models.dto import CameraInspectionData, InspectionCycleData
from services.database_service import DatabaseService

logger = get_logger(LogSource.VISION)

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
    ) -> None:
        self._cameras = camera_manager
        self._vision = vision_engine
        self._calibration = calibration_manager
        self._plc = plc_manager
        self._database = database_service
        self._app_state = app_state
        self._config = config_manager

    # -------------------------------------------------------------- pipeline
    def run_inspection(self, machine_number: int) -> InspectionCycleData:
        """Execute the full cycle. Never raises — faults degrade the result."""
        cycle_started = time.perf_counter()
        started_at = datetime.now()
        app_cfg = self._config.load("app_config")
        application = app_cfg.get("application", {})

        self._app_state.notify_trigger(machine_number)
        logger.info("Inspection started (machine %d)", machine_number)

        # 1. capture all enabled+connected cameras in parallel
        frames = self._cameras.capture_all()
        enabled = [
            index
            for index, camera in self._cameras.cameras.items()
            if camera.settings.enabled
        ]

        # 2. detect in parallel (engine serialises non-thread-safe strategies)
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

        # 3. overall judgement
        overall = self._overall_result(camera_results)

        # 4. PLC output — positions for found holes, sentinel otherwise
        positions = {
            index: (data.x_mm, data.y_mm) if data.hole_found else None
            for index, data in camera_results.items()
        }
        plc_write_ok = True
        try:
            self._plc.write_inspection_output(positions, _RESULT_TO_PLC[overall])
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
            shift=str(application.get("shift", "")),
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
            detection = self._vision.detect(frame)
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

        annotated = draw_detection_overlay(frame, detection, label=camera_name)
        best = detection.best
        if best is None or len(detection.holes) < self._vision.expected_hole_count:
            return CameraInspectionData(
                camera_index=camera_index,
                camera_name=camera_name,
                result=InspectionResult.NG,
                hole_found=False,
                detection=detection,
                frame=annotated,
            )

        x_mm, y_mm, deviation = self._calibration.evaluate(
            camera_index, best.x_px, best.y_px
        )
        tolerance = self._vision.position_tolerance_mm
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

    @staticmethod
    def _serial_number(application_cfg: dict, machine_number: int) -> str:
        prefix = str(application_cfg.get("serial_prefix", ""))
        return f"{prefix}{machine_number:06d}"

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

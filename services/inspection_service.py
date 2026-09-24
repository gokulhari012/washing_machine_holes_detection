"""The inspection pipeline — the single place the production workflow lives.

    trigger (machine number) ──► read each camera's gantry status
                                 ──► capture + detect per *active* camera
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

Strobe lighting
----------------
A camera whose ``led_strobe`` is on (Camera page) gets its configured LED
Controller channel lit only while it is needed for a capture — never left lit
between cycles, same as the Camera page's own Test Camera / Continuous
Capture strobing (see ``CameraService.light_on``/``light_off``).

The global trigger (``run_inspection``, both capture modes) lights every
strobe-enabled, gantry-active camera together in *one* grouped serial command
(``LedManager.send_multichannel``, the controller's documented multi-channel
T/F frame) immediately before the cycle's first capture, and turns them all
off again in one more grouped command immediately after the last capture —
so in sequential mode all of this cycle's lights stay on together across the
``camera_delay_ms`` gaps between cameras too, not just each camera's own
capture window, and in parallel mode this replaces what used to be a loop of
individual per-channel commands with a single simultaneous write. A channel
not being toggled this way (a non-strobe camera's steady light) is re-sent at
its own already-configured brightness rather than left out of the frame,
because the controller's multi-channel command always sets all 4 channels in
one write — see ``_channel_states``.

The per-camera trigger (``run_camera_inspection``, register 132+N) is
unaffected: there is only ever one camera to light there, so it still
brackets just that camera's own grab with a single-channel command via
``_strobe_on``/``_strobe_off``.

Best-effort, like every other LED write: a communication failure is logged,
never raised, so it can never stall or fail an inspection cycle. Skipped
(gantry-inactive) cameras are never strobed, since they are never captured
either. ``led_manager`` is optional — a service built without one (tests, or
a station that hasn't finished the LED integration) simply never strobes.

Every cycle is stamped with the operator and the shift it ran in. The shift
comes from ``ShiftService`` -- resolved from the configured rota against the
cycle's own ``started_at``, so a cycle that straddles a handover is filed
under the shift it *began* in, and a cycle replayed from a queue would be
filed correctly too.

Gantry gating
-------------
Every cycle — the global one and a single-camera one alike — starts by reading
each camera's ``gantry_status`` register. A camera whose gantry the PLC reports
inactive is **skipped**: not captured, not detected, and none of its PLC
registers written, so the last cycle that really inspected it still owns them.
It is still recorded, as ``InspectionResult.SKIPPED``, and shown on its
dashboard panel, so a part inspected by two of four cameras is visibly that
rather than silently short. A skipped camera takes no part in the overall
verdict; a cycle in which *every* camera was skipped has nothing to judge and
reports ERROR, the same as a cycle with no enabled cameras at all.

A camera with no gantry-status register configured is always active, so a
station that has not wired the register up behaves exactly as before. A failed
read is also treated as active, deliberately: a comms glitch that silently
stopped inspecting a camera would ship an un-inspected part, which is worse
than capturing one whose gantry happens to be parked.

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
from core.led import LedManager
from core.logging import get_logger
from core.plc import PlcManager
from core.utilities import ConfigManager
from core.utilities.enums import InspectionResult, LogSource, PlcResultCode
from core.utilities.exceptions import CameraError, DatabaseError, DetectionError, LedError, PlcError
from core.vision import DetectionResult, VisionEngine, draw_detection_overlay
from models.app_state import AppState
from models.dto import CameraInspectionData, InspectionCycleData
from services.database_service import DatabaseService
from services.shift_service import ShiftService

logger = get_logger(LogSource.VISION)

SEQUENTIAL_MODE = "sequential"
PARALLEL_MODE = "parallel"


def _resolve_capture_mode(value: object) -> str:
    """Normalise a configured/overridden capture mode to a known one.

    Anything unrecognised resolves to :data:`SEQUENTIAL_MODE` — the station's
    documented default — and says so once per cycle in the log. It used to fall
    through to the parallel branch simply because it was not equal to
    ``"sequential"``, which turned a typo in ``app_config.json``
    (``"parellel"``) into a silent parallel cycle with ``camera_delay_ms``
    ignored and nothing anywhere saying why.
    """
    mode = str(value).strip().lower()
    if mode in (SEQUENTIAL_MODE, PARALLEL_MODE):
        return mode
    logger.warning(
        "Unknown capture_mode %r - running %s. Valid values are %r and %r.",
        value,
        SEQUENTIAL_MODE,
        SEQUENTIAL_MODE,
        PARALLEL_MODE,
    )
    return SEQUENTIAL_MODE


def select_hole(
    calibration: CalibrationManager, camera_index: int, detection: DetectionResult
) -> None:
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

    Module-level so the Detection page's "Test on Camera" judges the same
    candidate a real cycle would.
    """
    holes = detection.holes
    if len(holes) < 2 or not calibration.has(camera_index):
        return
    nearest = min(
        holes,
        key=lambda hole: calibration.evaluate(camera_index, hole.x_px, hole.y_px)[2],
    )
    if nearest is not holes[0]:
        holes.remove(nearest)
        holes.insert(0, nearest)


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
        led_manager: LedManager | None = None,
    ) -> None:
        self._cameras = camera_manager
        self._vision = vision_engine
        self._calibration = calibration_manager
        self._plc = plc_manager
        self._database = database_service
        self._app_state = app_state
        self._config = config_manager
        self._shifts = shift_service
        self._led = led_manager

    # -------------------------------------------------------------- pipeline
    def run_inspection(
        self, machine_number: int, capture_mode: str | None = None
    ) -> InspectionCycleData:
        """Execute the full cycle. Never raises — faults degrade the result.

        ``capture_mode`` overrides ``app_config.inspection.capture_mode`` for
        this one cycle without persisting anything — the toolbar's "all cameras
        at a time" button passes :data:`PARALLEL_MODE` so it grabs every camera
        at once regardless of the station's configured (normally sequential)
        mode. A PLC trigger passes nothing and follows the configuration.
        """
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

        # 0. gantry gating — only the cameras the PLC says are in position
        active, skipped = self._resolve_gantries(enabled)

        # 1.+2. capture and detect, one camera at a time or all at once.
        # All of this cycle's strobe lights go on together in one grouped
        # command before either path starts, and off together in one more
        # once it returns — see the module docstring's Strobe lighting note.
        inspection_cfg = app_cfg.get("inspection", {})
        mode = _resolve_capture_mode(
            capture_mode or inspection_cfg.get("capture_mode", SEQUENTIAL_MODE)
        )
        lit = self._strobe_group_on(active)
        try:
            if mode == SEQUENTIAL_MODE:
                camera_results, detection_ms = self._run_sequential(active, inspection_cfg)
            else:
                camera_results, detection_ms = self._run_parallel(active)
        finally:
            self._strobe_group_off(lit)

        # the skipped cameras are recorded and shown, but judged by nobody
        for index in skipped:
            data = self._skipped_data(index)
            camera_results[index] = data
            self._app_state.publish_camera_result(index, data)
        camera_results = {index: camera_results[index] for index in sorted(camera_results)}

        # 3. overall judgement (skipped cameras take no part in it)
        overall = self._overall_result(camera_results)

        # 4. PLC output — positions for found holes, sentinel otherwise, plus
        # each camera's own GOOD/NG/ERROR verdict alongside its position. A
        # skipped camera contributes neither: its registers are left holding
        # whatever the last cycle that really inspected it wrote. A camera
        # with an active screw-driver compensation offset gets it added here
        # only — the recorded/dashboard x_mm/y_mm on `data` stays the raw
        # measured hole position (see `_plc_position`).
        positions = {
            index: self._plc_position(index, data)
            for index, data in camera_results.items()
            if index not in skipped
        }
        plc_camera_results = {
            index: _RESULT_TO_PLC[data.result]
            for index, data in camera_results.items()
            if index not in skipped
        }
        plc_write_ok = True
        try:
            self._plc.write_inspection_output(
                positions,
                plc_camera_results,
                _RESULT_TO_PLC[overall],
                skipped=skipped,
            )
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

        The gantry gate applies here too: if the PLC reports this camera's
        gantry inactive, nothing is captured, judged or written to its
        position/result registers, but the handshake is still answered (its
        trigger released, its vision_complete raised) so the PLC that raised
        the trigger never dead-waits. The cycle is recorded as SKIPPED.

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

        _active, skipped = self._resolve_gantries([camera_index])
        if skipped:
            return self._skipped_camera_cycle(
                camera_index, machine_number, started_at, cycle_started, application
            )

        self._app_state.post_status(f"Capturing {camera.name} (single)")

        camera_started = time.perf_counter()
        lit = self._strobe_on([camera_index])
        try:
            frame = self._cameras.capture(camera_index)
        except CameraError:
            frame = None  # already logged and recorded in health by capture()
        finally:
            self._strobe_off(lit)
        if frame is not None:
            self._app_state.publish_camera_capture(camera_index, frame)

        detect_started = time.perf_counter()
        data = self._inspect_one(camera_index, frame)
        now = time.perf_counter()
        detection_ms = (now - detect_started) * 1000.0
        data.cycle_time_ms = (now - camera_started) * 1000.0
        self._app_state.publish_camera_result(camera_index, data)

        plc_write_ok = True
        try:
            self._plc.write_camera_inspection_output(
                camera_index,
                self._plc_position(camera_index, data),
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

    # -------------------------------------------------------- gantry gating
    def _resolve_gantries(self, enabled: list[int]) -> tuple[list[int], set[int]]:
        """Split *enabled* into the cameras to inspect and the ones to skip.

        Reads each camera's ``gantry_status`` register on this (inspection)
        thread, the same way servo homes and the serial number are read — the
        PLC may park a gantry between cycles, so the value is never cached.

        A camera with no gantry-status register configured is active, and so
        is one whose read fails: losing a register read must not silently stop
        inspecting a camera, because that ships an un-inspected part. The read
        failure is logged and raised as an alarm; the link being down is
        already visible elsewhere, and the cycle's own PLC write will fail
        loudly a moment later anyway.
        """
        active: list[int] = []
        skipped: set[int] = set()
        for index in enabled:
            try:
                gantry_ok = self._plc.read_gantry_status(index)
            except PlcError as exc:
                logger.warning(
                    "Gantry status read failed for camera %d, inspecting it anyway: %s",
                    index,
                    exc,
                )
                gantry_ok = True
            if gantry_ok:
                active.append(index)
            else:
                skipped.add(index)
        if skipped:
            logger.info(
                "Cameras skipped this cycle (gantry inactive): %s",
                ", ".join(str(index) for index in sorted(skipped)),
            )
        return active, skipped

    def _skipped_data(self, camera_index: int) -> CameraInspectionData:
        """The recorded outcome of a camera the PLC's gantry status ruled out.

        Carries no frame, no coordinates and no confidence — nothing was
        measured. ``error`` holds the reason rather than a fault message: the
        Database Viewer's detail column is the only place an operator can find
        out *why* a camera has no numbers for a given part.
        """
        return CameraInspectionData(
            camera_index=camera_index,
            camera_name=self._cameras.get(camera_index).name,
            result=InspectionResult.SKIPPED,
            error="gantry inactive",
        )

    def _skipped_camera_cycle(
        self,
        camera_index: int,
        machine_number: int,
        started_at: datetime,
        cycle_started: float,
        application_cfg: dict,
    ) -> InspectionCycleData:
        """Answer a per-camera trigger for a camera whose gantry is inactive.

        Completes the PLC handshake (trigger released, vision_complete raised)
        without writing a position or a result, then records and publishes the
        cycle as SKIPPED so the skip is auditable rather than invisible.
        """
        data = self._skipped_data(camera_index)
        self._app_state.publish_camera_result(camera_index, data)
        self._app_state.post_status(
            f"{data.camera_name} skipped — gantry inactive"
        )

        plc_write_ok = True
        try:
            self._plc.write_camera_skipped_output(camera_index)
        except PlcError as exc:
            plc_write_ok = False
            logger.error(
                "PLC handshake failed for skipped camera %d: %s", camera_index, exc
            )
            self._app_state.raise_alarm("error", f"PLC write failed: {exc}")

        cycle = InspectionCycleData(
            machine_number=machine_number,
            serial_number=self._serial_number(application_cfg, machine_number),
            started_at=started_at,
            overall_result=InspectionResult.SKIPPED,
            cameras={camera_index: data},
            plc_cycle_time_ms=(time.perf_counter() - cycle_started) * 1000.0,
            detection_time_ms=0.0,
            operator=str(application_cfg.get("operator_name", "")),
            shift=self._shifts.current_name(started_at),
            plc_write_ok=plc_write_ok,
            partial=True,
        )
        try:
            self._database.save_inspection(cycle)
        except DatabaseError as exc:
            logger.error("Inspection persistence failed: %s", exc)
            self._app_state.raise_alarm("error", f"Database write failed: {exc}")

        self._app_state.publish_inspection(cycle)
        logger.info(
            "Single-camera inspection skipped (camera %d): gantry inactive",
            camera_index,
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
            camera_started = time.perf_counter()
            try:
                frame = self._cameras.capture(index)
            except CameraError:
                frame = None  # already logged and recorded in health by capture()
            if frame is not None:
                self._app_state.publish_camera_capture(index, frame)

            detect_started = time.perf_counter()
            data = self._inspect_one(index, frame)
            now = time.perf_counter()
            detection_ms += (now - detect_started) * 1000.0
            data.cycle_time_ms = (now - camera_started) * 1000.0  # this camera's own capture+detect

            camera_results[index] = data
            self._app_state.publish_camera_result(index, data)

        return camera_results, detection_ms

    def _run_parallel(
        self, enabled: list[int]
    ) -> tuple[dict[int, CameraInspectionData], float]:
        """All cameras grab together, then detect together (shortest cycle)."""
        capture_started = time.perf_counter()
        # Only the gantry-active cameras: a skipped camera is not captured at
        # all (same as the sequential path), and grabbing it would cost a
        # full-resolution transfer nothing then reads.
        frames = self._cameras.capture_all(enabled)
        capture_ms = (time.perf_counter() - capture_started) * 1000.0
        for index, frame in frames.items():
            if frame is not None:
                self._app_state.publish_camera_capture(index, frame)

        detect_started = time.perf_counter()
        camera_results: dict[int, CameraInspectionData] = {}
        with ThreadPoolExecutor(
            max_workers=max(1, len(enabled)), thread_name_prefix="detect"
        ) as pool:
            futures = {
                index: pool.submit(self._timed_inspect_one, index, frames.get(index))
                for index in enabled
            }
            for index, future in futures.items():
                data, detect_ms = future.result()
                # the shared parallel capture phase plus this camera's own
                # detect time — the closest thing to "how long this camera
                # took" when every camera was grabbed in one batch.
                data.cycle_time_ms = capture_ms + detect_ms
                camera_results[index] = data
        detection_ms = (time.perf_counter() - detect_started) * 1000.0

        for index, data in camera_results.items():
            self._app_state.publish_camera_result(index, data)
        return camera_results, detection_ms

    def _timed_inspect_one(
        self, camera_index: int, frame: np.ndarray | None
    ) -> tuple[CameraInspectionData, float]:
        """``_inspect_one`` plus its own wall time, measured inside the worker
        thread — timing it from the submitting thread instead would measure
        queue wait, not the task's actual duration, once several run at once."""
        started = time.perf_counter()
        data = self._inspect_one(camera_index, frame)
        return data, (time.perf_counter() - started) * 1000.0

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
        """See :func:`select_hole`."""
        select_hole(self._calibration, camera_index, detection)

    def _plc_position(
        self, camera_index: int, data: CameraInspectionData
    ) -> tuple[float, float] | None:
        """The position written to the PLC for *camera_index*: the detected
        hole's x_mm/y_mm plus that camera's screw-driver offset, when screw
        driver compensation is enabled on that camera's calibration
        (Calibration page, Step 5 — see
        ``CalibrationManager.screw_offset``). ``None`` (the
        no-hole sentinel) when no hole was found, same as before compensation
        existed — an offset is never invented for a position that was never
        measured. The offset is applied here only: `data.x_mm`/`data.y_mm`
        themselves — what the dashboard and database show — are left as the
        true detected position.
        """
        if not data.hole_found:
            return None
        dx, dy = self._calibration.screw_offset(camera_index)
        return data.x_mm + dx, data.y_mm + dy

    # ------------------------------------------------------------ strobing
    def _strobe_on(self, indices: list[int]) -> list[int]:
        """Light every strobe-enabled camera among *indices*, just before
        their capture, via a single-channel command per camera. Used only by
        :meth:`run_camera_inspection` (the per-camera trigger), which only
        ever has one camera to light — the global trigger uses the grouped
        :meth:`_strobe_group_on` instead. Returns the subset actually
        addressed (strobe on, channel configured), so the matching
        :meth:`_strobe_off` turns off exactly what this turned on rather than
        re-deriving it from settings that could theoretically change
        mid-cycle.
        """
        if self._led is None:
            return []
        lit: list[int] = []
        for index in indices:
            settings = self._cameras.get(index).settings
            if not getattr(settings, "led_strobe", False):
                continue
            channel = getattr(settings, "led_channel", 0)
            if channel <= 0:
                continue
            try:
                self._led.send_channel(channel, settings.brightness)
            except LedError as exc:
                logger.warning(
                    "Camera %d: strobe light-on failed on LED channel %d (%s)",
                    index, channel, exc,
                )
            lit.append(index)
        return lit

    def _strobe_off(self, indices: list[int]) -> None:
        """The other half of :meth:`_strobe_on` — always called, capture
        success or failure, so a strobe channel is never left lit."""
        if self._led is None or not indices:
            return
        for index in indices:
            settings = self._cameras.get(index).settings
            channel = getattr(settings, "led_channel", 0)
            if channel <= 0:
                continue
            try:
                self._led.send_channel(channel, 0)
            except LedError as exc:
                logger.warning(
                    "Camera %d: strobe light-off failed on LED channel %d (%s)",
                    index, channel, exc,
                )

    def _strobe_group_on(self, indices: list[int]) -> list[int]:
        """Light every strobe-enabled camera among *indices* together, in one
        grouped multi-channel command, before any of this cycle's captures
        begin. Returns the subset of *indices* actually addressed (strobe on,
        channel configured), so the matching :meth:`_strobe_group_off` turns
        off exactly what this turned on.
        """
        if self._led is None:
            return []
        channel_of: dict[int, int] = {}
        for index in indices:
            settings = self._cameras.get(index).settings
            if not getattr(settings, "led_strobe", False):
                continue
            channel = getattr(settings, "led_channel", 0)
            if channel <= 0:
                continue
            channel_of[channel] = index
        if not channel_of:
            return []
        try:
            self._led.send_multichannel(self._channel_states(channel_of, on=True))
        except LedError as exc:
            logger.warning("Grouped strobe light-on failed: %s", exc)
        return list(channel_of.values())

    def _strobe_group_off(self, indices: list[int]) -> None:
        """The other half of :meth:`_strobe_group_on` — always called,
        capture success or failure, so no strobe channel from this cycle is
        left lit."""
        if self._led is None or not indices:
            return
        channel_of: dict[int, int] = {}
        for index in indices:
            channel = getattr(self._cameras.get(index).settings, "led_channel", 0)
            if channel > 0:
                channel_of[channel] = index
        if not channel_of:
            return
        try:
            self._led.send_multichannel(self._channel_states(channel_of, on=False))
        except LedError as exc:
            logger.warning("Grouped strobe light-off failed: %s", exc)

    def _channel_states(
        self, toggled: dict[int, int], *, on: bool
    ) -> list[tuple[int, bool]]:
        """Build the 4-channel (brightness, on) frame for
        :meth:`_strobe_group_on`/:meth:`_strobe_group_off`.

        ``toggled`` maps LED channel -> camera index for the channels this
        call is switching. The controller's multi-channel command always
        sets all 4 channels in one write, so every channel *not* in
        ``toggled`` that belongs to a non-strobe camera is re-sent at its own
        configured steady brightness (whatever
        ``CameraService._push_brightness`` last drove it to) instead of being
        left out of the frame — otherwise a grouped strobe write would blank
        out that camera's light for the duration of the cycle. A
        strobe-enabled camera's channel that isn't in ``toggled`` (skipped
        this cycle, e.g. a parked gantry) is *not* treated as steady — its
        resting state is off, same as ``_strobe_on``/``_strobe_off`` leave it
        between cycles — and neither is a channel nothing is wired to; both
        go out at (0, off). ``send_multichannel`` does not clamp to
        ``max_brightness`` itself (it also backs the raw hardware tester), so
        every value here is clamped before sending.
        """
        steady = {
            getattr(camera.settings, "led_channel", 0): camera.settings.brightness
            for camera in self._cameras.cameras.values()
            if getattr(camera.settings, "led_channel", 0) > 0
            and not getattr(camera.settings, "led_strobe", False)
        }
        clamp = self._led.settings.clamp
        states: list[tuple[int, bool]] = []
        for channel in (1, 2, 3, 4):
            if channel in toggled:
                brightness = self._cameras.get(toggled[channel]).settings.brightness if on else 0
                states.append((clamp(brightness), on))
            elif channel in steady:
                states.append((clamp(steady[channel]), True))
            else:
                states.append((0, False))
        return states

    # -------------------------------------------------------------- internal
    @staticmethod
    def _overall_result(
        camera_results: dict[int, CameraInspectionData]
    ) -> InspectionResult:
        """Worst verdict wins: ERROR > NG > GOOD.

        A SKIPPED camera never inspected the part, so it cannot vote — it is
        neither a pass nor a failure. If that leaves nothing to judge (every
        camera's gantry was inactive, or no camera is enabled) the cycle is an
        ERROR: the PLC asked for a machine to be inspected and none of it was,
        which is a station problem, not a good part.
        """
        results = {
            data.result
            for data in camera_results.values()
            if data.result is not InspectionResult.SKIPPED
        }
        if not results:
            return InspectionResult.ERROR
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

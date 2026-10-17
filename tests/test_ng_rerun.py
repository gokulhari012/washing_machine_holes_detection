"""NG re-run ("Required another run", Detection page -> Judgement -> On NG).

An NG camera captures a *fresh* frame and is judged again, up to its
``ng_rerun_count``; the first non-NG run is reported. What is pinned here:

- a re-run is a new capture, not a re-detection of the same frame;
- GOOD and ERROR never re-run, and a camera with the option off never does;
- a re-run that fails (ERROR) stops the loop and the measured NG stands;
- it works the same in sequential mode, parallel mode and the per-camera
  trigger, and only ever touches the camera that was NG.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.utilities.enums import InspectionResult, PositionReference
from core.utilities.exceptions import CameraCaptureError, DetectionError
from core.vision.detection_result import DetectionResult, Hole
from models.app_state import AppState
from services.inspection_service import InspectionService
from tests.test_inspection_sequence import FakeCameraManager, FakeConfig, FakeShifts

GOOD, NG, ERROR = "good", "ng", "error"


class ScriptedVision:
    """Answers each camera's detect() calls in turn from a script; a camera
    whose script has run out keeps answering GOOD."""

    def __init__(self, reruns: dict[int, int] | None = None, script: dict | None = None) -> None:
        self._reruns = reruns or {}
        self._script = {index: list(steps) for index, steps in (script or {}).items()}
        self.detect_log: list[int] = []

    def expected_hole_count(self, camera_index: int) -> int:
        return 1

    def position_tolerance_mm(self, camera_index: int) -> float:
        return 0.0

    def ng_rerun_count(self, camera_index: int) -> int:
        return self._reruns.get(camera_index, 0)

    def detect(self, frame, camera_index: int) -> DetectionResult:
        self.detect_log.append(camera_index)
        steps = self._script.get(camera_index)
        outcome = steps.pop(0) if steps else GOOD
        if outcome == ERROR:
            raise DetectionError("algorithm failed")
        if outcome == NG:
            return DetectionResult(holes=[])
        return DetectionResult(holes=[Hole(10.0, 12.0, 30.0, 0.9, 0.8)])


class FlakyCameraManager(FakeCameraManager):
    """Fails one camera's grab on the listed capture numbers (1-based)."""

    def __init__(self, fail_on: dict[int, set[int]]) -> None:
        super().__init__()
        self._fail_on = fail_on
        self._grabs: dict[int, int] = {}

    def capture(self, index: int, *, apply_roi: bool = True) -> np.ndarray:
        self._grabs[index] = self._grabs.get(index, 0) + 1
        self.capture_log.append(index)
        if self._grabs[index] in self._fail_on.get(index, set()):
            raise CameraCaptureError(f"Camera {index}: grab timeout")
        return np.zeros((40, 40, 3), dtype=np.uint8)


def build(vision: ScriptedVision, cameras=None, **inspection):
    cameras = cameras or FakeCameraManager()
    app_state = AppState()
    calibration = SimpleNamespace(
        has=lambda index: True,
        evaluate=lambda index, x, y, w=None, h=None: (1.0, 2.0, 0.0),
        screw_offset=lambda index: (0.0, 0.0),
        position_reference=lambda index: PositionReference.SCREW_HOME,
    )
    plc = SimpleNamespace(
        write_inspection_output=lambda positions, camera_results, result, **kwargs: None,
        write_camera_inspection_output=lambda index, position, result, **kwargs: None,
        read_serial_number=lambda: None,
        read_gantry_status=lambda index: True,
    )
    database = SimpleNamespace(save_inspection=lambda cycle: 1)
    config = FakeConfig(**{"capture_mode": "sequential", "camera_delay_ms": 0, **inspection})
    service = InspectionService(
        cameras, vision, calibration, plc, database, app_state, config, FakeShifts()
    )
    return service, cameras, app_state


def grabs_of(cameras, index: int) -> int:
    return cameras.capture_log.count(index)


# ------------------------------------------------------------ switched off
def test_an_ng_is_reported_as_is_when_the_option_is_off() -> None:
    vision = ScriptedVision(script={2: [NG, GOOD]})
    service, cameras, _ = build(vision)
    cycle = service.run_inspection(machine_number=1)
    assert cycle.cameras[2].result is InspectionResult.NG
    assert grabs_of(cameras, 2) == 1


# ----------------------------------------------------- re-run until good
def test_an_ng_recaptures_and_reports_the_first_good() -> None:
    vision = ScriptedVision(reruns={2: 3}, script={2: [NG, NG, GOOD]})
    service, cameras, _ = build(vision)
    cycle = service.run_inspection(machine_number=1)
    assert cycle.cameras[2].result is InspectionResult.GOOD
    assert cycle.overall_result is InspectionResult.GOOD
    assert grabs_of(cameras, 2) == 3  # stopped at the GOOD, not all 1 + 3
    assert [grabs_of(cameras, i) for i in (1, 3, 4)] == [1, 1, 1]


def test_every_run_is_a_fresh_capture_shown_on_the_dashboard() -> None:
    vision = ScriptedVision(reruns={1: 2}, script={1: [NG, GOOD]})
    service, cameras, app_state = build(vision)
    captured: list[int] = []
    app_state.camera_captured.connect(lambda index, frame: captured.append(index))
    service.run_inspection(machine_number=1)
    # camera 1 re-runs before camera 2 is touched — the order the operator sees
    assert cameras.capture_log == [1, 1, 2, 3, 4]
    assert captured == [1, 1, 2, 3, 4]
    assert vision.detect_log == [1, 1, 2, 3, 4]


def test_still_ng_after_every_rerun_is_reported_ng() -> None:
    vision = ScriptedVision(reruns={3: 2}, script={3: [NG, NG, NG, GOOD]})
    service, cameras, _ = build(vision)
    cycle = service.run_inspection(machine_number=1)
    assert cycle.cameras[3].result is InspectionResult.NG
    assert grabs_of(cameras, 3) == 3  # 1 + 2 re-runs, never the 4th


def test_the_verdict_is_published_once_per_camera() -> None:
    """The dashboard panel gets the final verdict, not every interim NG."""
    vision = ScriptedVision(reruns={1: 3}, script={1: [NG, NG, GOOD]})
    service, _, app_state = build(vision)
    published: list = []
    app_state.camera_inspected.connect(lambda index, data: published.append((index, data.result)))
    service.run_inspection(machine_number=1)
    assert published[0] == (1, InspectionResult.GOOD)
    assert [index for index, _ in published] == [1, 2, 3, 4]


# --------------------------------------------------------- never re-run
def test_good_is_never_rerun() -> None:
    vision = ScriptedVision(reruns={1: 3, 2: 3, 3: 3, 4: 3})
    service, cameras, _ = build(vision)
    service.run_inspection(machine_number=1)
    assert cameras.capture_log == [1, 2, 3, 4]


def test_error_is_never_rerun() -> None:
    vision = ScriptedVision(reruns={2: 3}, script={2: [ERROR, GOOD]})
    service, cameras, _ = build(vision)
    cycle = service.run_inspection(machine_number=1)
    assert cycle.cameras[2].result is InspectionResult.ERROR
    assert grabs_of(cameras, 2) == 1


# ------------------------------------------------- a re-run that fails
def test_a_failed_rerun_detection_keeps_the_measured_ng() -> None:
    vision = ScriptedVision(reruns={2: 3}, script={2: [NG, ERROR, GOOD]})
    service, cameras, _ = build(vision)
    cycle = service.run_inspection(machine_number=1)
    assert cycle.cameras[2].result is InspectionResult.NG
    assert grabs_of(cameras, 2) == 2  # stopped at the failure


def test_a_failed_recapture_keeps_the_measured_ng() -> None:
    vision = ScriptedVision(reruns={2: 3}, script={2: [NG, GOOD]})
    cameras = FlakyCameraManager(fail_on={2: {2}})
    service, cameras, _ = build(vision, cameras=cameras)
    cycle = service.run_inspection(machine_number=1)
    assert cycle.cameras[2].result is InspectionResult.NG
    assert cycle.cameras[2].error == ""  # the NG, not the grab failure
    assert grabs_of(cameras, 2) == 2


# ------------------------------------------------------- other paths
def test_parallel_mode_reruns_the_ng_camera_only() -> None:
    vision = ScriptedVision(reruns={4: 2}, script={4: [NG, GOOD]})
    service, cameras, _ = build(vision, capture_mode="parallel")
    cycle = service.run_inspection(machine_number=1)
    assert cameras.capture_all_calls == 1
    assert cycle.cameras[4].result is InspectionResult.GOOD
    assert grabs_of(cameras, 4) == 2
    assert [grabs_of(cameras, i) for i in (1, 2, 3)] == [1, 1, 1]


def test_the_per_camera_trigger_reruns_too() -> None:
    vision = ScriptedVision(reruns={3: 2}, script={3: [NG, GOOD]})
    service, cameras, _ = build(vision)
    cycle = service.run_camera_inspection(3, machine_number=1)
    assert cycle.overall_result is InspectionResult.GOOD
    assert cameras.capture_log == [3, 3]


class SlowCameraManager(FakeCameraManager):
    """Every grab takes 100 ms of a fake clock, so time spent is countable."""

    def __init__(self, clock: list[float]) -> None:
        super().__init__()
        self._clock = clock

    def capture(self, index: int, *, apply_roi: bool = True) -> np.ndarray:
        self._clock[0] += 0.1
        return super().capture(index, apply_roi=apply_roi)


@pytest.mark.parametrize("mode", ["sequential", "parallel"])
def test_cycle_time_includes_the_reruns(mode, monkeypatch) -> None:
    """A re-run is part of reaching the camera's verdict, so it is part of
    the time that camera took (and of what the PLC waited for)."""
    clock = [0.0]
    monkeypatch.setattr("services.inspection_service.time.perf_counter", lambda: clock[0])

    def camera_1_time(vision: ScriptedVision) -> float:
        service, _, _ = build(vision, cameras=SlowCameraManager(clock), capture_mode=mode)
        return service.run_inspection(machine_number=1).cameras[1].cycle_time_ms

    plain = camera_1_time(ScriptedVision())
    rerun = camera_1_time(ScriptedVision(reruns={1: 2}, script={1: [NG, NG, NG]}))
    assert rerun - plain == pytest.approx(200)  # two 100 ms re-captures

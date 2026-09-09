"""Sequential capture: one camera at a time, with the configured delay.

Pins what the operator sees after pressing "Simulate Trigger" — cameras fire
in index order, never overlapping, each picture and verdict reaching the
dashboard as it happens, and the pause between cameras coming from
``app_config.inspection.camera_delay_ms``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.utilities.enums import InspectionResult
from core.vision.detection_result import DetectionResult, Hole
from models.app_state import AppState
from services.inspection_service import InspectionService

CAMERA_INDEXES = (1, 2, 3, 4)


class FakeCamera:
    def __init__(self, index: int, enabled: bool = True) -> None:
        self.name = f"Camera {index}"
        self.settings = SimpleNamespace(enabled=enabled)


class FakeCameraManager:
    """Records the order of the grabs and how they overlap in time."""

    def __init__(self, enabled=CAMERA_INDEXES, failing=()) -> None:
        self.cameras = {index: FakeCamera(index, index in enabled) for index in CAMERA_INDEXES}
        self._failing = set(failing)
        self.capture_log: list[int] = []
        self.capture_all_calls = 0

    def get(self, index: int) -> FakeCamera:
        return self.cameras[index]

    def health(self, index: int):
        return SimpleNamespace(last_error="camera offline")

    def capture(self, index: int) -> np.ndarray:
        self.capture_log.append(index)
        if index in self._failing:
            from core.utilities.exceptions import CameraCaptureError

            raise CameraCaptureError(f"Camera {index}: boom")
        return np.zeros((40, 40, 3), dtype=np.uint8)

    def capture_all(self, indexes=None):
        self.capture_all_calls += 1
        indexes = indexes or [i for i, c in self.cameras.items() if c.settings.enabled]
        return {index: self.capture(index) for index in indexes}


class FakeVision:
    @staticmethod
    def expected_hole_count(camera_index: int) -> int:
        return 1

    @staticmethod
    def position_tolerance_mm(camera_index: int) -> float:
        return 0.0

    @staticmethod
    def detect(frame, camera_index: int) -> DetectionResult:
        return DetectionResult(holes=[Hole(10.0, 12.0, 30.0, 0.9, 0.8)])


class FakeConfig:
    def __init__(self, **inspection) -> None:
        self._document = {
            "application": {"serial_prefix": "WM-"},
            "storage": {"save_images": False},
            "inspection": inspection,
        }

    def load(self, name: str) -> dict:
        return self._document


class FakeShifts:
    """Stands in for ShiftService: the pipeline only ever asks it to name the
    shift a cycle started in."""

    def __init__(self, name: str = "Morning") -> None:
        self.name = name
        self.asked: list = []

    def current_name(self, moment=None) -> str:
        self.asked.append(moment)
        return self.name


@pytest.fixture()
def service_parts():
    cameras = FakeCameraManager()
    app_state = AppState()
    calibration = SimpleNamespace(
        has=lambda index: True,
        evaluate=lambda index, x, y, w=None, h=None: (1.0, 2.0, 0.0),
        screw_offset=lambda index: (0.0, 0.0),
    )
    plc = SimpleNamespace(
        write_inspection_output=lambda positions, camera_results, result, skipped=None: None,
        read_serial_number=lambda: None,  # register not configured
        read_gantry_status=lambda index: True,  # register not configured
    )
    database = SimpleNamespace(save_inspection=lambda cycle: 1)
    return cameras, app_state, calibration, plc, database


def build_service(parts, config: FakeConfig) -> InspectionService:
    cameras, app_state, calibration, plc, database = parts
    return InspectionService(
        cameras, FakeVision(), calibration, plc, database, app_state, config, FakeShifts()
    )


def test_cameras_fire_one_after_another_in_index_order(service_parts) -> None:
    cameras, app_state = service_parts[0], service_parts[1]
    captured: list[int] = []
    inspected: list[int] = []
    app_state.camera_captured.connect(lambda index, frame: captured.append(index))
    app_state.camera_inspected.connect(lambda index, data: inspected.append(index))

    service = build_service(service_parts, FakeConfig(capture_mode="sequential", camera_delay_ms=0))
    cycle = service.run_inspection(machine_number=1)

    assert cameras.capture_log == [1, 2, 3, 4]
    assert captured == [1, 2, 3, 4]  # each picture reaches the dashboard as it is taken
    assert inspected == [1, 2, 3, 4]  # ... and so does each verdict
    assert cameras.capture_all_calls == 0  # never a parallel grab
    assert cycle.overall_result is InspectionResult.GOOD


def test_delay_applies_between_cameras_but_not_before_the_first(service_parts, monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("services.inspection_service.time.sleep", slept.append)

    service = build_service(
        service_parts, FakeConfig(capture_mode="sequential", camera_delay_ms=750)
    )
    service.run_inspection(machine_number=2)

    assert slept == [0.75, 0.75, 0.75]  # three gaps between four cameras


def test_zero_delay_does_not_sleep(service_parts, monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("services.inspection_service.time.sleep", slept.append)

    service = build_service(service_parts, FakeConfig(camera_delay_ms=0))
    service.run_inspection(machine_number=3)

    assert slept == []


def test_disabled_camera_is_skipped(service_parts) -> None:
    cameras = service_parts[0]
    cameras.cameras[3].settings.enabled = False

    service = build_service(service_parts, FakeConfig(camera_delay_ms=0))
    cycle = service.run_inspection(machine_number=4)

    assert cameras.capture_log == [1, 2, 4]
    assert set(cycle.cameras) == {1, 2, 4}


def test_a_failing_camera_does_not_stop_the_sequence(service_parts) -> None:
    cameras, app_state = service_parts[0], service_parts[1]
    cameras._failing = {2}
    captured: list[int] = []
    app_state.camera_captured.connect(lambda index, frame: captured.append(index))

    service = build_service(service_parts, FakeConfig(camera_delay_ms=0))
    cycle = service.run_inspection(machine_number=5)

    assert cameras.capture_log == [1, 2, 3, 4]  # the others still ran
    assert captured == [1, 3, 4]  # no picture published for the dead camera
    assert cycle.cameras[2].result is InspectionResult.ERROR
    assert cycle.overall_result is InspectionResult.ERROR


def test_parallel_mode_still_grabs_everything_at_once(service_parts) -> None:
    cameras = service_parts[0]
    service = build_service(service_parts, FakeConfig(capture_mode="parallel"))
    cycle = service.run_inspection(machine_number=6)

    assert cameras.capture_all_calls == 1
    assert set(cycle.cameras) == set(CAMERA_INDEXES)


def test_each_camera_gets_its_own_cycle_time_sequential(service_parts) -> None:
    service = build_service(service_parts, FakeConfig(capture_mode="sequential", camera_delay_ms=0))
    cycle = service.run_inspection(machine_number=7)

    for index in CAMERA_INDEXES:
        assert cycle.cameras[index].cycle_time_ms > 0.0


def test_each_camera_gets_its_own_cycle_time_parallel(service_parts) -> None:
    service = build_service(service_parts, FakeConfig(capture_mode="parallel"))
    cycle = service.run_inspection(machine_number=8)

    for index in CAMERA_INDEXES:
        assert cycle.cameras[index].cycle_time_ms > 0.0


def test_skipped_camera_has_no_cycle_time(service_parts) -> None:
    cameras, _app_state, _calibration, plc, _database = service_parts
    plc.read_gantry_status = lambda index: index != 2  # camera 2's gantry is parked

    service = build_service(service_parts, FakeConfig(capture_mode="sequential", camera_delay_ms=0))
    cycle = service.run_inspection(machine_number=9)

    assert cycle.cameras[2].result is InspectionResult.SKIPPED
    assert cycle.cameras[2].cycle_time_ms == 0.0


# A station can legitimately have more than one real hole in its field of
# view (see core/vision/detection_result.py: DetectionResult.best is plain
# "highest confidence"). These two candidates are used to prove the pipeline
# — not the detector — is responsible for picking the *right* one when a
# calibrated reference point is available to tell them apart.
_NEAR_REF_HOLE = Hole(x_px=50.0, y_px=50.0, diameter_px=30.0, circularity=0.9, confidence=0.70)
_FAR_FROM_REF_HOLE = Hole(x_px=200.0, y_px=200.0, diameter_px=30.0, circularity=0.9, confidence=0.95)


class MultiHoleVision(FakeVision):
    """Reports two real holes per frame, sorted best-confidence-first."""

    @staticmethod
    def detect(frame, camera_index: int) -> DetectionResult:
        return DetectionResult(holes=[_FAR_FROM_REF_HOLE, _NEAR_REF_HOLE])


def test_calibrated_camera_picks_the_hole_nearest_the_reference_point(service_parts) -> None:
    """The higher-confidence candidate is the wrong hole for this station —
    position-aware selection must still report the one near the reference
    point, not just whichever scored higher."""
    cameras, app_state, _, plc, database = service_parts

    def evaluate(index: int, x: float, y: float, w=None, h=None):
        deviation = 0.1 if (x, y) == (_NEAR_REF_HOLE.x_px, _NEAR_REF_HOLE.y_px) else 50.0
        return x / 10.0, y / 10.0, deviation

    calibration = SimpleNamespace(
        has=lambda index: True, evaluate=evaluate, screw_offset=lambda index: (0.0, 0.0)
    )
    service = InspectionService(
        cameras, MultiHoleVision(), calibration, plc, database, app_state,
        FakeConfig(camera_delay_ms=0), FakeShifts(),
    )

    cycle = service.run_inspection(machine_number=7)

    for data in cycle.cameras.values():
        assert (data.x_px, data.y_px) == (_NEAR_REF_HOLE.x_px, _NEAR_REF_HOLE.y_px)
        assert data.confidence == _NEAR_REF_HOLE.confidence


def test_uncalibrated_camera_still_picks_the_highest_confidence_hole(service_parts) -> None:
    """No reference point to judge distance against — behaviour is unchanged
    from before position-aware selection existed."""
    cameras, app_state, _, plc, database = service_parts
    calibration = SimpleNamespace(
        has=lambda index: False,
        evaluate=lambda index, x, y, w=None, h=None: (x, y, None),
        screw_offset=lambda index: (0.0, 0.0),
    )
    service = InspectionService(
        cameras, MultiHoleVision(), calibration, plc, database, app_state,
        FakeConfig(camera_delay_ms=0), FakeShifts(),
    )

    cycle = service.run_inspection(machine_number=8)

    for data in cycle.cameras.values():
        assert (data.x_px, data.y_px) == (_FAR_FROM_REF_HOLE.x_px, _FAR_FROM_REF_HOLE.y_px)
        assert data.confidence == _FAR_FROM_REF_HOLE.confidence


# ---------------------------------------------------------------------- shift
def test_cycle_is_stamped_with_the_shift_it_started_in(service_parts) -> None:
    """The shift comes from ShiftService, resolved against the cycle's own
    start time — not read as a static string out of app_config."""
    cameras, app_state, calibration, plc, database = service_parts
    shifts = FakeShifts("Night")
    service = InspectionService(
        cameras, FakeVision(), calibration, plc, database, app_state,
        FakeConfig(camera_delay_ms=0), shifts,
    )

    cycle = service.run_inspection(machine_number=11)

    assert cycle.shift == "Night"
    # asked exactly once, and with the moment the cycle started rather than
    # with no argument (which would re-read the clock after the cameras ran)
    assert len(shifts.asked) == 1
    assert shifts.asked[0] == cycle.started_at


def test_single_camera_cycle_is_stamped_the_same_way(service_parts) -> None:
    cameras, app_state, calibration, plc, database = service_parts
    plc = SimpleNamespace(
        write_camera_inspection_output=lambda index, position, result: None,
        read_serial_number=lambda: None,
        read_gantry_status=lambda index: True,
    )
    shifts = FakeShifts("Evening")
    service = InspectionService(
        cameras, FakeVision(), calibration, plc, database, app_state,
        FakeConfig(camera_delay_ms=0), shifts,
    )

    cycle = service.run_camera_inspection(camera_index=2, machine_number=12)

    assert cycle.shift == "Evening"
    assert shifts.asked == [cycle.started_at]

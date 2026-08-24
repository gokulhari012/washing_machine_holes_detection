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
    expected_hole_count = 1
    position_tolerance_mm = 0.0

    @staticmethod
    def detect(frame) -> DetectionResult:
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


@pytest.fixture()
def service_parts():
    cameras = FakeCameraManager()
    app_state = AppState()
    calibration = SimpleNamespace(evaluate=lambda index, x, y: (1.0, 2.0, 0.0))
    plc = SimpleNamespace(write_inspection_output=lambda positions, result: None)
    database = SimpleNamespace(save_inspection=lambda cycle: 1)
    return cameras, app_state, calibration, plc, database


def build_service(parts, config: FakeConfig) -> InspectionService:
    cameras, app_state, calibration, plc, database = parts
    return InspectionService(
        cameras, FakeVision(), calibration, plc, database, app_state, config
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

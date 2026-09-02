"""Per-camera trigger: inspect one camera without disturbing the other three.

Covers the whole path the feature adds — register map parsing, the PLC write
that touches only one camera's registers, and the inspection cycle that
captures only that camera and stays out of the product counters.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.plc import PlcManager, RegisterMap, SimulatedPlc
from core.utilities.enums import InspectionResult, PlcResultCode
from core.vision.detection_result import DetectionResult, Hole
from models.app_state import AppState
from services.inspection_service import InspectionService

from tests.test_register_map import make_config

CAMERA_INDEXES = (1, 2, 3, 4)


# --------------------------------------------------------------- register map
def test_camera_handshake_defaults_to_empty() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.camera_triggers == {}
    assert rmap.camera_vision_complete == {}


def test_camera_handshake_parsed_when_present() -> None:
    config = make_config()
    config["registers"]["camera_triggers"] = {"1": 132, "2": 133}
    config["registers"]["camera_vision_complete"] = {"1": 136, "2": 137}
    config["registers"]["camera_status"] = {"1": 140, "2": 141}
    rmap = RegisterMap.from_config(config)

    assert rmap.camera_triggers == {1: 132, 2: 133}
    assert rmap.camera_vision_complete == {1: 136, 2: 137}
    assert rmap.camera_status == {1: 140, 2: 141}


def test_camera_status_defaults_to_empty() -> None:
    assert RegisterMap.from_config(make_config()).camera_status == {}


# ----------------------------------------------------------------- PLC writes
@pytest.fixture()
def camera_stack() -> tuple[SimulatedPlc, PlcManager, RegisterMap]:
    config = make_config()
    config["registers"]["camera_results"] = {"1": 128, "2": 129}
    config["registers"]["camera_triggers"] = {"1": 132, "2": 133}
    config["registers"]["camera_vision_complete"] = {"1": 136, "2": 137}
    config["registers"]["camera_status"] = {"1": 140, "2": 141}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    return client, manager, rmap


def test_camera_write_touches_only_that_camera(camera_stack) -> None:
    client, manager, rmap = camera_stack
    manager.write_camera_inspection_output(1, (12.3, -4.5), PlcResultCode.GOOD)

    x_addr, y_addr = rmap.camera_positions[1]
    assert rmap.decode_position(client.read_registers(x_addr, 1)[0]) == pytest.approx(12.3)
    assert rmap.decode_position(client.read_registers(y_addr, 1)[0]) == pytest.approx(-4.5)
    assert client.read_registers(128, 1)[0] == int(PlcResultCode.GOOD)
    assert client.read_registers(136, 1)[0] == 1  # this camera's completion flag

    # camera 2 and the overall handshake are untouched
    x2_addr, y2_addr = rmap.camera_positions[2]
    assert client.read_registers(x2_addr, 1)[0] == 0
    assert client.read_registers(y2_addr, 1)[0] == 0
    assert client.read_registers(129, 1)[0] == 0
    assert client.read_registers(137, 1)[0] == 0
    assert client.read_registers(rmap.result, 1)[0] == 0
    assert client.read_registers(rmap.vision_complete, 1)[0] == 0


def test_camera_write_uses_no_hole_sentinel(camera_stack) -> None:
    client, manager, rmap = camera_stack
    manager.write_camera_inspection_output(1, None, PlcResultCode.NG)

    x_addr, y_addr = rmap.camera_positions[1]
    assert client.read_registers(x_addr, 1)[0] == RegisterMap.NO_HOLE_RAW
    assert client.read_registers(y_addr, 1)[0] == RegisterMap.NO_HOLE_RAW
    assert client.read_registers(128, 1)[0] == int(PlcResultCode.NG)


def test_read_camera_trigger_is_inert_when_unconfigured(camera_stack) -> None:
    _client, manager, _rmap = camera_stack
    assert manager.read_camera_trigger(1) == 0
    assert manager.read_camera_trigger(4) is None  # no register for camera 4
    assert manager.camera_trigger_configured(1) is True
    assert manager.camera_trigger_configured(4) is False


# --------------------------------------------------------- camera status
def test_write_camera_status(camera_stack) -> None:
    client, manager, _rmap = camera_stack

    assert manager.write_camera_status(1, True) is True
    assert client.read_registers(140, 1)[0] == RegisterMap.CAMERA_AVAILABLE

    assert manager.write_camera_status(1, False) is True
    assert client.read_registers(140, 1)[0] == RegisterMap.CAMERA_UNAVAILABLE

    # camera 4 has no status register: inert, no I/O, never an error
    assert manager.write_camera_status(4, True) is False
    assert manager.camera_status_configured(1) is True
    assert manager.camera_status_configured(4) is False


def test_poll_worker_publishes_status_only_on_change(camera_stack) -> None:
    from workers.plc_poll_worker import PlcPollWorker

    client, manager, _rmap = camera_stack
    availability = {1: True, 2: True}
    writes: list[tuple[int, bool]] = []
    original = manager.write_camera_status

    def spy(index: int, available: bool) -> bool:
        writes.append((index, available))
        return original(index, available)

    manager.write_camera_status = spy
    worker = PlcPollWorker(manager, camera_status_provider=lambda: dict(availability))

    worker._publish_camera_status()
    assert writes == [(1, True), (2, True)]  # first pass publishes both

    worker._publish_camera_status()
    assert len(writes) == 2  # unchanged -> no further traffic

    availability[2] = False
    worker._publish_camera_status()
    assert writes[-1] == (2, False)
    assert client.read_registers(141, 1)[0] == RegisterMap.CAMERA_UNAVAILABLE

    # a link loss clears the cache, so everything is re-sent to a PLC that
    # may have been power-cycled while we were away
    worker._last_camera_status.clear()
    worker._publish_camera_status()
    assert writes[-2:] == [(1, True), (2, False)]


def test_poll_worker_without_provider_is_inert(camera_stack) -> None:
    from workers.plc_poll_worker import PlcPollWorker

    _client, manager, _rmap = camera_stack
    worker = PlcPollWorker(manager)
    worker._publish_camera_status()  # must not raise


# ------------------------------------------------------------- the cycle
class FakeCamera:
    def __init__(self, index: int) -> None:
        self.name = f"Camera {index}"
        self.settings = SimpleNamespace(enabled=True)


class FakeCameraManager:
    def __init__(self) -> None:
        self.cameras = {index: FakeCamera(index) for index in CAMERA_INDEXES}
        self.capture_log: list[int] = []
        self.capture_all_calls = 0

    def get(self, index: int) -> FakeCamera:
        return self.cameras[index]

    def health(self, index: int):
        return SimpleNamespace(last_error="camera offline")

    def capture(self, index: int) -> np.ndarray:
        self.capture_log.append(index)
        return np.zeros((40, 40, 3), dtype=np.uint8)

    def capture_all(self, indexes=None):
        self.capture_all_calls += 1
        return {}


class FakeVision:
    expected_hole_count = 1
    position_tolerance_mm = 0.0

    @staticmethod
    def detect(frame) -> DetectionResult:
        return DetectionResult(holes=[Hole(10.0, 12.0, 30.0, 0.9, 0.8)])


class FakeConfig:
    def load(self, name: str) -> dict:
        return {
            "application": {"serial_prefix": "WM-"},
            "storage": {"save_images": False},
            "inspection": {"capture_mode": "sequential", "camera_delay_ms": 0},
        }


class RecordingPlc:
    def __init__(self) -> None:
        self.camera_writes: list[tuple] = []
        self.full_writes = 0

    def write_camera_inspection_output(self, camera_index, position, result) -> None:
        self.camera_writes.append((camera_index, position, result))

    def write_inspection_output(self, positions, camera_results, result) -> None:
        self.full_writes += 1


@pytest.fixture()
def service() -> tuple[InspectionService, FakeCameraManager, AppState, RecordingPlc]:
    cameras = FakeCameraManager()
    app_state = AppState()
    plc = RecordingPlc()
    calibration = SimpleNamespace(
        has=lambda index: True,
        evaluate=lambda index, x, y, w=None, h=None: (1.0, 2.0, 0.0),
    )
    database = SimpleNamespace(save_inspection=lambda cycle: 1)
    svc = InspectionService(
        cameras, FakeVision(), calibration, plc, database, app_state, FakeConfig()
    )
    return svc, cameras, app_state, plc


def test_only_the_named_camera_is_captured(service) -> None:
    svc, cameras, app_state, plc = service
    captured: list[int] = []
    app_state.camera_captured.connect(lambda index, frame: captured.append(index))

    cycle = svc.run_camera_inspection(camera_index=2, machine_number=7)

    assert cameras.capture_log == [2]  # the other three are never grabbed
    assert cameras.capture_all_calls == 0
    assert captured == [2]
    assert list(cycle.cameras) == [2]
    assert cycle.overall_result is InspectionResult.GOOD


def test_only_that_cameras_registers_are_written(service) -> None:
    svc, _cameras, _app_state, plc = service
    svc.run_camera_inspection(camera_index=3, machine_number=7)

    assert plc.full_writes == 0  # never the whole-cycle write
    assert len(plc.camera_writes) == 1
    camera_index, position, result = plc.camera_writes[0]
    assert camera_index == 3
    assert position == (1.0, 2.0)
    assert result is PlcResultCode.GOOD


def test_partial_cycle_stays_out_of_the_product_counters(service) -> None:
    svc, _cameras, app_state, _plc = service
    app_state.set_counters(10, 8, 2)

    cycle = svc.run_camera_inspection(camera_index=1, machine_number=7)

    assert cycle.partial is True
    assert app_state.counters == (10, 8, 2)  # a single camera is not a product


def test_full_cycle_still_counts(service) -> None:
    svc, _cameras, app_state, plc = service
    app_state.set_counters(10, 8, 2)

    cycle = svc.run_inspection(machine_number=7)

    assert cycle.partial is False
    assert app_state.counters == (11, 9, 2)
    assert plc.full_writes == 1
    assert plc.camera_writes == []


def test_camera_fault_degrades_to_error_and_still_answers_the_plc(service) -> None:
    svc, cameras, _app_state, plc = service

    def boom(index: int):
        from core.utilities.exceptions import CameraCaptureError

        raise CameraCaptureError("Camera 1: boom")

    cameras.capture = boom
    cycle = svc.run_camera_inspection(camera_index=1, machine_number=7)

    assert cycle.overall_result is InspectionResult.ERROR
    # the PLC is still told, with the no-hole sentinel, so it never dead-waits
    assert plc.camera_writes == [(1, None, PlcResultCode.ERROR)]

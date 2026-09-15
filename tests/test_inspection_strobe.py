"""LED strobe brackets around the production trigger paths.

``CameraService.light_on``/``light_off`` already strobe the Camera page's
manual Test Camera / Continuous Capture actions (see
``test_camera_service.py``). This file pins the on-just-before,
off-just-after bracket around the PLC trigger paths themselves:
``run_inspection`` (both capture modes) lights every strobe-enabled,
gantry-active camera together with one grouped multi-channel command
(``InspectionService._strobe_group_on``/``_strobe_group_off``) around the
whole cycle's captures, while ``run_camera_inspection`` — only ever one
camera — still uses the single-channel ``_strobe_on``/``_strobe_off``. A
shared event trace lets capture and LED commands be checked in the order
they actually happened, not just that both occurred somewhere.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.utilities.enums import InspectionResult
from core.utilities.exceptions import LedWriteError
from core.vision.detection_result import DetectionResult, Hole
from models.app_state import AppState
from services.inspection_service import InspectionService

CAMERA_INDEXES = (1, 2, 3, 4)


class FakeCamera:
    def __init__(
        self,
        index: int,
        *,
        enabled: bool = True,
        led_strobe: bool = False,
        led_channel: int = 0,
        brightness: int = 0,
    ) -> None:
        self.name = f"Camera {index}"
        self.settings = SimpleNamespace(
            enabled=enabled,
            led_strobe=led_strobe,
            led_channel=led_channel,
            brightness=brightness,
        )


class FakeCameraManager:
    """Records every capture into *trace*, shared with the fake LED so
    ordering between the two can be asserted."""

    def __init__(self, cameras: dict[int, FakeCamera], trace: list) -> None:
        self.cameras = cameras
        self.trace = trace
        self.capture_log: list[int] = []
        self.capture_all_calls = 0

    def get(self, index: int) -> FakeCamera:
        return self.cameras[index]

    def health(self, index: int):
        return SimpleNamespace(last_error="camera offline")

    def capture(self, index: int) -> np.ndarray:
        self.capture_log.append(index)
        self.trace.append(("capture", index))
        return np.zeros((40, 40, 3), dtype=np.uint8)

    def capture_all(self, indexes=None):
        self.capture_all_calls += 1
        indexes = list(indexes) if indexes else list(self.cameras)
        return {index: self.capture(index) for index in indexes}


class FakeLed:
    """Records every command into the same shared trace as the camera."""

    def __init__(self, trace: list, fail_channels: set[int] = frozenset()) -> None:
        self.trace = trace
        self._fail = set(fail_channels)
        self.settings = SimpleNamespace(clamp=lambda value: value)

    def send_channel(self, channel: int, brightness: int) -> str:
        self.trace.append(("led", channel, brightness))
        if channel in self._fail:
            raise LedWriteError("comm failure")
        return "!"

    def send_multichannel(self, states) -> str:
        self.trace.append(("led_group", tuple(states)))
        for channel in range(1, 5):
            if channel in self._fail:
                raise LedWriteError("comm failure")
        return "!"


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
    def current_name(self, moment=None) -> str:
        return "Morning"


class FakePlc:
    def __init__(self) -> None:
        #: camera index -> gantry active; anything absent is active, same as
        #: an unconfigured gantry_status register
        self.gantries: dict[int, bool] = {}

    def read_serial_number(self):
        return None

    def read_gantry_status(self, index: int) -> bool:
        return self.gantries.get(index, True)

    def write_inspection_output(self, positions, camera_results, result, skipped=None) -> None:
        pass

    def write_camera_inspection_output(self, camera_index, position, result) -> None:
        pass

    def write_camera_skipped_output(self, camera_index) -> None:
        pass


def _build(
    cameras: dict[int, FakeCamera],
    trace: list,
    *,
    fail_channels: set[int] = frozenset(),
    **inspection_cfg,
):
    camera_manager = FakeCameraManager(cameras, trace)
    led = FakeLed(trace, fail_channels=fail_channels)
    calibration = SimpleNamespace(
        has=lambda index: True,
        evaluate=lambda index, x, y, w=None, h=None: (1.0, 2.0, 0.0),
        screw_offset=lambda index: (0.0, 0.0),
    )
    database = SimpleNamespace(save_inspection=lambda cycle: 1)
    plc = FakePlc()
    service = InspectionService(
        camera_manager, FakeVision(), calibration, plc, database, AppState(),
        FakeConfig(**inspection_cfg), FakeShifts(), led,
    )
    return service, camera_manager, led, plc


def test_sequential_cycle_strobes_every_camera_in_one_grouped_command() -> None:
    """All strobe-enabled cameras light together in a single grouped
    multi-channel command before the cycle's first capture, and off together
    in one more after its last — not bracketed per camera anymore. A
    non-strobe but wired channel (camera 4) is re-sent at its own steady
    brightness in every frame rather than left out of it; a channel with
    neither (camera 2) always goes out at (0, off)."""
    cameras = {
        1: FakeCamera(1, led_strobe=True, led_channel=1, brightness=200),
        2: FakeCamera(2),  # no channel, no strobe
        3: FakeCamera(3, led_strobe=True, led_channel=3, brightness=90),
        4: FakeCamera(4, led_strobe=False, led_channel=4, brightness=50),  # wired but not strobing
    }
    trace: list = []
    service, _cameras, _led, _plc = _build(
        cameras, trace, capture_mode="sequential", camera_delay_ms=0
    )

    service.run_inspection(machine_number=1)

    assert trace == [
        ("led_group", ((200, True), (0, False), (90, True), (50, True))),
        ("capture", 1),
        ("capture", 2),
        ("capture", 3),
        ("capture", 4),
        ("led_group", ((0, False), (0, False), (0, False), (50, True))),
    ]


def test_parallel_cycle_strobes_every_strobe_camera_in_one_grouped_command() -> None:
    """Parallel mode already grabs every camera in one ``capture_all`` call;
    the light bracket is the same single grouped command as sequential mode
    now, not a loop of individual per-channel writes."""
    cameras = {
        1: FakeCamera(1, led_strobe=True, led_channel=1, brightness=200),
        2: FakeCamera(2, led_strobe=True, led_channel=2, brightness=150),
        3: FakeCamera(3),
        4: FakeCamera(4),
    }
    trace: list = []
    service, _cameras, _led, _plc = _build(
        cameras, trace, capture_mode="parallel", camera_delay_ms=0
    )

    service.run_inspection(machine_number=1)

    group_events = [event for event in trace if event[0] == "led_group"]
    capture_events = [event for event in trace if event[0] == "capture"]

    assert len(group_events) == 2
    assert group_events[0][1] == ((200, True), (150, True), (0, False), (0, False))
    assert group_events[1][1] == ((0, False), (0, False), (0, False), (0, False))
    assert len(capture_events) == 4
    assert trace.index(group_events[0]) < trace.index(capture_events[0])
    assert trace.index(group_events[1]) > trace.index(capture_events[-1])


def test_single_camera_trigger_strobes_only_that_camera() -> None:
    cameras = {
        index: FakeCamera(index, led_strobe=True, led_channel=index, brightness=77)
        for index in CAMERA_INDEXES
    }
    trace: list = []
    service, _cameras, _led, _plc = _build(
        cameras, trace, capture_mode="sequential", camera_delay_ms=0
    )

    service.run_camera_inspection(camera_index=2, machine_number=1)

    assert trace == [("led", 2, 77), ("capture", 2), ("led", 2, 0)]


def test_light_turns_off_even_when_capture_fails() -> None:
    """The bracket is a try/finally, not try/except — a dropped frame must
    never leave a strobe channel lit."""
    from core.utilities.exceptions import CameraCaptureError

    cameras = {1: FakeCamera(1, led_strobe=True, led_channel=1, brightness=100)}
    trace: list = []
    service, camera_manager, _led, _plc = _build(
        cameras, trace, capture_mode="sequential", camera_delay_ms=0
    )

    def boom(index: int):
        raise CameraCaptureError("boom")

    camera_manager.capture = boom
    cycle = service.run_camera_inspection(camera_index=1, machine_number=1)

    assert ("led", 1, 100) in trace
    assert ("led", 1, 0) in trace
    assert trace.index(("led", 1, 0)) > trace.index(("led", 1, 100))
    assert cycle.overall_result is InspectionResult.ERROR


def test_led_failure_never_blocks_the_capture() -> None:
    """A dead LED link degrades like every other best-effort LED write in
    this app — logged, never raised, never stalls the cycle."""
    cameras = {1: FakeCamera(1, led_strobe=True, led_channel=1, brightness=100)}
    trace: list = []
    service, _cameras, _led, _plc = _build(
        cameras, trace, fail_channels={1}, capture_mode="sequential", camera_delay_ms=0
    )

    cycle = service.run_camera_inspection(camera_index=1, machine_number=1)

    assert ("capture", 1) in trace
    assert cycle.overall_result is InspectionResult.GOOD


def test_skipped_camera_is_never_strobed() -> None:
    """A gantry-inactive camera is never captured — it must never be lit
    either, not even as a "steady" channel in the grouped command: a
    strobe-enabled camera's resting state is off, not a brightness to
    preserve."""
    cameras = {
        1: FakeCamera(1, led_strobe=True, led_channel=1, brightness=100),
        2: FakeCamera(2, led_strobe=True, led_channel=2, brightness=100),
    }
    trace: list = []
    service, _cameras, _led, plc = _build(
        cameras, trace, capture_mode="sequential", camera_delay_ms=0
    )
    plc.gantries[1] = False  # camera 1's gantry is parked

    service.run_inspection(machine_number=1)

    group_events = [event for event in trace if event[0] == "led_group"]
    assert group_events, "expected a grouped strobe command for camera 2"
    assert all(states[0] == (0, False) for _, states in group_events)  # channel 1 never lit
    assert ("capture", 1) not in trace


def test_no_led_manager_never_strobes_and_never_raises() -> None:
    """``led_manager`` is optional — a service built without one (its
    default) must behave exactly as it did before strobing existed."""
    cameras = {1: FakeCamera(1, led_strobe=True, led_channel=1, brightness=100)}
    trace: list = []
    camera_manager = FakeCameraManager(cameras, trace)
    calibration = SimpleNamespace(
        has=lambda index: True,
        evaluate=lambda index, x, y, w=None, h=None: (1.0, 2.0, 0.0),
        screw_offset=lambda index: (0.0, 0.0),
    )
    database = SimpleNamespace(save_inspection=lambda cycle: 1)
    service = InspectionService(
        camera_manager, FakeVision(), calibration, FakePlc(), database, AppState(),
        FakeConfig(capture_mode="sequential", camera_delay_ms=0), FakeShifts(),
    )  # led_manager omitted entirely

    service.run_inspection(machine_number=1)

    assert trace == [("capture", 1)]

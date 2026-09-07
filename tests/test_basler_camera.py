"""Basler adapter logic that runs without a camera on the network.

The pypylon calls themselves need hardware, but the parts that decide *what*
gets written to the device — option parsing, node lookup across SFNC naming
variants, clamping and increment snapping — are pure and worth pinning: a
value silently rejected by a GenICam node is the classic Basler bring-up bug.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.camera import create_camera
from core.camera.basler_camera import INCOMPLETE_BUFFER_ERROR, BaslerCamera
from core.camera.camera_base import CameraSettings
from core.utilities.enums import TriggerMode
from core.utilities.exceptions import CameraCaptureError, CameraConfigurationError


class FakeNumberNode:
    """Stands in for a GenICam IInteger / IFloat node."""

    def __init__(self, low, high, inc=1, value=None) -> None:
        self._low, self._high, self._inc = low, high, inc
        self.value = value

    def GetMin(self):
        return self._low

    def GetMax(self):
        return self._high

    def GetInc(self):
        return self._inc

    def GetValue(self):
        return self.value if self.value is not None else self._high

    def SetValue(self, value):
        self.value = value


class FakeEnumNode:
    """Stands in for a GenICam IEnumeration node."""

    def __init__(self, *symbolics: str) -> None:
        self.Symbolics = list(symbolics)
        self.value = None

    def SetValue(self, value):
        self.value = value


@pytest.fixture(autouse=True)
def permissive_genicam(monkeypatch):
    """Treat every fake node as available and writable."""
    monkeypatch.setattr(
        "core.camera.basler_camera.genicam",
        SimpleNamespace(IsAvailable=lambda node: True, IsWritable=lambda node: True),
        raising=False,
    )


def make_camera(**overrides) -> BaslerCamera:
    config = {
        "index": 1,
        "name": "Camera 1",
        "driver": "basler",
        "connection_id": "40123456",
        **overrides,
    }
    return BaslerCamera(CameraSettings.from_config(config))


def test_factory_builds_the_basler_adapter() -> None:
    camera = create_camera(CameraSettings.from_config({"index": 1, "driver": "basler"}))
    assert isinstance(camera, BaslerCamera)


def test_driver_options_come_from_the_basler_block() -> None:
    camera = make_camera(basler={"packet_size": 8192, "trigger_source": "Line3"})
    assert camera._options["packet_size"] == 8192
    assert camera._options["trigger_source"] == "Line3"


def test_missing_or_malformed_block_leaves_defaults() -> None:
    assert make_camera()._options == {}
    assert make_camera(basler="nonsense")._options == {}


def test_write_int_clamps_and_snaps_to_the_increment() -> None:
    camera = make_camera()
    node = FakeNumberNode(low=64, high=8192, inc=4)
    camera._camera = SimpleNamespace(GevSCPSPacketSize=node)

    assert camera._write_int(9000, "GevSCPSPacketSize") is True
    assert node.value == 8192  # clamped to the node maximum
    camera._write_int(1503, "GevSCPSPacketSize")
    assert node.value == 1500  # snapped down onto min + n*inc


def test_write_float_clamps_to_the_node_range() -> None:
    camera = make_camera()
    node = FakeNumberNode(low=20.0, high=10000.0)
    camera._camera = SimpleNamespace(ExposureTime=node)

    assert camera._write_float(1.0, "ExposureTime") is True
    assert node.value == 20.0


def test_writes_fall_back_to_the_sfnc_1_node_name() -> None:
    """An ace classic GigE has ExposureTimeAbs, not ExposureTime."""
    camera = make_camera()
    node = FakeNumberNode(low=20.0, high=10000.0)
    camera._camera = SimpleNamespace(ExposureTimeAbs=node)

    assert camera._write_float(5000.0, "ExposureTime", "ExposureTimeAbs") is True
    assert node.value == 5000.0


def test_unknown_node_is_skipped_instead_of_raising() -> None:
    camera = make_camera()
    camera._camera = SimpleNamespace()
    assert camera._write_float(1.0, "Gain", "GainAbs") is False


def test_enum_entry_the_model_lacks_is_skipped() -> None:
    camera = make_camera()
    node = FakeEnumNode("Line1", "Line2", "Software")
    camera._camera = SimpleNamespace(TriggerSource=node)

    assert camera._write_enum("Line3", "TriggerSource") is False
    assert node.value is None
    assert camera._write_enum("Software", "TriggerSource") is True
    assert node.value == "Software"


def test_detect_resolution_reads_sensor_max_dimensions() -> None:
    camera = make_camera()
    camera._camera = SimpleNamespace(
        WidthMax=FakeNumberNode(low=0, high=4096, value=4096),
        HeightMax=FakeNumberNode(low=0, high=3000, value=3000),
    )
    assert camera._detect_resolution() == (4096, 3000)


def test_detect_resolution_raises_when_nodes_absent() -> None:
    camera = make_camera()
    camera._camera = SimpleNamespace()
    with pytest.raises(CameraConfigurationError):
        camera._detect_resolution()


def test_timeout_message_names_the_trigger_line_in_hardware_mode() -> None:
    camera = make_camera(trigger_mode="hardware", basler={"trigger_source": "Line3"})
    assert "Line3" in camera._timeout_hint(5000)
    assert camera._settings.trigger_mode is TriggerMode.HARDWARE


# ------------------------------------------------------- incomplete buffers
class FakeGrabResult:
    """One RetrieveResult outcome: a good frame or an error code."""

    def __init__(self, error_code: int | None = None) -> None:
        self._error_code = error_code
        self.released = False

    def IsValid(self):
        return True

    def GrabSucceeded(self):
        return self._error_code is None

    def GetErrorCode(self):
        return self._error_code

    def GetErrorDescription(self):
        return "The buffer was incompletely grabbed."

    def GetArray(self):
        return np.zeros((4, 4), np.uint8)

    def Release(self):
        self.released = True


def grabbing_camera(results: list[FakeGrabResult]) -> SimpleNamespace:
    """A camera whose RetrieveResult hands back ``results`` in order."""
    pending = list(results)
    return SimpleNamespace(
        IsGrabbing=lambda: True,
        WaitForFrameTriggerReady=lambda _timeout, _handling: True,
        ExecuteSoftwareTrigger=lambda: None,
        RetrieveResult=lambda _timeout, _handling: pending.pop(0) if pending else None,
    )


def make_grabbing(results, **overrides) -> BaslerCamera:
    camera = make_camera(trigger_mode="continuous", **overrides)  # skips _flush_pending
    camera._camera = grabbing_camera(results)
    camera._options.setdefault("color_conversion", "native")
    return camera


def test_incomplete_buffer_is_retried_rather_than_failing_the_grab() -> None:
    """0xE1000014 is a dropped packet on a shared NIC — the next trigger usually works."""
    results = [FakeGrabResult(INCOMPLETE_BUFFER_ERROR), FakeGrabResult()]
    camera = make_grabbing(results)

    assert camera._grab().shape == (4, 4)
    assert all(result.released for result in results)  # buffers always returned


def test_incomplete_buffer_raises_once_the_retries_are_spent() -> None:
    camera = make_grabbing(
        [FakeGrabResult(INCOMPLETE_BUFFER_ERROR) for _ in range(3)],
        basler={"grab_retries": 2},
    )
    with pytest.raises(CameraCaptureError) as excinfo:
        camera._grab()

    assert "0xe1000014" in str(excinfo.value)
    assert "jumbo frames" in str(excinfo.value)  # says what actually cures it


def test_a_non_transient_grab_error_is_not_retried() -> None:
    """Only an incomplete buffer is worth another trigger; anything else raises."""
    results = [FakeGrabResult(0xE1000001), FakeGrabResult()]
    camera = make_grabbing(results)

    with pytest.raises(CameraCaptureError):
        camera._grab()
    assert results[1].released is False  # the second result was never fetched

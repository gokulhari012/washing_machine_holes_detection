"""Basler adapter logic that runs without a camera on the network.

The pypylon calls themselves need hardware, but the parts that decide *what*
gets written to the device — option parsing, node lookup across SFNC naming
variants, clamping and increment snapping — are pure and worth pinning: a
value silently rejected by a GenICam node is the classic Basler bring-up bug.
"""

from types import SimpleNamespace

import pytest

from core.camera import create_camera
from core.camera.basler_camera import BaslerCamera
from core.camera.camera_base import CameraSettings
from core.utilities.enums import TriggerMode
from core.utilities.exceptions import CameraConfigurationError


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


def test_brightness_maps_onto_the_node_range() -> None:
    camera = make_camera()
    node = FakeNumberNode(low=-1.0, high=1.0)
    camera._camera = SimpleNamespace(BslBrightness=node)

    camera._write_scaled(0.5, "BslBrightness")
    assert node.value == pytest.approx(0.5)
    camera._write_scaled(-2.0, "BslBrightness")  # out of range
    assert node.value == pytest.approx(-1.0)


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

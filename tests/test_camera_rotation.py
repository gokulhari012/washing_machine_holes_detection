"""Per-camera frame rotation, for a camera mounted on its side.

Two properties matter and are pinned here:

- the turn is **clockwise**, the direction the Camera page's combo promises;
- it happens **before** the ROI crop, so `roi`, the ROI drawn on the preview,
  the detection parameters and the calibration all live in the frame the rest
  of the app sees — not in the sensor's own orientation.
"""

import numpy as np
import pytest

from core.camera import VALID_ROTATIONS, CameraSettings, rotate_frame
from core.camera.camera_base import CameraBase
from core.utilities.exceptions import ConfigurationError

BASE_CONFIG = {
    "index": 1, "name": "Cam 1", "driver": "simulated", "enabled": True,
    "width": 8, "height": 6, "trigger_mode": "software",
}


class StubCamera(CameraBase):
    """Hands back a fixed frame so capture()'s own pipeline is what's tested."""

    def __init__(self, settings: CameraSettings, frame: np.ndarray) -> None:
        super().__init__(settings)
        self._frame = frame
        self._connected = True

    def _connect_device(self) -> None: ...
    def _disconnect_device(self) -> None: ...
    def _apply_to_device(self, settings) -> None: ...

    def _grab(self) -> np.ndarray:
        return self._frame


def make_camera(frame: np.ndarray, **overrides) -> StubCamera:
    return StubCamera(CameraSettings.from_config({**BASE_CONFIG, **overrides}), frame)


# --------------------------------------------------------------- the turn
def test_rotation_is_clockwise() -> None:
    """Top-left must land top-right at 90° — anti-clockwise would fail this."""
    frame = np.array([[1, 2], [3, 4]], np.uint8)
    assert rotate_frame(frame, 90).tolist() == [[3, 1], [4, 2]]
    assert rotate_frame(frame, 180).tolist() == [[4, 3], [2, 1]]
    assert rotate_frame(frame, 270).tolist() == [[2, 4], [1, 3]]


def test_rotation_zero_returns_the_frame_untouched() -> None:
    frame = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert rotate_frame(frame, 0) is frame


def test_rotated_frame_is_contiguous() -> None:
    """np.rot90 returns a negative-stride view; OpenCV rejects some of those."""
    frame = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert rotate_frame(frame, 90).flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("rotation, shape", [(0, (3, 4)), (90, (4, 3)), (180, (3, 4)), (270, (4, 3))])
def test_quarter_turns_swap_the_frame_dimensions(rotation, shape) -> None:
    camera = make_camera(np.zeros((3, 4), np.uint8), rotation=rotation)
    assert camera.capture().shape == shape


# ------------------------------------------------------------ ordering
def test_roi_is_cropped_out_of_the_rotated_frame() -> None:
    """The ROI is in the operator's frame, not the sensor's.

    A 2x6 sensor frame turned 90° is 6x2; an ROI of x=0,y=0,w=2,h=3 only fits
    (and only means anything) in the rotated one.
    """
    frame = np.arange(12, dtype=np.uint8).reshape(2, 6)
    camera = make_camera(frame, rotation=90, roi={"x": 0, "y": 0, "width": 2, "height": 3})
    captured = camera.capture()

    assert captured.shape == (3, 2)
    assert captured.tolist() == rotate_frame(frame, 90)[0:3, 0:2].tolist()


def test_an_roi_that_only_fits_the_unrotated_frame_is_clamped_not_honoured() -> None:
    """Proof the crop runs second: a sensor-basis ROI no longer fits after the turn."""
    camera = make_camera(
        np.zeros((2, 6), np.uint8),
        rotation=90,
        roi={"x": 0, "y": 0, "width": 6, "height": 2},  # valid before the turn only
    )
    assert camera.capture().shape == (2, 2)  # clamped to the 6x2 rotated frame


# --------------------------------------------------------------- validation
@pytest.mark.parametrize("rotation", VALID_ROTATIONS)
def test_every_quarter_turn_is_accepted(rotation) -> None:
    assert CameraSettings.from_config({**BASE_CONFIG, "rotation": rotation}).rotation == rotation


def test_rotation_defaults_to_none_for_a_config_predating_the_setting() -> None:
    assert CameraSettings.from_config(BASE_CONFIG).rotation == 0


@pytest.mark.parametrize("rotation", [45, 1, -30, "sideways"])
def test_a_non_quarter_turn_is_rejected(rotation) -> None:
    """Anything else would resample the image — refuse it at load time."""
    with pytest.raises(ConfigurationError):
        CameraSettings.from_config({**BASE_CONFIG, "rotation": rotation})


def test_a_full_turn_normalises_to_none() -> None:
    assert CameraSettings.from_config({**BASE_CONFIG, "rotation": 360}).rotation == 0

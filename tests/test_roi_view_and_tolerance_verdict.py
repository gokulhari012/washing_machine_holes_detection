"""Camera page ROI view, overlay legibility, and the Detection page's
position-tolerance verdict.

- ``capture(apply_roi=False)`` hands back the whole rotated frame, so the
  Camera page can draw the ROI *on* it; ``crop_roi`` is the one definition of
  the crop that both the camera and the page's ROI view use.
- ``draw_detection_overlay`` scales its strokes with the frame, so a 12-20 MP
  frame fitted into a panel still shows a visible circle.
- The Detection page's verdict follows ``InspectionService._inspect_one``
  rule for rule, including *not* claiming a tolerance check on an
  uncalibrated camera.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.calibration import CameraCalibration
from core.calibration.calibration_manager import CalibrationManager
from core.camera import crop_roi
from core.vision import DetectionResult, draw_detection_overlay
from core.vision.detection_result import Hole
from tests.test_camera_rotation import make_camera
from ui.detection.detection_page import DetectionPage


# ------------------------------------------------------------ full frame + crop
def test_capture_without_roi_returns_the_whole_rotated_frame() -> None:
    frame = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
    camera = make_camera(frame, rotation=90, roi={"x": 1, "y": 1, "width": 2, "height": 2})
    assert camera.capture().shape == (2, 2)
    assert camera.capture(apply_roi=False).shape == (8, 6)  # rotated, not cropped


def test_crop_roi_matches_what_capture_crops() -> None:
    frame = np.arange(6 * 8, dtype=np.uint8).reshape(6, 8)
    roi = {"x": 2, "y": 1, "width": 3, "height": 4}
    camera = make_camera(frame, roi=roi)
    np.testing.assert_array_equal(
        crop_roi(camera.capture(apply_roi=False), (2, 1, 3, 4)), camera.capture()
    )


def test_crop_roi_without_a_region_is_the_frame_and_clamps_overhang() -> None:
    frame = np.zeros((10, 20), dtype=np.uint8)
    assert crop_roi(frame, (0, 0, 0, 0)) is frame
    assert crop_roi(frame, (15, 5, 50, 50)).shape == (5, 5)


# ------------------------------------------------------------------- overlay
def _green_pixels(image: np.ndarray) -> int:
    return int(np.count_nonzero(np.all(image == (80, 220, 80), axis=2)))


def test_overlay_stroke_grows_with_the_frame() -> None:
    def ring_pixels(size: int) -> float:
        frame = np.zeros((size, size), dtype=np.uint8)
        hole = Hole(x_px=size / 2, y_px=size / 2, diameter_px=size / 4,
                    circularity=1.0, confidence=0.9)
        drawn = draw_detection_overlay(frame, DetectionResult(holes=[hole]))
        return _green_pixels(drawn) / size  # normalise by the ring's own length

    assert ring_pixels(3000) > 3 * ring_pixels(480)


def test_overlay_labels_secondary_candidates_with_their_confidence() -> None:
    frame = np.zeros((400, 400), dtype=np.uint8)
    best = Hole(x_px=100, y_px=200, diameter_px=30, circularity=1.0, confidence=0.9)
    other = Hole(x_px=300, y_px=200, diameter_px=30, circularity=1.0, confidence=0.7)
    ring_only = draw_detection_overlay(frame, DetectionResult(holes=[best]))
    with_other = draw_detection_overlay(frame, DetectionResult(holes=[best, other]))
    # Below the secondary circle is where its confidence text goes.
    region = (slice(225, 280), slice(280, 360))
    assert not ring_only[region].any()
    assert with_other[region].any()


# ------------------------------------------------------------ tolerance verdict
FRAME = np.zeros((100, 100), dtype=np.uint8)
# 10 px/mm, reference at (5, 5) mm = pixel (50, 50). A hole at (80, 50) px is
# (8, 5) mm -> 3.0 mm from the reference.
HOLE = Hole(x_px=80, y_px=50, diameter_px=10, circularity=1.0, confidence=0.9)


def _calibration(calibrated: bool) -> CalibrationManager:
    manager = CalibrationManager(SimpleNamespace(get_all_active=lambda: {}))
    if calibrated:
        manager.apply_live(CameraCalibration(
            camera_index=1, pixels_per_mm_x=10.0, pixels_per_mm_y=10.0,
            ref_point_mm=(5.0, 5.0),
        ))
    return manager


def _verdict(*, tolerance: float, calibrated: bool = True, expected: int = 1,
             holes: list[Hole] | None = None) -> tuple[str, str]:
    shown: dict[str, str] = {}
    page = SimpleNamespace(
        _last_frame=FRAME,
        _last_result=DetectionResult(holes=[HOLE] if holes is None else holes),
        _last_camera_index=1,
        _expected=SimpleNamespace(value=lambda: expected),
        _tolerance=SimpleNamespace(value=lambda: tolerance),
        _calibration=_calibration(calibrated),
        _set_verdict=lambda result, text: shown.update(result=result, text=text),
    )
    DetectionPage._refresh_verdict(page)
    return shown["result"], shown["text"]


def test_deviation_beyond_tolerance_is_ng() -> None:
    result, text = _verdict(tolerance=2.5)
    assert result == "NG"
    assert "3.00 mm" in text and "2.50 mm" in text


def test_deviation_within_tolerance_is_good() -> None:
    result, text = _verdict(tolerance=5.0)
    assert result == "GOOD"
    assert "within" in text


def test_zero_tolerance_is_reported_as_disabled() -> None:
    result, text = _verdict(tolerance=0.0)
    assert result == "GOOD"
    assert "disabled" in text


def test_uncalibrated_camera_says_the_tolerance_was_not_checked() -> None:
    result, text = _verdict(tolerance=0.1, calibrated=False)
    assert result == "GOOD"  # what the pipeline would report too
    assert "NOT checked" in text


@pytest.mark.parametrize("holes, expected", [([], 1), ([HOLE], 2)])
def test_too_few_holes_is_ng_before_any_tolerance(holes, expected) -> None:
    result, _text = _verdict(tolerance=100.0, holes=holes, expected=expected)
    assert result == "NG"

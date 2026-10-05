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
from core.camera import crop_roi, roi_rect
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


# ------------------------------------------------- overlay on the original frame
def test_roi_rect_is_the_region_crop_roi_keeps() -> None:
    frame = np.arange(10 * 20, dtype=np.uint8).reshape(10, 20)
    x0, y0, x1, y1 = roi_rect(frame, (15, 5, 50, 50))
    np.testing.assert_array_equal(frame[y0:y1, x0:x1], crop_roi(frame, (15, 5, 50, 50)))
    assert roi_rect(frame, (0, 0, 0, 0)) == (0, 0, 20, 10)


def test_overlay_with_roi_draws_the_hole_where_it_sits_in_the_original_frame() -> None:
    """A hole detected at (50, 50) inside an ROI starting at (300, 200) is
    drawn at (350, 250) on the whole frame, and the ROI is outlined."""
    frame = np.zeros((600, 800), dtype=np.uint8)
    hole = Hole(x_px=50, y_px=50, diameter_px=30, circularity=1.0, confidence=0.9)
    drawn = draw_detection_overlay(
        frame, DetectionResult(holes=[hole]), roi=(300, 200, 500, 400)
    )
    green = np.all(drawn == (80, 220, 80), axis=2)
    assert green[250, 350]  # the hole's crosshair, shifted by the ROI origin
    assert not green[:150, :150].any()  # nothing at the ROI-relative (50, 50)
    assert np.all(drawn[300, 400] == (255, 0, 255))  # ROI centre mark
    assert np.all(drawn[200, 420] == (255, 170, 0))  # ROI outline, top edge


def test_pipeline_detects_on_the_roi_but_shows_the_original_frame() -> None:
    """Detection — and so every x_px/x_mm and verdict — still sees only the
    ROI crop; the dashboard frame is the whole 40x40 picture."""
    from tests.test_inspection_strobe import FakeCamera, FakeVision, _build

    seen: list[tuple[int, ...]] = []

    class RecordingVision(FakeVision):
        @staticmethod
        def detect(frame, camera_index: int) -> DetectionResult:
            seen.append(frame.shape)
            return FakeVision.detect(frame, camera_index)

    camera = FakeCamera(1)
    camera.settings.roi = (5, 10, 20, 15)
    service, _cameras, _led, _plc = _build(
        {1: camera}, [], capture_mode="sequential", camera_delay_ms=0
    )
    service._vision = RecordingVision()

    data = service.run_camera_inspection(camera_index=1, machine_number=1).cameras[1]

    assert seen == [(15, 20, 3)]
    assert data.x_px == 10.0  # ROI-relative, as before
    assert data.frame.shape == (40, 40, 3)


def test_full_view_off_shows_only_the_roi_crop_like_earlier_versions() -> None:
    """``ui.dashboard_full_frame = false`` (Settings page) brings back the
    ROI-only picture; the measurement is identical either way."""
    from tests.test_inspection_strobe import FakeCamera, _build

    camera = FakeCamera(1)
    camera.settings.roi = (5, 10, 20, 15)
    service, _cameras, _led, _plc = _build(
        {1: camera}, [], capture_mode="sequential", camera_delay_ms=0
    )
    service._config._document["ui"] = {"dashboard_full_frame": False}

    data = service.run_camera_inspection(camera_index=1, machine_number=1).cameras[1]

    assert data.frame.shape == (15, 20, 3)
    assert data.x_px == 10.0


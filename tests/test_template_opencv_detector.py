"""template_opencv: the template locates a region, OpenCV finds the hole in it.

The point of the strategy is the *restriction*: a hole that is perfectly
findable by OpenCV but sits outside every template match must not be
reported. These tests use a frame carrying two identical bores — one inside a
distinctive stamped plate (the template), one on bare metal — so a plain
OpenCV pass finds both and the two-stage one must find only the first.

The plate needs strong features of its own (ribs, a raised rim): cropped with
its bore on a featureless panel, the dark bore dominates the correlation and
the template then matches *any* bore — 0.92 against the lone one here. That
is the operator's lesson too, and why the Detection page says to crop the
template around the feature that holds the hole.
"""

import copy

import cv2
import numpy as np
import pytest

from core.utilities.exceptions import DetectionError
from core.vision import (
    DetectionResult,
    Hole,
    OpenCVHoleDetector,
    SearchRegion,
    TemplateOpenCVDetector,
    VisionEngine,
    draw_detection_overlay,
)
from core.vision.template_opencv_detector import _without_duplicates

METAL = 190
PLATE = 90
RIB = 250
BORE = 20

PLATE_X, PLATE_Y, PLATE_SIDE = 50, 60, 100
HOLE_IN_PLATE = (PLATE_X + 60, PLATE_Y + 40)  # (110, 100)
HOLE_ON_METAL = (320, 200)
RADIUS = 15

OPENCV = {
    "detection_threshold": 60,
    "adaptive_threshold": False,
    "blur_kernel_size": 3,
    "morphology_operation": "none",
    "min_hole_diameter_px": 20,
    "max_hole_diameter_px": 80,
    "min_circularity": 0.5,
    "min_aspect_ratio": 0.5,
    "contour_retrieval_mode": "list",
}


def draw_plate(frame: np.ndarray, x: int, y: int, with_hole: bool = True) -> None:
    """A distinctive stamped plate: a panel with a raised rim and two ribs."""
    frame[y : y + PLATE_SIDE, x : x + PLATE_SIDE] = PLATE
    cv2.line(frame, (x + 10, y + 75), (x + 90, y + 75), RIB, 12)
    cv2.line(frame, (x + 20, y + 10), (x + 20, y + 90), RIB, 12)
    cv2.rectangle(frame, (x + 3, y + 3), (x + 96, y + 96), RIB, 4)
    if with_hole:
        cv2.circle(frame, (x + 60, y + 40), RADIUS, BORE, -1)


def make_frame(plate: bool = True, hole_in_plate: bool = True) -> np.ndarray:
    frame = np.full((300, 420), METAL, np.uint8)
    if plate:
        draw_plate(frame, PLATE_X, PLATE_Y, with_hole=hole_in_plate)
    cv2.circle(frame, HOLE_ON_METAL, RADIUS, BORE, -1)
    return frame


@pytest.fixture
def template_path(tmp_path) -> str:
    reference = np.full((300, 420), METAL, np.uint8)
    draw_plate(reference, PLATE_X, PLATE_Y)
    crop = reference[PLATE_Y : PLATE_Y + PLATE_SIDE, PLATE_X : PLATE_X + PLATE_SIDE]
    path = tmp_path / "plate.png"
    cv2.imwrite(str(path), crop)
    return str(path)


def params(template_path: str, **own) -> dict:
    return {
        **own,
        "template_matching": {
            "template_path": template_path,
            "match_threshold": 0.8,
            "method": "TM_CCOEFF_NORMED",
            "scales": [1.0],
            "max_matches": 5,
        },
        "opencv": dict(OPENCV),
    }


def near(hole: Hole, point: tuple[int, int], tolerance: float = 1.5) -> bool:
    return abs(hole.x_px - point[0]) <= tolerance and abs(hole.y_px - point[1]) <= tolerance


# ------------------------------------------------------------- the point
def test_plain_opencv_finds_both_bores(template_path) -> None:
    """Sanity check of the fixture: the restriction is what removes one."""
    result = OpenCVHoleDetector(OPENCV).detect(make_frame())
    assert len(result.holes) == 2


def test_only_the_bore_inside_the_template_is_reported(template_path) -> None:
    result = TemplateOpenCVDetector(params(template_path)).detect(make_frame())
    assert len(result.holes) == 1
    assert near(result.holes[0], HOLE_IN_PLATE)  # frame pixels, not crop pixels


def test_the_matched_region_is_kept_on_the_result(template_path) -> None:
    result = TemplateOpenCVDetector(params(template_path)).detect(make_frame())
    assert len(result.regions) == 1
    region = result.regions[0]
    assert (region.x, region.y, region.width, region.height) == (
        PLATE_X, PLATE_Y, PLATE_SIDE, PLATE_SIDE,
    )
    assert region.score > 0.95


def test_no_template_match_is_no_hole_not_an_error(template_path) -> None:
    """The bore on bare metal is still there; without the plate it is ignored."""
    result = TemplateOpenCVDetector(params(template_path)).detect(make_frame(plate=False))
    assert result.holes == []
    assert result.regions == []


def test_a_match_with_no_hole_inside_is_empty_but_keeps_the_region(template_path) -> None:
    """The plate is found (0.98 without its bore) but holds no hole: NG, and
    the overlay still shows where the template was."""
    detector = TemplateOpenCVDetector(params(template_path))
    result = detector.detect(make_frame(hole_in_plate=False))
    assert result.holes == []
    assert len(result.regions) == 1


def test_confidence_is_the_opencv_stage_score(template_path) -> None:
    """The template score gates; it does not multiply the hole's confidence."""
    frame = make_frame()
    combined = TemplateOpenCVDetector(params(template_path)).detect(frame)
    crop = frame[PLATE_Y : PLATE_Y + PLATE_SIDE, PLATE_X : PLATE_X + PLATE_SIDE]
    alone = OpenCVHoleDetector(OPENCV).detect(crop)
    assert combined.holes[0].confidence == pytest.approx(alone.holes[0].confidence)


# ------------------------------------------------------ region handling
def test_padding_grows_the_region_and_is_clamped_to_the_frame(template_path) -> None:
    frame = make_frame()
    padded = TemplateOpenCVDetector(params(template_path, region_padding_px=10)).detect(frame)
    region = padded.regions[0]
    assert (region.x, region.y, region.width, region.height) == (
        PLATE_X - 10, PLATE_Y - 10, PLATE_SIDE + 20, PLATE_SIDE + 20,
    )
    huge = TemplateOpenCVDetector(params(template_path, region_padding_px=5000)).detect(frame)
    region = huge.regions[0]
    assert (region.x, region.y, region.width, region.height) == (0, 0, 420, 300)


def test_max_regions_limits_how_many_matches_are_searched(template_path) -> None:
    frame = make_frame()
    draw_plate(frame, 260, 40)  # a second plate, with its own bore at (320, 80)
    one = TemplateOpenCVDetector(params(template_path, max_regions=1)).detect(frame)
    two = TemplateOpenCVDetector(params(template_path, max_regions=2)).detect(frame)
    assert len(one.regions) == 1 and len(one.holes) == 1
    assert len(two.regions) == 2 and len(two.holes) == 2
    assert not any(near(hole, HOLE_ON_METAL) for hole in two.holes)


def test_a_hole_seen_by_two_overlapping_regions_is_reported_once() -> None:
    holes = [
        Hole(100.0, 100.0, 30.0, 0.9, 0.70),
        Hole(101.0, 100.5, 30.0, 0.9, 0.80),  # same bore, found from the other region
        Hole(300.0, 100.0, 30.0, 0.9, 0.60),
    ]
    kept = _without_duplicates(holes)
    assert [hole.confidence for hole in kept] == [0.80, 0.60]


# -------------------------------------------------------------- failures
def test_missing_template_is_a_detect_time_error_naming_the_stage(template_path) -> None:
    """Built fine (the station must boot), fails when run — CLAUDE.md §9."""
    detector = TemplateOpenCVDetector(params(""))
    with pytest.raises(DetectionError, match="template stage"):
        detector.detect(make_frame())


def test_unreadable_template_raises_at_configure(tmp_path) -> None:
    with pytest.raises(DetectionError):
        TemplateOpenCVDetector(params(str(tmp_path / "missing.png")))


# ------------------------------------------------------------- debug view
def test_debug_stages_are_frame_sized_and_blank_outside_the_region(template_path) -> None:
    frame = make_frame()
    stages = TemplateOpenCVDetector(params(template_path)).debug_stages(frame)
    assert stages["mask"].shape == frame.shape
    inside = stages["mask"][PLATE_Y : PLATE_Y + PLATE_SIDE, PLATE_X : PLATE_X + PLATE_SIDE]
    assert inside.any()  # the bore inside the plate shows up
    outside = stages["mask"].copy()
    outside[PLATE_Y : PLATE_Y + PLATE_SIDE, PLATE_X : PLATE_X + PLATE_SIDE] = 0
    assert not outside.any()  # the bore on bare metal does not


# ---------------------------------------------------- engine integration
def camera_block(template_path: str) -> dict:
    return {
        "active_detector": "template_opencv",
        "common": {"confidence_threshold": 0.3, "expected_hole_count": 1},
        "template_opencv": {"max_regions": 1, "region_padding_px": 0},
        "template_matching": params(template_path)["template_matching"],
        "opencv": dict(OPENCV),
    }


def test_params_are_gathered_from_the_sibling_blocks(template_path) -> None:
    block = camera_block(template_path)
    gathered = TemplateOpenCVDetector.params_from_camera_config(block)
    assert gathered["template_matching"] == block["template_matching"]
    assert gathered["opencv"] == block["opencv"]
    assert gathered["max_regions"] == 1
    gathered["opencv"]["detection_threshold"] = 1  # a copy: the config is untouched
    assert block["opencv"]["detection_threshold"] == 60


def test_engine_runs_it_from_a_camera_block(template_path) -> None:
    engine = VisionEngine({"cameras": {"1": camera_block(template_path)}})
    assert engine.active_detector_name(1) == "template_opencv"
    result = engine.detect(make_frame(), 1)
    assert len(result.holes) == 1 and near(result.holes[0], HOLE_IN_PLATE)
    assert len(result.regions) == 1


def test_engine_keeps_the_camera_block_verbatim(template_path) -> None:
    """The Detection page renders camera_config — it must not see the nested
    stage blocks the detector was built from."""
    block = camera_block(template_path)
    engine = VisionEngine({"cameras": {"1": copy.deepcopy(block)}})
    assert engine.camera_config(1) == block


# ---------------------------------------------------------------- overlay
def test_overlay_outlines_the_region_in_frame_and_roi_coordinates() -> None:
    frame = np.full((300, 420), METAL, np.uint8)
    result = DetectionResult(regions=[SearchRegion(50, 60, 100, 100, 0.9)])
    plain = draw_detection_overlay(frame, result)
    assert tuple(plain[60, 100]) == (0, 140, 255)  # top edge, mid-way along

    shifted = draw_detection_overlay(frame, result, roi=(20, 30, 400, 290))
    assert tuple(shifted[90, 120]) == (0, 140, 255)  # moved by the ROI origin

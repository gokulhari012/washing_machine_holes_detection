"""template_caliper_based_circle_fitting: the template locates a region, calipers measure the hole in it.

Caliper fitting on its own aims its calipers at the largest dark round blob in
the frame, so a *bigger* bore elsewhere takes the calipers with it. The
fixture is built to hit exactly that — a small bore inside a distinctive
stamped plate (the template) and a bigger bore on bare metal — the same plate
``test_template_opencv_detector`` and ``test_template_blob_analysis_detector``
use, for the same reason: a template that is mostly a bore matches any bore.

The seed threshold is fixed at 50 here rather than Otsu: inside the matched
plate there are three grey levels (bore 20, panel 90, ribs 250) and Otsu
splits them at 150 — between panel and ribs — so the calipers are seeded on
the whole panel and fit a 67 px circle with 53 % inliers, which passes the
default 30 % gate. That is a real tuning point, and why "Seed Threshold" is
swept: the sweep scores 50 at confidence 1.00 against Otsu's 0.53.
"""

import copy

import cv2
import numpy as np
import pytest

from core.utilities.exceptions import DetectionError
from core.vision import (
    CaliperCircleDetector,
    Hole,
    TemplateCaliperCircleDetector,
    VisionEngine,
)

METAL = 190
PLATE = 90
RIB = 250
BORE = 20

PLATE_X, PLATE_Y, PLATE_SIDE = 50, 60, 100
HOLE_IN_PLATE = (PLATE_X + 60, PLATE_Y + 40)  # (110, 100)
RADIUS_IN_PLATE = 15
HOLE_ON_METAL = (320, 210)
RADIUS_ON_METAL = 30

CALIPER = {
    "coarse_threshold": 50,
    "blur_kernel_size": 3,
    "num_calipers": 72,
    "caliper_width_px": 3,
    "search_band": 0.5,
    "edge_polarity": "dark_to_light",
    "edge_selection": "first",
    "min_edge_strength": 10,
    "inlier_threshold_px": 2.0,
    "min_inlier_ratio": 0.3,
    "max_rms_px": 1.5,
}


def draw_plate(frame: np.ndarray, x: int, y: int, with_hole: bool = True) -> None:
    """A distinctive stamped plate: a panel with a raised rim and two ribs."""
    frame[y : y + PLATE_SIDE, x : x + PLATE_SIDE] = PLATE
    cv2.line(frame, (x + 10, y + 75), (x + 90, y + 75), RIB, 12)
    cv2.line(frame, (x + 20, y + 10), (x + 20, y + 90), RIB, 12)
    cv2.rectangle(frame, (x + 3, y + 3), (x + 96, y + 96), RIB, 4)
    if with_hole:
        cv2.circle(frame, (x + 60, y + 40), RADIUS_IN_PLATE, BORE, -1)


def make_frame(plate: bool = True, hole_in_plate: bool = True) -> np.ndarray:
    frame = np.full((300, 420), METAL, np.uint8)
    if plate:
        draw_plate(frame, PLATE_X, PLATE_Y, with_hole=hole_in_plate)
    cv2.circle(frame, HOLE_ON_METAL, RADIUS_ON_METAL, BORE, -1)
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
        "caliper_based_circle_fitting": dict(CALIPER),
    }


def near(hole: Hole, point: tuple[int, int], tolerance: float = 1.0) -> bool:
    return abs(hole.x_px - point[0]) <= tolerance and abs(hole.y_px - point[1]) <= tolerance


# ------------------------------------------------------------- the point
def test_plain_caliper_fitting_measures_the_bigger_bore_on_bare_metal() -> None:
    """Sanity check of the fixture: this is the failure the template fixes."""
    assert near(CaliperCircleDetector(CALIPER).detect(make_frame()).best, HOLE_ON_METAL)


def test_only_the_bore_inside_the_template_is_reported(template_path) -> None:
    result = TemplateCaliperCircleDetector(params(template_path)).detect(make_frame())
    assert len(result.holes) == 1
    hole = result.holes[0]
    assert near(hole, HOLE_IN_PLATE)  # frame pixels, not crop pixels
    assert hole.diameter_px == pytest.approx(2 * RADIUS_IN_PLATE, abs=2)
    assert len(result.regions) == 1
    region = result.regions[0]
    assert (region.x, region.y, region.width, region.height) == (
        PLATE_X, PLATE_Y, PLATE_SIDE, PLATE_SIDE,
    )


def test_no_template_match_is_no_hole_not_an_error(template_path) -> None:
    result = TemplateCaliperCircleDetector(params(template_path)).detect(make_frame(plate=False))
    assert result.holes == []
    assert result.regions == []


def test_a_match_with_no_hole_inside_is_empty_but_keeps_the_region(template_path) -> None:
    result = TemplateCaliperCircleDetector(params(template_path)).detect(
        make_frame(hole_in_plate=False)
    )
    assert result.holes == []
    assert len(result.regions) == 1


def test_confidence_is_the_caliper_stage_score(template_path) -> None:
    frame = make_frame()
    combined = TemplateCaliperCircleDetector(params(template_path)).detect(frame)
    crop = frame[PLATE_Y : PLATE_Y + PLATE_SIDE, PLATE_X : PLATE_X + PLATE_SIDE]
    alone = CaliperCircleDetector(CALIPER).detect(crop)
    assert combined.holes[0].confidence == pytest.approx(alone.holes[0].confidence)


def test_one_hole_per_matched_region(template_path) -> None:
    frame = make_frame()
    draw_plate(frame, 280, 40)  # a second plate, its bore at (340, 80)
    one = TemplateCaliperCircleDetector(params(template_path, max_regions=1)).detect(frame)
    two = TemplateCaliperCircleDetector(params(template_path, max_regions=2)).detect(frame)
    assert len(one.regions) == 1 and len(one.holes) == 1
    assert len(two.regions) == 2 and len(two.holes) == 2
    assert not any(near(hole, HOLE_ON_METAL, tolerance=5) for hole in two.holes)


# -------------------------------------------------------------- failures
def test_missing_template_is_a_detect_time_error_naming_the_stage() -> None:
    detector = TemplateCaliperCircleDetector(params(""))
    with pytest.raises(
        DetectionError, match="template_caliper_based_circle_fitting — template stage"
    ):
        detector.detect(make_frame())


# ------------------------------------------------------------- debug view
def test_debug_stages_are_mask_and_edges_inside_the_region_only(template_path) -> None:
    frame = make_frame()
    stages = TemplateCaliperCircleDetector(params(template_path)).debug_stages(frame)
    assert set(stages) == {"mask", "edges"}
    assert stages["mask"].shape == stages["edges"].shape == frame.shape
    assert stages["edges"][PLATE_Y : PLATE_Y + PLATE_SIDE, PLATE_X : PLATE_X + PLATE_SIDE].any()
    outside = stages["edges"].copy()
    outside[PLATE_Y : PLATE_Y + PLATE_SIDE, PLATE_X : PLATE_X + PLATE_SIDE] = 0
    assert not outside.any()  # the bore on bare metal was never looked at


def test_debug_stages_are_blank_not_missing_without_a_match(template_path) -> None:
    stages = TemplateCaliperCircleDetector(params(template_path)).debug_stages(
        make_frame(plate=False)
    )
    assert set(stages) == {"mask", "edges"}
    assert not stages["mask"].any() and not stages["edges"].any()


# ------------------------------------------------- parameters and engine
def camera_block(template_path: str) -> dict:
    return {
        "active_detector": "template_caliper_based_circle_fitting",
        "common": {"confidence_threshold": 0.3, "expected_hole_count": 1},
        "template_caliper_based_circle_fitting": {"max_regions": 1, "region_padding_px": 0},
        "template_matching": params(template_path)["template_matching"],
        "caliper_based_circle_fitting": dict(CALIPER),
        "blob_analysis": {"threshold": 1},  # another strategy's block: not read
    }


def test_params_are_gathered_from_the_sibling_blocks(template_path) -> None:
    block = camera_block(template_path)
    gathered = TemplateCaliperCircleDetector.params_from_camera_config(block)
    assert gathered["template_matching"] == block["template_matching"]
    assert gathered["caliper_based_circle_fitting"] == block["caliper_based_circle_fitting"]
    assert gathered["max_regions"] == 1
    assert "blob_analysis" not in gathered
    gathered["caliper_based_circle_fitting"]["num_calipers"] = 8  # a copy
    assert block["caliper_based_circle_fitting"]["num_calipers"] == 72


def test_downscale_converts_both_stages_and_the_padding() -> None:
    scaled = TemplateCaliperCircleDetector.downscaled_params(
        {
            "region_padding_px": 20,
            "template_matching": {"scales": [1.0]},
            "caliper_based_circle_fitting": {"inlier_threshold_px": 3.0, "caliper_width_px": 8},
        },
        4.0,
    )
    assert scaled["region_padding_px"] == 5
    assert scaled["template_matching"]["scales"] == [0.25]
    assert scaled["caliper_based_circle_fitting"]["inlier_threshold_px"] == 0.75
    assert scaled["caliper_based_circle_fitting"]["caliper_width_px"] == 2


def test_engine_runs_it_from_a_camera_block(template_path) -> None:
    engine = VisionEngine({"cameras": {"1": camera_block(template_path)}})
    assert engine.active_detector_name(1) == "template_caliper_based_circle_fitting"
    result = engine.detect(make_frame(), 1)
    assert len(result.holes) == 1 and near(result.holes[0], HOLE_IN_PLATE)


def test_engine_keeps_the_camera_block_verbatim(template_path) -> None:
    block = camera_block(template_path)
    engine = VisionEngine({"cameras": {"1": copy.deepcopy(block)}})
    assert engine.camera_config(1) == block

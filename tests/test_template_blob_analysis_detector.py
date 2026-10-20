"""template_blob_analysis: the template locates a region, blob analysis finds the hole in it.

Blob analysis on its own reports the *largest* round dark blob in the frame,
so its weak spot is any bigger round dark feature elsewhere. The fixture
here is built to hit exactly that: a small bore inside a distinctive stamped
plate (the template), and a **bigger** bore on bare metal. Plain blob
analysis picks the bigger one; the two-stage strategy must pick the one in
the plate. (The plate is the same ribbed one ``test_template_opencv_detector``
uses, for the same reason: a template that is mostly a bore matches any bore.)
"""

import copy

import cv2
import numpy as np
import pytest

from core.utilities.exceptions import DetectionError
from core.vision import (
    BlobAnalysisDetector,
    Hole,
    TemplateBlobAnalysisDetector,
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
RADIUS_ON_METAL = 30  # four times the area of the bore in the plate

BLOB = {
    "threshold": 50,
    "blur_kernel_size": 3,
    "morphology_kernel_size": 3,
    "min_area_px": 100,
    "min_circularity": 0.7,
    "min_hole_diameter_px": 0,
    "max_hole_diameter_px": 0,
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
        "blob_analysis": dict(BLOB),
    }


def near(hole: Hole, point: tuple[int, int], tolerance: float = 1.5) -> bool:
    return abs(hole.x_px - point[0]) <= tolerance and abs(hole.y_px - point[1]) <= tolerance


# ------------------------------------------------------------- the point
def test_plain_blob_analysis_picks_the_bigger_bore_on_bare_metal() -> None:
    """Sanity check of the fixture: this is the failure the template fixes."""
    result = BlobAnalysisDetector(BLOB).detect(make_frame())
    assert near(result.best, HOLE_ON_METAL)


def test_only_the_bore_inside_the_template_is_reported(template_path) -> None:
    result = TemplateBlobAnalysisDetector(params(template_path)).detect(make_frame())
    assert len(result.holes) == 1
    assert near(result.holes[0], HOLE_IN_PLATE)  # frame pixels, not crop pixels
    assert len(result.regions) == 1
    region = result.regions[0]
    assert (region.x, region.y, region.width, region.height) == (
        PLATE_X, PLATE_Y, PLATE_SIDE, PLATE_SIDE,
    )


def test_no_template_match_is_no_hole_not_an_error(template_path) -> None:
    """The big bore on bare metal is still there; without the plate it is ignored."""
    result = TemplateBlobAnalysisDetector(params(template_path)).detect(make_frame(plate=False))
    assert result.holes == []
    assert result.regions == []


def test_a_match_with_no_hole_inside_is_empty_but_keeps_the_region(template_path) -> None:
    result = TemplateBlobAnalysisDetector(params(template_path)).detect(
        make_frame(hole_in_plate=False)
    )
    assert result.holes == []
    assert len(result.regions) == 1


def test_confidence_is_the_blob_stage_score(template_path) -> None:
    frame = make_frame()
    combined = TemplateBlobAnalysisDetector(params(template_path)).detect(frame)
    crop = frame[PLATE_Y : PLATE_Y + PLATE_SIDE, PLATE_X : PLATE_X + PLATE_SIDE]
    alone = BlobAnalysisDetector(BLOB).detect(crop)
    assert combined.holes[0].confidence == pytest.approx(alone.holes[0].confidence)


def test_one_hole_per_matched_region(template_path) -> None:
    """Blob analysis reports one hole per image, so Regions Searched is also
    how many holes this strategy can report."""
    frame = make_frame()
    draw_plate(frame, 280, 40)  # a second plate, its bore at (340, 80)
    one = TemplateBlobAnalysisDetector(params(template_path, max_regions=1)).detect(frame)
    two = TemplateBlobAnalysisDetector(params(template_path, max_regions=2)).detect(frame)
    assert len(one.regions) == 1 and len(one.holes) == 1
    assert len(two.regions) == 2 and len(two.holes) == 2
    assert not any(near(hole, HOLE_ON_METAL, tolerance=5) for hole in two.holes)


# -------------------------------------------------------------- failures
def test_missing_template_is_a_detect_time_error_naming_the_stage() -> None:
    detector = TemplateBlobAnalysisDetector(params(""))
    with pytest.raises(DetectionError, match="template_blob_analysis — template stage"):
        detector.detect(make_frame())


# ------------------------------------------------------------- debug view
def test_debug_stages_are_the_blob_mask_inside_the_region_only(template_path) -> None:
    frame = make_frame()
    stages = TemplateBlobAnalysisDetector(params(template_path)).debug_stages(frame)
    assert set(stages) == {"mask"}
    assert stages["mask"].shape == frame.shape
    assert stages["mask"][HOLE_IN_PLATE[1], HOLE_IN_PLATE[0]] == 255
    assert stages["mask"][HOLE_ON_METAL[1], HOLE_ON_METAL[0]] == 0  # never looked at


def test_debug_stages_are_blank_not_missing_without_a_match(template_path) -> None:
    """Blank keeps the overlay saying "0 contours" rather than "no debug view"."""
    stages = TemplateBlobAnalysisDetector(params(template_path)).debug_stages(
        make_frame(plate=False)
    )
    assert set(stages) == {"mask"} and not stages["mask"].any()


# ------------------------------------------------- parameters and engine
def camera_block(template_path: str) -> dict:
    return {
        "active_detector": "template_blob_analysis",
        "common": {"confidence_threshold": 0.3, "expected_hole_count": 1},
        "template_blob_analysis": {"max_regions": 1, "region_padding_px": 0},
        "template_matching": params(template_path)["template_matching"],
        "blob_analysis": dict(BLOB),
        "opencv": {"detection_threshold": 1},  # another strategy's block: not read
    }


def test_params_are_gathered_from_the_sibling_blocks(template_path) -> None:
    block = camera_block(template_path)
    gathered = TemplateBlobAnalysisDetector.params_from_camera_config(block)
    assert gathered["template_matching"] == block["template_matching"]
    assert gathered["blob_analysis"] == block["blob_analysis"]
    assert gathered["max_regions"] == 1
    assert "opencv" not in gathered
    gathered["blob_analysis"]["threshold"] = 1  # a copy: the config is untouched
    assert block["blob_analysis"]["threshold"] == 50


def test_downscale_converts_both_stages_and_the_padding() -> None:
    scaled = TemplateBlobAnalysisDetector.downscaled_params(
        {
            "region_padding_px": 20,
            "template_matching": {"scales": [1.0]},
            "blob_analysis": {"min_area_px": 1600, "blur_kernel_size": 7},
        },
        4.0,
    )
    assert scaled["region_padding_px"] == 5
    assert scaled["template_matching"]["scales"] == [0.25]
    assert scaled["blob_analysis"]["min_area_px"] == 100  # an area: ÷ 4²
    assert scaled["blob_analysis"]["blur_kernel_size"] == 2


def test_engine_runs_it_from_a_camera_block(template_path) -> None:
    engine = VisionEngine({"cameras": {"1": camera_block(template_path)}})
    assert engine.active_detector_name(1) == "template_blob_analysis"
    result = engine.detect(make_frame(), 1)
    assert len(result.holes) == 1 and near(result.holes[0], HOLE_IN_PLATE)


def test_engine_keeps_the_camera_block_verbatim(template_path) -> None:
    block = camera_block(template_path)
    engine = VisionEngine({"cameras": {"1": copy.deepcopy(block)}})
    assert engine.camera_config(1) == block

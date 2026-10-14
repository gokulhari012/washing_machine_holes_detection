"""Per-camera downscale (``common.downscale_enabled`` / ``downscale_factor``).

The two promises ``VisionEngine`` makes, pinned here:

1. results come back in the *input* frame's pixels — the calibration, the
   overlay and the PLC position must never see a shrunk coordinate;
2. parameters keep their full-resolution meaning — a camera tuned at full
   size keeps finding its hole when downscale is switched on, with nothing
   re-tuned. A 90 px bore at ÷4 is 22 px, which an unconverted
   ``min_hole_diameter_px: 60`` gate would reject outright.

Also the two new per-camera accessors the Detection page and the pipeline
read: ``downscale_factor`` and ``ng_rerun_count``.
"""

import copy

import cv2
import numpy as np
import pytest

from core.vision import (
    DarkHoleDetector,
    OpenCVHoleDetector,
    TemplateMatchingDetector,
    TemplateOpenCVDetector,
    VisionEngine,
)
from core.vision.vision_engine import MAX_DOWNSCALE_FACTOR, MAX_NG_RERUNS

CENTRE = (523, 377)  # deliberately not a multiple of any factor
RADIUS = 45

OPENCV = {
    "detection_threshold": 60,
    "blur_kernel_size": 5,
    "morphology_operation": "open",
    "morphology_kernel_size": 5,
    "morphology_iterations": 1,
    "min_hole_diameter_px": 60,
    "max_hole_diameter_px": 130,
    "min_circularity": 0.5,
    "min_aspect_ratio": 0.5,
    "contour_retrieval_mode": "external",
}
DARK_HOLE = {
    "channel": "gray",
    "blur_kernel_size": 3,
    "min_contrast": 20,
    "use_otsu": True,
    "morphology_kernel_size": 3,
    "min_hole_diameter_px": 60,
    "max_hole_diameter_px": 130,
    "min_fill_ratio": 0.35,
    "max_fit_error": 0.3,
}


def make_frame() -> np.ndarray:
    frame = np.full((800, 1100), 190, np.uint8)
    cv2.circle(frame, CENTRE, RADIUS, 25, -1)
    return frame


def engine_for(strategy: str, params: dict, factor: float | None, **extra) -> VisionEngine:
    common = {"confidence_threshold": 0.0, "expected_hole_count": 1}
    if factor is not None:
        common.update(downscale_enabled=True, downscale_factor=factor)
    block = {"active_detector": strategy, "common": common, strategy: params, **extra}
    return VisionEngine({"cameras": {"1": block}})


# ------------------------------------------------- full-resolution results
@pytest.mark.parametrize("strategy, params", [("opencv", OPENCV), ("dark_hole", DARK_HOLE)])
@pytest.mark.parametrize("factor", [2.0, 3.0, 4.0])
def test_hole_is_reported_in_full_resolution_pixels(strategy, params, factor) -> None:
    full = engine_for(strategy, params, None).detect(make_frame(), 1).best
    small = engine_for(strategy, params, factor).detect(make_frame(), 1).best
    assert full is not None and small is not None  # gates were converted, not left at 60 px
    assert small.x_px == pytest.approx(full.x_px, abs=factor)
    assert small.y_px == pytest.approx(full.y_px, abs=factor)
    assert small.diameter_px == pytest.approx(full.diameter_px, abs=2 * factor)


def test_downscale_off_is_bit_for_bit_the_old_behaviour() -> None:
    frame = make_frame()
    reference = OpenCVHoleDetector(OPENCV).detect(frame).holes
    assert engine_for("opencv", OPENCV, None).detect(frame, 1).holes == reference


def test_template_matching_shrinks_its_template_with_the_frame(tmp_path) -> None:
    template = np.full((110, 110), 190, np.uint8)
    cv2.circle(template, (55, 55), RADIUS, 25, -1)
    path = tmp_path / "bore.png"
    cv2.imwrite(str(path), template)
    params = {"template_path": str(path), "match_threshold": 0.8, "scales": [1.0]}
    best = engine_for("template_matching", params, 2.0).detect(make_frame(), 1).best
    assert best is not None
    assert best.x_px == pytest.approx(CENTRE[0], abs=2)
    assert best.y_px == pytest.approx(CENTRE[1], abs=2)
    assert best.diameter_px == pytest.approx(110, abs=3)  # the template's full-size side


def test_search_regions_come_back_in_full_resolution_pixels(tmp_path) -> None:
    frame = make_frame()
    crop = frame[CENTRE[1] - 80 : CENTRE[1] + 80, CENTRE[0] - 80 : CENTRE[0] + 80].copy()
    cv2.rectangle(crop, (4, 4), (155, 155), 250, 6)  # give the template a feature
    frame[CENTRE[1] - 80 : CENTRE[1] + 80, CENTRE[0] - 80 : CENTRE[0] + 80] = crop
    path = tmp_path / "region.png"
    cv2.imwrite(str(path), crop)
    template = {"template_path": str(path), "match_threshold": 0.8, "scales": [1.0]}
    engine = engine_for(
        "template_opencv", {"max_regions": 1}, 2.0,
        template_matching=template, opencv=OPENCV,
    )
    result = engine.detect(frame, 1)
    region = result.regions[0]
    assert region.x == pytest.approx(CENTRE[0] - 80, abs=2)
    assert region.y == pytest.approx(CENTRE[1] - 80, abs=2)
    assert region.width == pytest.approx(160, abs=2)
    assert result.best.x_px == pytest.approx(CENTRE[0], abs=2)


def test_debug_stages_are_scaled_back_to_the_input_size() -> None:
    """draw_debug_overlay indexes the full frame with the mask."""
    frame = make_frame()
    stages = engine_for("opencv", OPENCV, 4.0).debug_stages(frame, 1)
    assert stages["mask"].shape == frame.shape
    assert stages["edges"].shape == frame.shape
    assert stages["mask"][CENTRE[1], CENTRE[0]] == 255


# --------------------------------------------------- parameter conversion
def test_opencv_converts_gates_and_kernels() -> None:
    scaled = OpenCVHoleDetector.downscaled_params(OPENCV, 2.0)
    assert scaled["min_hole_diameter_px"] == 30
    assert scaled["max_hole_diameter_px"] == 65
    assert scaled["blur_kernel_size"] == 2  # rounded; the detector makes it odd
    assert scaled["morphology_kernel_size"] == 2
    assert scaled["detection_threshold"] == 60  # a grey level, not a length
    assert OPENCV["min_hole_diameter_px"] == 60  # the input is not modified


def test_a_kernel_never_shrinks_below_one_pixel() -> None:
    assert DarkHoleDetector.downscaled_params({"blur_kernel_size": 1}, 8.0)["blur_kernel_size"] == 1


def test_a_disabled_gate_stays_disabled() -> None:
    scaled = TemplateMatchingDetector.downscaled_params(
        {"min_hole_diameter_px": 0, "max_hole_diameter_px": 0, "scales": [0.8, 1.0]}, 2.0
    )
    assert scaled["min_hole_diameter_px"] == 0
    assert scaled["max_hole_diameter_px"] == 0
    assert scaled["scales"] == [0.4, 0.5]


def test_the_combined_strategy_converts_both_stages() -> None:
    params = {
        "region_padding_px": 20,
        "template_matching": {"scales": [1.0]},
        "opencv": {"min_hole_diameter_px": 60, "blur_kernel_size": 5},
    }
    scaled = TemplateOpenCVDetector.downscaled_params(params, 4.0)
    assert scaled["region_padding_px"] == 5
    assert scaled["template_matching"]["scales"] == [0.25]
    assert scaled["opencv"]["min_hole_diameter_px"] == 15
    assert scaled["opencv"]["blur_kernel_size"] == 1


def test_the_page_still_sees_the_values_that_were_typed() -> None:
    """camera_config is what the Detection page renders; the converted
    parameters exist only inside the detector."""
    engine = engine_for("opencv", copy.deepcopy(OPENCV), 2.0)
    assert engine.camera_config(1)["opencv"] == OPENCV


# ------------------------------------------------------------ accessors
@pytest.mark.parametrize(
    "common, expected",
    [
        ({}, 1.0),  # an older detection.json with no key at all
        ({"downscale_enabled": False, "downscale_factor": 4.0}, 1.0),
        ({"downscale_enabled": True, "downscale_factor": 4.0}, 4.0),
        ({"downscale_enabled": True}, 2.0),  # the default factor
        ({"downscale_enabled": True, "downscale_factor": 0.5}, 1.0),  # an upscale: off
        ({"downscale_enabled": True, "downscale_factor": "abc"}, 1.0),
        ({"downscale_enabled": True, "downscale_factor": 999}, MAX_DOWNSCALE_FACTOR),
    ],
)
def test_downscale_factor(common, expected) -> None:
    engine = VisionEngine({"cameras": {"1": {"active_detector": "opencv", "common": common}}})
    assert engine.downscale_factor(1) == expected


@pytest.mark.parametrize(
    "common, expected",
    [
        ({}, 0),
        ({"ng_rerun_enabled": False, "ng_rerun_count": 3}, 0),
        ({"ng_rerun_enabled": True, "ng_rerun_count": 3}, 3),
        ({"ng_rerun_enabled": True}, 1),
        ({"ng_rerun_enabled": True, "ng_rerun_count": 999}, MAX_NG_RERUNS),
        ({"ng_rerun_enabled": True, "ng_rerun_count": -2}, 0),
        ({"ng_rerun_enabled": True, "ng_rerun_count": "x"}, 0),
    ],
)
def test_ng_rerun_count(common, expected) -> None:
    engine = VisionEngine({"cameras": {"1": {"active_detector": "opencv", "common": common}}})
    assert engine.ng_rerun_count(1) == expected


def test_each_camera_has_its_own_downscale() -> None:
    engine = VisionEngine({"cameras": {
        "1": {"active_detector": "opencv",
              "common": {"downscale_enabled": True, "downscale_factor": 3.0}},
        "2": {"active_detector": "opencv", "common": {}},
    }})
    assert engine.downscale_factor(1) == 3.0
    assert engine.downscale_factor(2) == 1.0

"""Detection page Auto Sweep: ROI-derived gates, cropping, and grid coverage.

These exercise plain functions and ``@staticmethod``s on ``DetectionPage`` —
no ``QApplication``/widget instantiation needed — plus an end-to-end check
that the crop+offset math still lands a hole inside the original ROI.
"""

import cv2
import numpy as np
import pytest

from core.utilities.exceptions import DetectionError
from core.vision import DarkHoleDetector, OpenCVHoleDetector, TemplateMatchingDetector
from ui.detection.detection_page import (
    DetectionPage,
    _SWEEP_GRID_BUILDERS,
    _SWEEP_LEVELS,
    _SWEEP_SCALE_STEPS,
    _dark_hole_param_grid,
    _format_scales,
    _opencv_param_grid,
    _scales_from_roi,
    _template_matching_param_grid,
)


def test_diameter_bounds_from_roi() -> None:
    min_d, max_d = DetectionPage._diameter_bounds_from_roi((10, 10, 40, 60))
    assert min_d == pytest.approx(20.0)  # 0.5 * short side (40)
    assert max_d == pytest.approx(90.0)  # 1.5 * long side (60)


def test_diameter_bounds_from_roi_floors_tiny_boxes() -> None:
    min_d, max_d = DetectionPage._diameter_bounds_from_roi((0, 0, 2, 2))
    assert min_d == pytest.approx(4.0)  # floor, not 0.5*2=1.0
    assert max_d == pytest.approx(8.0)  # floor, not 1.5*2=3.0
    assert max_d > min_d


def test_crop_around_roi_shape_and_offset() -> None:
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    roi = (100, 80, 40, 30)
    cropped, offset_x, offset_y = DetectionPage._crop_around_roi(frame, roi, margin_factor=1.0)
    # pad = 40*1.0=40, 30*1.0=30 -> x:[60,180), y:[50,140)
    assert (offset_x, offset_y) == (60, 50)
    assert cropped.shape == (90, 120, 3)


def test_crop_around_roi_clamps_to_frame_bounds() -> None:
    frame = np.zeros((50, 50), dtype=np.uint8)
    cropped, offset_x, offset_y = DetectionPage._crop_around_roi(
        frame, (0, 0, 10, 10), margin_factor=5.0
    )
    assert (offset_x, offset_y) == (0, 0)
    assert cropped.shape[0] <= 50 and cropped.shape[1] <= 50


def test_best_in_roi_requires_center_inside_box() -> None:
    from core.vision.detection_result import Hole

    inside = Hole(x_px=15.0, y_px=15.0, diameter_px=10.0, circularity=1.0, confidence=0.9)
    outside = Hole(x_px=500.0, y_px=500.0, diameter_px=10.0, circularity=1.0, confidence=0.99)
    result = DetectionPage._best_in_roi([outside, inside], (10, 10, 20, 20))
    assert result is inside


def test_opencv_grid_prunes_adaptive_and_none_morphology() -> None:
    combos = _opencv_param_grid({})
    # adaptive branch shouldn't vary detection_threshold at all
    adaptive_thresholds = {c["detection_threshold"] for c in combos if c["adaptive_threshold"]}
    assert len(adaptive_thresholds) == 1
    # morphology_operation "none" shouldn't vary kernel/iterations
    none_morph = [c for c in combos if c["morphology_operation"] == "none"]
    assert none_morph
    assert len({c["morphology_kernel_size"] for c in none_morph}) == 1
    assert len({c["morphology_iterations"] for c in none_morph}) == 1


def test_opencv_grid_passes_through_unswept_base_params() -> None:
    combos = _opencv_param_grid({"edge_threshold_low": 77, "edge_threshold_high": 199})
    assert all(c["edge_threshold_low"] == 77 for c in combos)
    assert all(c["edge_threshold_high"] == 199 for c in combos)


def test_dark_hole_grid_covers_every_channel() -> None:
    combos = _dark_hole_param_grid({})
    assert {c["channel"] for c in combos} == {"auto", "gray", "red", "green", "blue"}


def _frame_with_hole(size: int = 300, center: tuple[int, int] = (150, 150), radius: int = 25):
    image = np.full((size, size), 180, dtype=np.uint8)
    cv2.circle(image, center, radius, 20, -1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_sweep_pipeline_finds_hole_through_crop_and_offset() -> None:
    """End-to-end: derive gates from an ROI around a synthetic hole, crop the
    frame to it, run a grid combo through the real detector, and confirm the
    offset-corrected hole still lands inside the *original* (uncropped) ROI —
    the exact path _on_run_sweep exercises per combination."""
    frame = _frame_with_hole()
    roi = (120, 120, 60, 60)  # tightly around the 50px-diameter hole

    min_d, max_d = DetectionPage._diameter_bounds_from_roi(roi)
    cropped, offset_x, offset_y = DetectionPage._crop_around_roi(frame, roi)

    base = {"min_hole_diameter_px": min_d, "max_hole_diameter_px": max_d}
    combos = _opencv_param_grid(base)

    found = False
    for params in combos[:50]:  # a sample is enough to prove the pipeline works
        result = OpenCVHoleDetector(params).detect(cropped)
        from dataclasses import replace

        holes = [replace(h, x_px=h.x_px + offset_x, y_px=h.y_px + offset_y) for h in result.holes]
        hole = DetectionPage._best_in_roi(holes, roi)
        if hole is not None:
            found = True
            assert roi[0] <= hole.x_px <= roi[0] + roi[2]
            assert roi[1] <= hole.y_px <= roi[1] + roi[3]
            break
    assert found, "expected at least one opencv combo to find the synthetic hole"


def test_dark_hole_sweep_pipeline_finds_hole() -> None:
    frame = _frame_with_hole()
    roi = (120, 120, 60, 60)
    min_d, max_d = DetectionPage._diameter_bounds_from_roi(roi)
    cropped, offset_x, offset_y = DetectionPage._crop_around_roi(frame, roi)

    base = {"min_hole_diameter_px": min_d, "max_hole_diameter_px": max_d}
    combos = _dark_hole_param_grid(base)

    found = False
    for params in combos[:50]:
        result = DarkHoleDetector(params).detect(cropped)
        from dataclasses import replace

        holes = [replace(h, x_px=h.x_px + offset_x, y_px=h.y_px + offset_y) for h in result.holes]
        hole = DetectionPage._best_in_roi(holes, roi)
        if hole is not None:
            found = True
            break
    assert found, "expected at least one dark_hole combo to find the synthetic hole"


# ---------------------------------------------- template_matching sweep
def _template_file(tmp_path, side: int = 100, radius: int = 40) -> str:
    """A template whose bore is ``2*radius`` px across in a ``side`` px crop."""
    template = np.full((side, side), 180, np.uint8)
    cv2.circle(template, (side // 2, side // 2), radius, 20, -1)
    path = tmp_path / "template.png"
    cv2.imwrite(str(path), template)
    return str(path)


def test_scales_come_from_the_roi_not_a_fixed_ladder(tmp_path) -> None:
    """The ROI bounds the hole's diameter; dividing by the template's own
    size turns that straight into the scale range worth searching."""
    base = {
        "template_path": _template_file(tmp_path),  # 100 px mean side
        "min_hole_diameter_px": 30.0,
        "max_hole_diameter_px": 90.0,
    }
    scales = _scales_from_roi(base)
    assert scales[0] == pytest.approx(0.3, abs=0.01)
    assert scales[-1] == pytest.approx(0.9, abs=0.01)
    assert len(scales) == _SWEEP_SCALE_STEPS
    assert scales == sorted(scales)


def test_scales_from_roi_stay_positive_for_a_zero_gate(tmp_path) -> None:
    """A 0 min gate must not collapse the range to 0 (or divide by it)."""
    base = {
        "template_path": _template_file(tmp_path),
        "min_hole_diameter_px": 0.0,
        "max_hole_diameter_px": 0.0,
    }
    scales = _scales_from_roi(base)
    assert all(scale > 0 for scale in scales)
    assert scales[-1] > scales[0]


def test_scales_from_roi_without_a_template_is_a_configuration_error() -> None:
    with pytest.raises(DetectionError, match="template_path"):
        _scales_from_roi({"min_hole_diameter_px": 10, "max_hole_diameter_px": 50})


def test_template_grid_covers_methods_scales_and_thresholds(tmp_path) -> None:
    base = {
        "template_path": _template_file(tmp_path),
        "min_hole_diameter_px": 30.0,
        "max_hole_diameter_px": 90.0,
    }
    combos = _template_matching_param_grid(base)
    levels = _SWEEP_LEVELS["template_matching"]
    assert len(combos) == len(levels["method"]) * _SWEEP_SCALE_STEPS * len(
        levels["match_threshold"]
    )
    assert {combo["method"] for combo in combos} == set(levels["method"])
    # one scale per trial, so the winning row names the scale that fitted
    assert all(len(combo["scales"]) == 1 for combo in combos)
    assert combos[0]["template_path"] == base["template_path"]  # base passes through


def test_template_grid_tries_the_strictest_threshold_first(tmp_path) -> None:
    """Every threshold a candidate clears scores it identically, and the sort
    is stable — so descending order is what makes the top row the tightest
    threshold that still found the hole, rather than the loosest."""
    base = {
        "template_path": _template_file(tmp_path),
        "min_hole_diameter_px": 30.0,
        "max_hole_diameter_px": 90.0,
    }
    combos = _template_matching_param_grid(base)
    first_group = [
        combo["match_threshold"]
        for combo in combos
        if combo["method"] == combos[0]["method"] and combo["scales"] == combos[0]["scales"]
    ]
    assert first_group == sorted(first_group, reverse=True)


def test_template_sweep_pipeline_finds_the_hole(tmp_path) -> None:
    """End-to-end, the path _on_run_sweep drives: ROI -> gates -> scales ->
    crop -> one reconfigured detector per trial -> offset back into the ROI."""
    from dataclasses import replace

    frame = _frame_with_hole()  # 50 px bore at (150, 150)
    roi = (120, 120, 60, 60)
    min_d, max_d = DetectionPage._diameter_bounds_from_roi(roi)
    cropped, offset_x, offset_y = DetectionPage._crop_around_roi(frame, roi)

    base = {
        "template_path": _template_file(tmp_path),
        "min_hole_diameter_px": min_d,
        "max_hole_diameter_px": max_d,
    }
    detector = TemplateMatchingDetector(base)  # one instance, reconfigured per trial
    found = []
    for params in _template_matching_param_grid(base):
        try:
            detector.configure(params)
            result = detector.detect(cropped)
        except DetectionError:
            continue
        holes = [
            replace(h, x_px=h.x_px + offset_x, y_px=h.y_px + offset_y) for h in result.holes
        ]
        hole = DetectionPage._best_in_roi(holes, roi)
        if hole is not None:
            found.append((params, hole))

    assert found, "expected at least one template_matching combo to find the hole"
    best_params, best_hole = sorted(found, key=lambda item: item[1].confidence, reverse=True)[0]
    assert roi[0] <= best_hole.x_px <= roi[0] + roi[2]
    assert roi[1] <= best_hole.y_px <= roi[1] + roi[3]
    # the winning scale must be the one that maps the template onto a ~50 px bore
    assert best_params["scales"][0] == pytest.approx(0.6, abs=0.15)


def test_yolo_is_not_sweepable_and_says_why() -> None:
    assert "yolo" not in _SWEEP_GRID_BUILDERS
    reason = DetectionPage._unsweepable_reason("yolo")
    assert "yolo" in reason and "crop" in reason


def test_unknown_strategy_gets_a_generic_reason() -> None:
    assert "whatever" in DetectionPage._unsweepable_reason("whatever")


# ------------------------------------------------------- cell formatting
@pytest.mark.parametrize(
    "value, expected",
    [
        ([1.05], "1.05"),
        ([0.9, 1.0, 1.1], "0.9, 1, 1.1"),
        ("close", "close"),
        (True, "True"),
        (5, "5"),
    ],
)
def test_sweep_cell_formatting(value, expected) -> None:
    assert DetectionPage._format_cell(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [([1.0], "1"), ([0.9, 1.0, 1.1], "0.9, 1, 1.1"), ("", "1"), (None, "1")],
)
def test_scales_round_trip_through_the_form_field(value, expected) -> None:
    assert _format_scales(value) == expected

"""Hole detectors on synthetic images with known ground truth."""

import cv2
import numpy as np
import pytest

from core.utilities.exceptions import DetectionError
from core.vision import OpenCVHoleDetector, TemplateMatchingDetector, VisionEngine

PARAMS = {
    "detection_threshold": 60,
    "blur_kernel_size": 5,
    "edge_threshold_low": 50,
    "edge_threshold_high": 150,
    "morphology_operation": "close",
    "morphology_kernel_size": 5,
    "morphology_iterations": 1,
    "min_hole_diameter_px": 20,
    "max_hole_diameter_px": 200,
    "min_circularity": 0.7,
}

HOLE_CENTER = (320, 240)
HOLE_RADIUS = 30


def make_frame(*, hole: bool = True) -> np.ndarray:
    image = np.full((480, 640), 110, dtype=np.uint8)
    # bolt distractors below the minimum diameter
    for center in ((80, 90), (560, 400), (500, 100)):
        cv2.circle(image, center, 7, 50, -1)
    if hole:
        cv2.circle(image, HOLE_CENTER, HOLE_RADIUS, 20, -1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_finds_hole_accurately() -> None:
    result = OpenCVHoleDetector(PARAMS).detect(make_frame())
    assert len(result.holes) == 1
    best = result.best
    assert abs(best.x_px - HOLE_CENTER[0]) <= 2
    assert abs(best.y_px - HOLE_CENTER[1]) <= 2
    assert abs(best.diameter_px - 2 * HOLE_RADIUS) <= 4
    assert best.confidence > 0.7


def test_no_hole_is_empty_not_error() -> None:
    result = OpenCVHoleDetector(PARAMS).detect(make_frame(hole=False))
    assert not result.found and result.best is None


def test_size_gate_rejects_large_blobs() -> None:
    params = dict(PARAMS, max_hole_diameter_px=40)
    result = OpenCVHoleDetector(params).detect(make_frame())
    assert not result.found  # 60 px hole exceeds the 40 px maximum


def test_template_detector_requires_template() -> None:
    detector = TemplateMatchingDetector({"match_threshold": 0.8})
    with pytest.raises(DetectionError):
        detector.detect(make_frame())


def test_engine_filters_by_confidence() -> None:
    engine = VisionEngine(
        {
            "active_detector": "opencv",
            "common": {"confidence_threshold": 0.99},
            "opencv": PARAMS,
        }
    )
    assert not engine.detect(make_frame()).found  # nothing scores 0.99

    engine.apply_config(
        {
            "active_detector": "opencv",
            "common": {"confidence_threshold": 0.5},
            "opencv": PARAMS,
        }
    )
    assert engine.detect(make_frame()).found

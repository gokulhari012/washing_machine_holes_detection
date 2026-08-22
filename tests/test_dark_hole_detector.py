"""dark_hole strategy: whole and partially visible bores, and what it refuses.

The scenes mimic the production images — a bore on textured, unevenly lit
metal under a red light, next to large dark and specular areas that a global
threshold would trip over.
"""

import math

import cv2
import numpy as np
import pytest

from core.utilities.exceptions import DetectionError
from core.vision.dark_hole_detector import DarkHoleDetector

CENTRE = (330, 218)
RADIUS = 26
PARAMS = {
    "channel": "auto",
    "blur_kernel_size": 3,
    "min_contrast": 18,
    "use_otsu": True,
    "morphology_kernel_size": 3,
    "min_hole_diameter_px": 15,
    "max_hole_diameter_px": 120,
    "min_fill_ratio": 0.35,
    "max_fit_error": 0.25,
}
CONFIDENCE = 0.6


def scene(clip: str = "none", *, bore: bool = True, glare: bool = False) -> np.ndarray:
    """Red-lit textured plate, a stamped bracket, and a bore inside its slot."""
    rng = np.random.default_rng(7)
    height, width = 436, 662
    plate = np.full((height, width), 165, np.float32)
    plate += rng.normal(0, 14, (height, width)).astype(np.float32)
    yy = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    plate *= 1.0 - 0.30 * (0.6 * xx * xx + yy * yy)      # uneven lighting
    cv2.rectangle(plate, (0, 0), (150, 200), 55, -1)      # shadowed corner
    cv2.rectangle(plate, (560, 0), (585, height), 245, -1)  # specular band
    cv2.rectangle(plate, (215, 150), (450, 290), 90, 14)  # bracket outline...
    cv2.rectangle(plate, (240, 175), (425, 265), 205, -1)  # ...around a bright slot
    image = np.clip(plate, 0, 255).astype(np.uint8)

    if bore:
        stencil = np.zeros_like(image)
        cv2.circle(stencil, CENTRE, RADIUS, 255, -1)
        if clip == "half":            # rim runs behind the slot edge
            cv2.rectangle(stencil, (0, 0), (width, CENTRE[1]), 0, -1)
        elif clip == "third":
            cv2.rectangle(stencil, (0, 0), (CENTRE[0] - 8, height), 0, -1)
        image[stencil > 0] = 28
        if glare:
            cv2.circle(image, (CENTRE[0] + 6, CENTRE[1] - 5), 6, 230, -1)

    colour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.float32)
    colour *= np.array([0.62, 0.66, 1.0], np.float32)     # red ring light
    return np.clip(colour, 0, 255).astype(np.uint8)


def best_hole(image: np.ndarray, **overrides):
    result = DarkHoleDetector(dict(PARAMS, **overrides)).detect(image)
    hits = [hole for hole in result.holes if hole.confidence >= CONFIDENCE]
    return hits[0] if hits else None


@pytest.mark.parametrize(
    "clip, glare, min_coverage",
    [("none", False, 0.9), ("none", True, 0.9), ("half", False, 0.4), ("third", False, 0.4)],
)
def test_finds_bore_whole_or_partly_visible(clip, glare, min_coverage) -> None:
    hole = best_hole(scene(clip, glare=glare))
    assert hole is not None, f"missed the bore (clip={clip}, glare={glare})"
    assert math.hypot(hole.x_px - CENTRE[0], hole.y_px - CENTRE[1]) <= 6.0
    # the fit describes the whole bore even when only an arc of it is visible
    assert hole.diameter_px == pytest.approx(2 * RADIUS, rel=0.2)
    assert hole.circularity >= min_coverage


def test_coverage_reports_how_much_of_the_rim_is_visible() -> None:
    whole = best_hole(scene("none"))
    half = best_hole(scene("half"))
    assert whole is not None and half is not None
    assert whole.circularity > half.circularity + 0.2


def test_no_bore_gives_no_candidate() -> None:
    assert best_hole(scene(bore=False)) is None


def test_low_contrast_bore_still_found() -> None:
    """A bore only ~75 grey levels darker than the metal is still a bore."""
    image = scene("none")
    stencil = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(stencil, CENTRE, RADIUS, 255, -1)
    image[stencil > 0] = (80, 85, 128)
    assert best_hole(image) is not None


def test_size_gates_are_honoured() -> None:
    image = scene("none")
    assert best_hole(image, min_hole_diameter_px=90) is None   # bore is ~52 px
    assert best_hole(image, max_hole_diameter_px=30) is None


def test_explain_reports_rejection_reasons() -> None:
    stages, candidates = DarkHoleDetector(
        dict(PARAMS, min_hole_diameter_px=90)
    ).explain(scene("none"))
    assert {"gray", "response", "mask"} <= set(stages)
    assert candidates, "the bore should still be listed, as a rejected candidate"
    assert all(candidate.rejected_because for candidate in candidates)
    assert any("below min" in candidate.rejected_because for candidate in candidates)


def test_invalid_configuration_raises() -> None:
    with pytest.raises(DetectionError):
        DarkHoleDetector(dict(PARAMS, min_hole_diameter_px=200)).detect(scene("none"))
    with pytest.raises(DetectionError):
        DarkHoleDetector(dict(PARAMS, channel="infrared")).detect(scene("none"))


def test_grayscale_input_is_accepted() -> None:
    colour = scene("none")
    assert best_hole(cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)) is not None

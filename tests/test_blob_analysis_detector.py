"""blob_analysis: the largest dark blob that is round enough is the hole.

The rule ported from the lab script (``development_files/testing_own``) is
*largest area wins among the blobs that pass the gates* — not "roundest
wins", which is what reporting every blob would turn it into, since the
pipeline judges the highest-confidence hole and confidence is circularity.
The fixture frame therefore carries three dark blobs, each there to break a
different wrong rule:

- a large round bore — the hole;
- a smaller, equally round bore — what "report the roundest" or "report
  them all" would let compete;
- a dark bar bigger than both — what "largest wins" with no roundness gate
  would pick.
"""

import copy

import cv2
import numpy as np
import pytest

from core.utilities.exceptions import DetectionError
from core.vision import BlobAnalysisDetector, VisionEngine

METAL = 190
DARK = 20

BIG_BORE, BIG_RADIUS = (150, 150), 40
SMALL_BORE, SMALL_RADIUS = (420, 150), 20
BAR = ((100, 330), (500, 365))  # 400 x 35 = 14000 px², far more than the big bore

BLOB = {
    "threshold": 40,
    "blur_kernel_size": 7,
    "morphology_kernel_size": 5,
    "min_area_px": 500,
    "min_circularity": 0.8,
    "min_hole_diameter_px": 0,
    "max_hole_diameter_px": 0,
}


def make_frame(big: bool = True, small: bool = True, bar: bool = True) -> np.ndarray:
    frame = np.full((420, 600), METAL, np.uint8)
    if big:
        cv2.circle(frame, BIG_BORE, BIG_RADIUS, DARK, -1)
    if small:
        cv2.circle(frame, SMALL_BORE, SMALL_RADIUS, DARK, -1)
    if bar:
        cv2.rectangle(frame, *BAR, DARK, -1)
    return frame


def detect(frame: np.ndarray, **overrides):
    return BlobAnalysisDetector({**BLOB, **overrides}).detect(frame)


def near(hole, point, tolerance: float = 1.5) -> bool:
    return abs(hole.x_px - point[0]) <= tolerance and abs(hole.y_px - point[1]) <= tolerance


# ------------------------------------------------------------- the rule
def test_the_largest_round_blob_is_the_hole() -> None:
    result = detect(make_frame())
    assert len(result.holes) == 1
    assert near(result.best, BIG_BORE)
    # ~75 px, not 80: after the blur, a threshold this close to the bore's
    # own grey level crosses a few pixels inside the rim. The lab script
    # measures exactly the same.
    assert result.best.diameter_px == pytest.approx(2 * BIG_RADIUS, abs=6)


def test_a_bigger_blob_that_is_not_round_only_wins_without_the_roundness_gate() -> None:
    """The bar is the largest blob; min_circularity is what keeps it out."""
    result = detect(make_frame(), min_circularity=0.0)
    bar_centre = ((BAR[0][0] + BAR[1][0]) / 2, (BAR[0][1] + BAR[1][1]) / 2)
    assert near(result.best, bar_centre, tolerance=3)


def test_exactly_one_hole_is_reported_even_with_several_round_blobs() -> None:
    """The smaller bore is just as round — reporting it too would let the
    pipeline's highest-confidence pick choose it over the largest."""
    result = detect(make_frame(bar=False))
    assert len(result.holes) == 1
    assert near(result.best, BIG_BORE)


def test_confidence_is_the_blob_circularity() -> None:
    hole = detect(make_frame()).best
    assert hole.confidence == hole.circularity
    assert 0.8 <= hole.confidence <= 1.0


def test_a_blank_frame_is_no_hole_not_an_error() -> None:
    result = detect(np.full((200, 200), METAL, np.uint8))
    assert result.holes == []


# ---------------------------------------------------------------- gates
def test_min_area_rejects_small_blobs() -> None:
    frame = make_frame(big=False, bar=False)  # only the small bore, ~1250 px²
    assert near(detect(frame, min_area_px=500).best, SMALL_BORE)
    assert detect(frame, min_area_px=2000).holes == []


def test_the_largest_blob_is_chosen_among_those_passing_the_diameter_gates() -> None:
    """A max diameter below the big bore hands the hole to the next largest."""
    assert near(detect(make_frame(), max_hole_diameter_px=60).best, SMALL_BORE)
    assert detect(make_frame(), min_hole_diameter_px=100).holes == []


def test_the_threshold_decides_what_counts_as_dark() -> None:
    frame = np.full((300, 300), METAL, np.uint8)
    cv2.circle(frame, (150, 150), 40, 100, -1)  # a grey, not black, bore
    assert detect(frame, threshold=40).holes == []
    assert near(detect(frame, threshold=140).best, (150, 150))


def test_grayscale_and_colour_frames_agree() -> None:
    gray = make_frame()
    colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert detect(gray).holes == detect(colour).holes


def test_an_even_blur_is_rounded_up_to_odd() -> None:
    frame = make_frame()
    assert detect(frame, blur_kernel_size=6).holes == detect(frame, blur_kernel_size=7).holes


def test_an_even_morph_kernel_is_rounded_up_to_odd_so_the_blob_does_not_shift() -> None:
    """An even kernel's anchor is off-centre; open+close with one moves the
    blob. Downscale makes even kernels routinely (5 px at ÷2 is 2 px)."""
    frame = make_frame()
    even = detect(frame, morphology_kernel_size=4).best
    assert even == detect(frame, morphology_kernel_size=5).best
    assert near(even, BIG_BORE)


# ------------------------------------------------------------ debug view
def test_debug_mask_shows_every_dark_blob_not_just_the_hole() -> None:
    mask = BlobAnalysisDetector(BLOB).debug_stages(make_frame())["mask"]
    assert mask[BIG_BORE[1], BIG_BORE[0]] == 255
    assert mask[SMALL_BORE[1], SMALL_BORE[0]] == 255
    assert mask[345, 300] == 255  # the bar, rejected only by the gates
    assert mask[50, 50] == 0  # bare metal


def test_morphology_removes_specks_from_the_mask() -> None:
    frame = np.full((200, 200), METAL, np.uint8)
    frame[50, 50] = frame[120, 80] = DARK  # single-pixel specks
    raw = BlobAnalysisDetector({**BLOB, "blur_kernel_size": 1, "morphology_kernel_size": 1})
    cleaned = BlobAnalysisDetector({**BLOB, "blur_kernel_size": 1, "morphology_kernel_size": 5})
    assert raw.debug_stages(frame)["mask"].any()
    assert not cleaned.debug_stages(frame)["mask"].any()


def test_a_broken_frame_is_a_detection_error() -> None:
    with pytest.raises(DetectionError):
        detect(np.zeros((0, 0), np.uint8))


# ------------------------------------------------------------- downscale
def test_downscale_converts_the_area_by_the_factor_squared() -> None:
    scaled = BlobAnalysisDetector.downscaled_params(BLOB, 2.0)
    assert scaled["min_area_px"] == 125  # 500 / 2²
    assert scaled["blur_kernel_size"] == 4  # rounded; the detector makes it odd
    assert scaled["morphology_kernel_size"] == 2
    assert scaled["threshold"] == 40  # a grey level, not a length
    assert scaled["min_hole_diameter_px"] == 0  # disabled stays disabled
    assert BLOB["min_area_px"] == 500  # the input is not modified


# ---------------------------------------------------- engine integration
def camera_block(**common) -> dict:
    return {
        "active_detector": "blob_analysis",
        "common": {"confidence_threshold": 0.6, "expected_hole_count": 1, **common},
        "blob_analysis": dict(BLOB),
    }


def test_engine_runs_it_from_a_camera_block() -> None:
    engine = VisionEngine({"cameras": {"1": camera_block()}})
    assert engine.active_detector_name(1) == "blob_analysis"
    assert near(engine.detect(make_frame(), 1).best, BIG_BORE)


def test_engine_finds_it_in_full_resolution_pixels_when_downscaled() -> None:
    """At ÷4 the bore is ~230 px² on the shrunk frame, so an *unconverted*
    3000 px² gate would reject it; converted (÷16) it keeps it."""
    full = VisionEngine({"cameras": {"1": camera_block()}}).detect(make_frame(), 1).best
    block = camera_block(downscale_enabled=True, downscale_factor=4.0)
    block["blob_analysis"]["min_area_px"] = 3000  # the bore is ~4500 px² full size
    engine = VisionEngine({"cameras": {"1": block}})
    best = engine.detect(make_frame(), 1).best
    assert best is not None and near(best, BIG_BORE, tolerance=4)
    assert best.diameter_px == pytest.approx(full.diameter_px, abs=8)  # 2 px per side at ÷4


def test_engine_keeps_the_camera_block_verbatim() -> None:
    block = camera_block()
    engine = VisionEngine({"cameras": {"1": copy.deepcopy(block)}})
    assert engine.camera_config(1) == block

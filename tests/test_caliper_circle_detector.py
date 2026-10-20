"""caliper_based_circle_fitting: radial calipers find the rim, a robust fit gives the hole.

What the strategy is for, pinned on synthetic bores with a known sub-pixel
centre: it measures the *edge*, so it stays on the true centre where an
area-based strategy (``blob_analysis``) is pulled off it — a rim the frame
cuts off, a bright bar over part of the rim. The gates (inlier ratio, RMS,
diameter) and the viewer-style result (points, inliers, RMS, inlier ratio)
are pinned too, as is determinism: the RANSAC sampling is seeded, so a frame
always gives the same answer.
"""

import copy

import cv2
import numpy as np
import pytest

from core.utilities.exceptions import DetectionError
from core.vision import BlobAnalysisDetector, CaliperCircleDetector, VisionEngine

METAL = 190
DARK = 25
CENTRE = (280.37, 210.81)  # deliberately off the pixel grid
RADIUS = 70

CALIPER = {
    "coarse_threshold": 0,
    "blur_kernel_size": 5,
    "num_calipers": 72,
    "caliper_width_px": 5,
    "search_band": 0.5,
    "edge_polarity": "dark_to_light",
    "edge_selection": "strongest",
    "min_edge_strength": 10,
    "inlier_threshold_px": 3.0,
    "min_inlier_ratio": 0.3,
    "max_rms_px": 2.0,
    "min_hole_diameter_px": 0,
    "max_hole_diameter_px": 0,
}


def bore(centre=CENTRE, radius=RADIUS, shape=(420, 560), value=DARK, frame=None) -> np.ndarray:
    """An anti-aliased dark disc with a sub-pixel centre (fixed-point drawing)."""
    if frame is None:
        frame = np.full(shape, METAL, np.uint8)
    shift = 8
    cv2.circle(
        frame,
        (int(round(centre[0] * 2**shift)), int(round(centre[1] * 2**shift))),
        int(round(radius * 2**shift)),
        value, -1, cv2.LINE_AA, shift,
    )
    return frame


def detector(**overrides) -> CaliperCircleDetector:
    return CaliperCircleDetector({**CALIPER, **overrides})


def error(hole, centre=CENTRE) -> float:
    return float(np.hypot(hole.x_px - centre[0], hole.y_px - centre[1]))


# ------------------------------------------------------------- accuracy
def test_a_clean_bore_is_measured_to_a_fraction_of_a_pixel() -> None:
    result = detector().detect(bore())
    assert len(result.holes) == 1
    assert error(result.best) < 0.1
    assert result.best.diameter_px == pytest.approx(2 * RADIUS, abs=2)


def test_a_rim_cut_off_by_the_frame_still_gives_the_true_centre() -> None:
    """Blob analysis takes the centroid of what is visible — 9 px off here."""
    centre = (40.25, 210.6)
    frame = bore(centre=centre)
    hole = detector().detect(frame).best
    assert hole is not None and error(hole, centre) < 0.3
    blob = BlobAnalysisDetector({"threshold": 100, "min_circularity": 0.5}).detect(frame).best
    assert error(blob, centre) > 5  # what the calipers are for


def test_a_bright_bar_over_the_rim_is_rejected_as_outliers() -> None:
    frame = bore()
    cv2.rectangle(frame, (300, 120), (330, 300), 230, -1)
    fit = detector().fit_circle(frame)
    hole = detector().detect(frame).best
    assert hole is not None and error(hole) < 0.3
    assert not fit.inliers.all()  # the bar's edges were found — and thrown out
    assert fit.inlier_ratio < 1.0


def test_noise_moves_the_centre_by_far_less_than_a_pixel() -> None:
    rng = np.random.default_rng(3)
    noisy = np.clip(bore().astype(float) + rng.normal(0, 12, (420, 560)), 0, 255).astype(np.uint8)
    hole = detector().detect(noisy).best
    assert hole is not None and error(hole) < 0.2


# ------------------------------------------------------------ the result
def test_confidence_is_the_inlier_ratio_and_one_hole_is_reported() -> None:
    frame = bore()
    bore(centre=(470, 80), radius=30, frame=frame)  # a second, smaller bore
    fit = detector().fit_circle(frame)
    result = detector().detect(frame)
    assert len(result.holes) == 1 and error(result.best) < 0.1  # the larger seed wins
    assert result.best.confidence == pytest.approx(fit.inlier_ratio)
    assert result.best.circularity == 1.0  # not applicable: fitted *as* a circle


def test_the_fit_reports_points_inliers_rms_and_ratio_like_the_viewer() -> None:
    fit = detector().fit_circle(bore())
    assert fit.points.shape == (72, 2)  # every caliper found the rim
    assert fit.inliers.shape == (72,) and fit.inliers.all()
    assert fit.inlier_ratio == 1.0
    assert fit.rms < 0.2
    assert fit.circle is not None


def test_the_same_frame_always_gives_the_same_answer() -> None:
    frame = bore()
    cv2.rectangle(frame, (300, 120), (330, 300), 230, -1)
    assert detector().detect(frame).holes == detector().detect(frame).holes


def test_a_blank_frame_is_no_hole_not_an_error() -> None:
    blank = np.full((300, 300), METAL, np.uint8)
    assert detector().detect(blank).holes == []
    assert detector().fit_circle(blank) is None or detector().fit_circle(blank).circle is None


# ---------------------------------------------------------------- gates
def test_too_little_visible_rim_fails_the_inlier_ratio_gate() -> None:
    """Most of the rim off-frame: the fit is fine, there is just too little of it."""
    centre = (-50.0, 210.0)  # ~25 % of a 70 px bore's rim is in the frame (x > 0)
    frame = bore(centre=centre)
    fit = detector().fit_circle(frame)
    assert fit is not None and fit.circle is not None and fit.inlier_ratio < 0.3
    assert detector().detect(frame).holes == []
    assert len(detector(min_inlier_ratio=0.05).detect(frame).holes) == 1


def test_a_ragged_rim_fails_a_tight_rms_gate() -> None:
    frame = np.full((420, 560), METAL, np.uint8)
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    radii = RADIUS + np.where(np.arange(40) % 2 == 0, 2.5, -2.5)  # a 2.5 px zig-zag rim
    polygon = np.column_stack([280 + radii * np.cos(angles), 210 + radii * np.sin(angles)])
    cv2.fillPoly(frame, [polygon.round().astype(np.int32)], DARK)
    assert len(detector().detect(frame).holes) == 1  # within 2 px RMS
    assert detector(max_rms_px=0.3).detect(frame).holes == []


def test_the_diameter_gates_apply_to_the_fitted_circle() -> None:
    frame = bore()
    assert detector(max_hole_diameter_px=120).detect(frame).holes == []
    assert detector(min_hole_diameter_px=160).detect(frame).holes == []
    assert len(detector(min_hole_diameter_px=120, max_hole_diameter_px=160).detect(frame).holes) == 1


def test_a_fixed_seed_threshold_that_misses_the_bore_finds_nothing() -> None:
    assert detector(coarse_threshold=10).detect(bore()).holes == []  # bore is 25 grey
    assert len(detector(coarse_threshold=100).detect(bore()).holes) == 1


# ------------------------------------------------------- edge selection
def test_first_takes_the_bore_and_strongest_the_brighter_ring_round_it() -> None:
    """A dark bore inside a mid-grey chamfer ring on bright metal: two rising
    edges going outward, the outer one stronger."""
    frame = bore(radius=90, value=110, frame=np.full((420, 560), 240, np.uint8))
    bore(radius=60, value=60, frame=frame)
    strongest = detector(edge_selection="strongest", coarse_threshold=150).detect(frame).best
    first = detector(edge_selection="first", coarse_threshold=150).detect(frame).best
    assert strongest.diameter_px == pytest.approx(180, abs=3)
    assert first.diameter_px == pytest.approx(120, abs=3)


def test_unknown_option_values_fail_at_detect_time_not_at_build_time() -> None:
    """Built fine (the station must boot), fails when run — CLAUDE.md §9."""
    built = detector(edge_polarity="sideways")
    with pytest.raises(DetectionError, match="edge_polarity"):
        built.detect(bore())
    with pytest.raises(DetectionError, match="edge_selection"):
        detector(edge_selection="last").detect(bore())


# ------------------------------------------------------------ debug view
def test_debug_shows_the_seed_mask_and_the_caliper_points() -> None:
    frame = bore()
    stages = detector().debug_stages(frame)
    assert set(stages) == {"mask", "edges"}
    assert stages["mask"].shape == stages["edges"].shape == frame.shape
    assert stages["mask"][int(CENTRE[1]), int(CENTRE[0])] == 255  # the seed blob
    rim = (int(round(CENTRE[0] + RADIUS)), int(round(CENTRE[1])))  # caliper at 0°
    assert stages["edges"][rim[1] - 1 : rim[1] + 2, rim[0] - 1 : rim[0] + 2].any()
    assert stages["edges"][int(CENTRE[1]), int(CENTRE[0])] == 0  # nothing at the centre


def test_debug_edges_are_blank_without_a_seed() -> None:
    stages = detector().debug_stages(np.full((200, 200), METAL, np.uint8))
    assert not stages["edges"].any()


# ------------------------------------------------------------- downscale
def test_downscale_converts_every_pixel_length() -> None:
    scaled = CaliperCircleDetector.downscaled_params(CALIPER, 2.0)
    assert scaled["inlier_threshold_px"] == 1.5
    assert scaled["max_rms_px"] == 1.0
    assert scaled["caliper_width_px"] == 2  # rounded, never below 1
    assert scaled["blur_kernel_size"] == 2  # rounded; the detector makes it odd
    assert scaled["min_edge_strength"] == 10  # grey levels per pixel, left alone
    assert scaled["num_calipers"] == 72  # a count
    assert scaled["coarse_threshold"] == 0
    assert CALIPER["inlier_threshold_px"] == 3.0  # the input is not modified


# ---------------------------------------------------- engine integration
def camera_block(**common) -> dict:
    return {
        "active_detector": "caliper_based_circle_fitting",
        "common": {"confidence_threshold": 0.6, "expected_hole_count": 1, **common},
        "caliper_based_circle_fitting": dict(CALIPER),
    }


def test_engine_runs_it_from_a_camera_block() -> None:
    engine = VisionEngine({"cameras": {"1": camera_block()}})
    assert engine.active_detector_name(1) == "caliper_based_circle_fitting"
    assert error(engine.detect(bore(), 1).best) < 0.1


def test_engine_downscaled_result_is_in_full_resolution_pixels() -> None:
    engine = VisionEngine({"cameras": {"1": camera_block(downscale_enabled=True, downscale_factor=3.0)}})
    hole = engine.detect(bore(), 1).best
    assert hole is not None and error(hole) < 1.0
    assert hole.diameter_px == pytest.approx(2 * RADIUS, abs=3)


def test_engine_keeps_the_camera_block_verbatim() -> None:
    block = camera_block()
    engine = VisionEngine({"cameras": {"1": copy.deepcopy(block)}})
    assert engine.camera_config(1) == block

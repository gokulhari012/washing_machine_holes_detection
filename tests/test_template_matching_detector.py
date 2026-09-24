"""template_matching strategy: multi-scale search, cross-scale NMS, gates.

The scenes carry bores of deliberately *different* sizes, because the whole
point of the scale pyramid is that one template cropped at one standoff still
finds a bore photographed at another — and the whole point of the cross-scale
suppression is that it then reports that bore once, not once per scale that
happened to fit it.
"""

import cv2
import numpy as np
import pytest

from core.utilities.exceptions import DetectionError
from core.vision.template_matching_detector import (
    TemplateMatchingDetector,
    parse_scales,
)

PLATE = 200  # sheet-metal grey
BORE = 30  # bore grey
TEMPLATE_RADIUS = 40
TEMPLATE_SIDE = 100


def make_template(tmp_path) -> str:
    """A tight crop of a bore at its authored size, written to disk."""
    template = np.full((TEMPLATE_SIDE, TEMPLATE_SIDE), PLATE, np.uint8)
    cv2.circle(template, (TEMPLATE_SIDE // 2, TEMPLATE_SIDE // 2), TEMPLATE_RADIUS, BORE, -1)
    path = tmp_path / "template.png"
    cv2.imwrite(str(path), template)
    return str(path)


def make_frame(bores: list[tuple[int, int, int]]) -> np.ndarray:
    """A plate carrying ``(x, y, radius)`` bores."""
    frame = np.full((600, 900), PLATE, np.uint8)
    for x, y, radius in bores:
        cv2.circle(frame, (x, y), radius, BORE, -1)
    return frame


def detector(tmp_path, **params) -> TemplateMatchingDetector:
    return TemplateMatchingDetector(
        {"template_path": make_template(tmp_path), "match_threshold": 0.8, **params}
    )


# --------------------------------------------------------------- scales
@pytest.mark.parametrize(
    "raw, expected",
    [
        ([0.5, 1.0, 1.5], [0.5, 1.0, 1.5]),
        ("0.5, 1.0 ,1.5", [0.5, 1.0, 1.5]),
        ("0.8;1.2", [0.8, 1.2]),
        ([1.2, 1.2, 0.8], [0.8, 1.2]),  # deduplicated and sorted
        ([0, -1, 2.0], [2.0]),  # non-positive dropped
        (1.5, [1.5]),  # a bare number
        ("", [1.0]),
        (None, [1.0]),
        ("nonsense", [1.0]),  # a typo must not take the detector out of service
        ([], [1.0]),
    ],
)
def test_scales_parse_forgivingly(raw, expected) -> None:
    assert parse_scales(raw) == expected


def test_scales_are_capped() -> None:
    assert len(parse_scales([0.01 * n for n in range(1, 200)])) == 25


# ------------------------------------------------------- multi-scale find
def test_single_scale_finds_only_the_matching_size(tmp_path) -> None:
    frame = make_frame([(200, 300, 40), (500, 300, 60), (780, 300, 20)])
    holes = detector(tmp_path, scales=[1.0]).detect(frame).holes
    assert len(holes) == 1
    assert abs(holes[0].x_px - 200) <= 2


def test_scale_pyramid_finds_every_size_once(tmp_path) -> None:
    """One bore per scale, and exactly one hole each — the cross-scale NMS."""
    frame = make_frame([(200, 300, 40), (500, 300, 60), (780, 300, 20)])
    result = detector(tmp_path, scales=[0.5, 0.75, 1.0, 1.25, 1.5]).detect(frame)
    assert len(result.holes) == 3
    found = sorted((hole.x_px, hole.diameter_px) for hole in result.holes)
    for (x_px, diameter), (expected_x, expected_d) in zip(
        found, [(200, 100), (500, 150), (780, 50)]
    ):
        assert abs(x_px - expected_x) <= 3
        assert abs(diameter - expected_d) <= 6


def test_one_bore_at_many_scales_is_one_hole(tmp_path) -> None:
    """Neighbouring scales all match the same bore; only the best survives."""
    frame = make_frame([(450, 300, 40)])
    scales = [0.9, 0.95, 1.0, 1.05, 1.1]
    result = detector(tmp_path, scales=scales).detect(frame)
    assert len(result.holes) == 1
    assert abs(result.best.x_px - 450) <= 3


def test_holes_are_sorted_best_first(tmp_path) -> None:
    frame = make_frame([(200, 300, 40), (500, 300, 40)])
    holes = detector(tmp_path, scales=[1.0]).detect(frame).holes
    assert len(holes) == 2
    assert holes[0].confidence >= holes[1].confidence


def test_max_matches_caps_the_result(tmp_path) -> None:
    frame = make_frame([(150 * n, 300, 40) for n in range(1, 6)])
    holes = detector(tmp_path, scales=[1.0], max_matches=2).detect(frame).holes
    assert len(holes) == 2


# --------------------------------------------------------------- gates
def test_no_bore_is_empty_not_error(tmp_path) -> None:
    result = detector(tmp_path, scales=[1.0]).detect(make_frame([]))
    assert result.holes == []
    assert result.found is False


def test_size_gate_rejects_out_of_range_matches(tmp_path) -> None:
    frame = make_frame([(200, 300, 40), (500, 300, 60), (780, 300, 20)])
    scales = [0.5, 1.0, 1.5]
    holes = detector(
        tmp_path, scales=scales, min_hole_diameter_px=60, max_hole_diameter_px=110
    ).detect(frame).holes
    assert [round(hole.diameter_px) for hole in holes] == [100]


def test_size_gate_off_by_default(tmp_path) -> None:
    frame = make_frame([(200, 300, 40), (500, 300, 60), (780, 300, 20)])
    holes = detector(tmp_path, scales=[0.5, 1.0, 1.5]).detect(frame).holes
    assert len(holes) == 3


def test_a_gated_match_does_not_consume_a_result_slot(tmp_path) -> None:
    """The oversized bore scores highest; capping at one match must still
    return the bore that passes the gate, not an empty result."""
    frame = make_frame([(500, 300, 60), (200, 300, 40)])
    holes = detector(
        tmp_path,
        scales=[1.0, 1.5],
        max_matches=1,
        min_hole_diameter_px=0,
        max_hole_diameter_px=110,
    ).detect(frame).holes
    assert len(holes) == 1
    assert abs(holes[0].x_px - 200) <= 3


def test_inverted_size_gate_raises(tmp_path) -> None:
    with pytest.raises(DetectionError, match="max_hole_diameter_px"):
        detector(
            tmp_path, min_hole_diameter_px=200, max_hole_diameter_px=100
        ).detect(make_frame([]))


def test_confidence_stays_within_zero_and_one(tmp_path) -> None:
    """TM_CCOEFF_NORMED scores -1..1; Hole.confidence is documented 0..1."""
    noise = np.random.default_rng(0).integers(0, 255, (300, 300), dtype=np.uint8)
    holes = detector(tmp_path, scales=[1.0], match_threshold=-5.0).detect(noise).holes
    assert holes
    assert all(0.0 <= hole.confidence <= 1.0 for hole in holes)


# ------------------------------------------------------- configuration
def test_missing_template_path_raises(tmp_path) -> None:
    with pytest.raises(DetectionError, match="template_path"):
        TemplateMatchingDetector({"template_path": ""}).detect(make_frame([]))


def test_unreadable_template_raises_at_configure(tmp_path) -> None:
    with pytest.raises(DetectionError, match="Cannot read template"):
        TemplateMatchingDetector({"template_path": str(tmp_path / "nope.png")})


def test_unknown_method_raises(tmp_path) -> None:
    with pytest.raises(DetectionError, match="method"):
        detector(tmp_path, method="TM_NONSENSE").detect(make_frame([]))


@pytest.mark.parametrize(
    "method", ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"]
)
def test_every_method_finds_the_bore(tmp_path, method) -> None:
    """SQDIFF is inverted internally so one 'higher is better' comparison
    serves all three — a regression here would silently return nothing."""
    frame = make_frame([(450, 300, 40)])
    holes = detector(tmp_path, scales=[1.0], method=method, match_threshold=0.8).detect(
        frame
    ).holes
    assert holes
    assert abs(holes[0].x_px - 450) <= 3


def test_template_bigger_than_frame_at_every_scale_raises(tmp_path) -> None:
    with pytest.raises(DetectionError, match="smaller than template"):
        detector(tmp_path, scales=[1.0]).detect(np.full((50, 50), PLATE, np.uint8))


def test_a_scale_that_does_not_fit_is_skipped_not_fatal(tmp_path) -> None:
    """A 1.5x template exceeds this frame; the 0.5x one still finds the bore."""
    frame = np.full((120, 120), PLATE, np.uint8)
    cv2.circle(frame, (60, 60), 20, BORE, -1)
    holes = detector(tmp_path, scales=[0.5, 1.5]).detect(frame).holes
    assert holes
    assert abs(holes[0].x_px - 60) <= 3


def test_template_is_reread_when_the_file_changes(tmp_path) -> None:
    """Re-cropping a template under the same filename must take effect."""
    path = make_template(tmp_path)
    instance = TemplateMatchingDetector(
        {"template_path": path, "match_threshold": 0.8, "scales": [1.0]}
    )
    assert instance.detect(make_frame([(450, 300, 40)])).holes

    replacement = np.full((60, 60), PLATE, np.uint8)
    cv2.circle(replacement, (30, 30), 20, BORE, -1)
    cv2.imwrite(path, replacement)
    instance.configure({"template_path": path})
    holes = instance.detect(make_frame([(450, 300, 20)])).holes
    assert holes
    assert abs(holes[0].diameter_px - 60) <= 4


def test_reconfiguring_unrelated_params_keeps_the_template(tmp_path) -> None:
    """What makes the Auto Sweep affordable: re-tuning the threshold must not
    re-read the template file thousands of times."""
    instance = detector(tmp_path, scales=[1.0])
    reads = []
    original = cv2.imread

    def counting_imread(*args, **kwargs):
        reads.append(args[0])
        return original(*args, **kwargs)

    cv2.imread = counting_imread
    try:
        for threshold in (0.5, 0.6, 0.7, 0.8):
            instance.configure({"match_threshold": threshold})
    finally:
        cv2.imread = original
    assert reads == []


# ------------------------------------------------------------ debug view
def test_debug_mask_is_frame_sized_and_centred_on_the_bore(tmp_path) -> None:
    """The score map is smaller than the frame and indexed by the template's
    top-left corner; the debug mask must be re-based onto the bore itself or
    the overlay draws the response up and to the left of what it matched."""
    frame = make_frame([(450, 300, 40)])
    mask = detector(tmp_path, scales=[1.0]).debug_stages(frame)["mask"]
    assert mask.shape == frame.shape
    assert mask.dtype == np.uint8
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1
    (x, y), _radius = cv2.minEnclosingCircle(contours[0])
    assert abs(x - 450) <= 4 and abs(y - 300) <= 4


def test_debug_mask_is_empty_on_a_bare_plate(tmp_path) -> None:
    mask = detector(tmp_path, scales=[1.0]).debug_stages(make_frame([]))["mask"]
    assert not mask.any()


def test_debug_mask_covers_every_scale(tmp_path) -> None:
    frame = make_frame([(200, 300, 40), (500, 300, 60), (780, 300, 20)])
    mask = detector(tmp_path, scales=[0.5, 1.0, 1.5]).debug_stages(frame)["mask"]
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 3


def test_colour_frames_are_accepted(tmp_path) -> None:
    frame = cv2.cvtColor(make_frame([(450, 300, 40)]), cv2.COLOR_GRAY2BGR)
    holes = detector(tmp_path, scales=[1.0]).detect(frame).holes
    assert holes
    assert abs(holes[0].x_px - 450) <= 3

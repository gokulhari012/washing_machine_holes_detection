"""Caliper-based circle fitting: radial calipers find the rim, a robust fit gives the hole.

The concept of the lab viewer
``development_files/testing_own/holes_detection_center_robust_caliper-based
circle fitting.py``. That script only *displays* the result; the algorithm it
imports (``hole_center_robust.find_hole_center``) was never part of this
repository, so this module implements the technique to the contract the
viewer shows: a fitted centre and radius, the caliper edge points with an
inlier flag each, the fit RMS, and the inlier ratio, judged OK at RMS <= 2 px
and >= 30 % inliers (the viewer's ``MAX_RMS`` / ``MIN_INLIERS``, which are
this strategy's defaults).

Why calipers
------------
Blob strategies measure a hole by the *area* a threshold leaves, so the centre
moves with the threshold, with a shadow on one side of the bore, or with a
rim the ROI cuts off. A caliper measures the *edge* itself: it samples a short
line across the rim, finds where the grey level steps, and places that step
to a fraction of a pixel. Fitting a circle to many such points — rejecting the
ones a burr, a scratch or a reflection put off the circle — gives a centre
that depends only on the part of the rim that really is a circle.

Pipeline
--------
1. grayscale -> Gaussian blur (``blur_kernel_size``, made odd).
2. **Seed** — a rough centre and radius to aim the calipers with: an inverse
   threshold (``coarse_threshold``; 0 = Otsu, picked per image), a small
   opening, then the largest dark blob that is at all round (circularity
   >= 0.3 — loose on purpose, a partly hidden bore is still a fine seed).
   Its centroid and equivalent radius are the seed.
3. **Calipers** — ``num_calipers`` rays, evenly spaced around the seed, each
   sampled at 1 px steps from ``(1 - search_band)`` to ``(1 + search_band)``
   times the seed radius. Each caliper is ``caliper_width_px`` wide: the
   samples across its width are averaged into one profile, which is what
   makes a caliper steadier than a single ray of pixels.
4. **Edge per caliper** — the profile's derivative; with ``edge_polarity``
   ``dark_to_light`` only rising steps (dark bore -> bright metal, going
   outward) count, with ``either`` both directions do. Of the peaks at least
   ``min_edge_strength`` (grey levels per pixel) high, ``edge_selection``
   takes the ``strongest`` or the ``first`` (innermost — for a bore with a
   brighter chamfer ring round it). A parabola through the peak gives the
   sub-pixel position. A caliper with no such peak contributes no point.
5. **Robust fit** — RANSAC over circles through three edge points (plus the
   seed circle itself), keeping the hypothesis that puts the most points
   within ``inlier_threshold_px`` of its rim; then a geometric least-squares
   (Gauss-Newton) refit on those inliers, twice, re-selecting inliers in
   between. The random sampling is seeded, so the same frame always gives the
   same answer.
6. **Refinement pass** — steps 3-5 again, cast from the fitted centre with
   half the search band. The first pass's calipers came from the seed, which
   can sit well off the true centre (the centroid of a partly visible bore
   does); from the fitted centre every caliper crosses the rim square-on, so
   the edges sharpen and the inlier ratio becomes the true share of the rim
   that is visible.

Gates and the result
--------------------
The fit is a hole only when the inlier ratio (inliers / calipers cast — not /
edges found, so rays that found nothing count against it) is at least
``min_inlier_ratio``, the RMS of the inliers' distance to the rim is at most
``max_rms_px``, and the fitted diameter passes ``min/max_hole_diameter_px``
(0 = off). Otherwise the result is empty — NG, not an error.

Like ``blob_analysis`` this reports **one hole per image**: the circle fitted
round the seed. ``confidence`` is the inlier ratio, so
``common.confidence_threshold`` acts as a second inlier-ratio floor — on a
camera that only ever sees part of the bore, lower both. ``circularity`` is
1.0 (not applicable: the rim was fitted *as* a circle; RMS is what says how
well).

``inlier_threshold_px`` must stay above ``max_rms_px`` for the RMS gate to
mean anything: the RMS is taken over the inliers, which by definition lie
within the inlier threshold. The defaults (3.0 / 2.0) keep it that way.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

import cv2
import numpy as np

from core.utilities.exceptions import DetectionError
from core.vision.detection_result import DetectionResult, Hole
from core.vision.detector_base import HoleDetector, shrink_kernel, shrink_length

POLARITIES = ("dark_to_light", "either")
SELECTIONS = ("strongest", "first")

#: Pixel-sized parameters beyond the diameter gates — see downscaled_params.
_KERNEL_KEYS = ("blur_kernel_size", "caliper_width_px")
_LENGTH_KEYS = ("inlier_threshold_px", "max_rms_px")

_SEED_KERNEL = 3  # opening that strips specks off the seed mask
_SEED_MIN_CIRCULARITY = 0.3
_SEED_MIN_AREA_PX = 12.0
_RANSAC_SAMPLES = 300
_RANSAC_SEED = 0
_REFIT_ROUNDS = 2
_GAUSS_NEWTON_STEPS = 8
_MIN_FIT_POINTS = 3


def _odd(value: Any) -> int:
    """Kernel sizes must be odd and >= 1."""
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


@dataclass(frozen=True)
class CaliperFit:
    """One caliper pass: where it searched, what it found, how well it fitted.

    ``circle`` is ``(cx, cy, radius)`` or ``None`` when no circle could be
    fitted. ``points`` are the edge points (``(N, 2)``, pixels of the analysed
    image) and ``inliers`` a matching ``(N,)`` bool array. ``origin`` and
    ``band`` are the centre the calipers were cast from and their inner/outer
    radius — what the debug view draws.
    """

    origin: tuple[float, float]
    band: tuple[float, float]
    calipers: int
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    circle: tuple[float, float, float] | None = None
    rms: float = math.inf
    inlier_ratio: float = 0.0


class CaliperCircleDetector(HoleDetector):
    """Seed blob -> radial calipers -> RANSAC + least-squares circle."""

    name: ClassVar[str] = "caliper_based_circle_fitting"
    thread_safe: ClassVar[bool] = True  # stateless per call; the RNG is local

    @classmethod
    def downscaled_params(cls, params: dict[str, Any], factor: float) -> dict[str, Any]:
        """Diameter gates, the blur and caliper widths, and the two fit
        tolerances — all pixel lengths. ``min_edge_strength`` is left alone:
        it is grey levels per pixel, and a shrunk edge is *sharper*, so it
        only ever reads as at least as strong."""
        scaled = super().downscaled_params(params, factor)
        for key in _KERNEL_KEYS:
            if key in scaled:
                scaled[key] = shrink_kernel(scaled[key], factor)
        for key in _LENGTH_KEYS:
            if key in scaled:
                scaled[key] = shrink_length(scaled[key], factor)
        return scaled

    # ---------------------------------------------------------- strategy
    def detect(self, image: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        _mask, fit = self._analyse(image)
        holes: list[Hole] = []
        if fit is not None and fit.circle is not None and self._accepts(fit):
            cx, cy, radius = fit.circle
            holes.append(
                Hole(
                    x_px=float(cx),
                    y_px=float(cy),
                    diameter_px=float(2.0 * radius),
                    circularity=1.0,
                    confidence=float(min(1.0, fit.inlier_ratio)),
                )
            )
        return DetectionResult(
            holes=holes, processing_ms=(time.perf_counter() - started) * 1000.0
        )

    def debug_stages(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """``mask``: the seed threshold (every blob the seed was chosen from).
        ``edges``: the final pass's search band (two thin circles) and its edge
        points — filled dots for inliers, rings for the outliers the fit
        rejected (the lab viewer's green / orange points)."""
        mask, fit = self._analyse(image)
        edges = np.zeros(mask.shape, np.uint8)
        if fit is not None:
            height, width = mask.shape
            stroke = max(1, round(min(height, width) / 500))
            dot = max(2, round(min(height, width) / 250))
            origin = (int(round(fit.origin[0])), int(round(fit.origin[1])))
            for radius in fit.band:
                cv2.circle(edges, origin, int(round(radius)), 255, stroke)
            for (x, y), inlier in zip(fit.points, fit.inliers):
                cv2.circle(
                    edges, (int(round(x)), int(round(y))), dot, 255, -1 if inlier else stroke
                )
        return {"mask": mask, "edges": edges}

    def fit_circle(self, image: np.ndarray) -> CaliperFit | None:
        """The final caliper pass, *before* the acceptance gates — the
        viewer-style full answer (points, inliers, RMS, inlier ratio).
        ``None`` when no seed blob was found at all."""
        return self._analyse(image)[1]

    # ------------------------------------------------------------ internal
    def _settings(self) -> dict[str, Any]:
        """The parameters, validated. Raised at detect time, not configure
        time — the station must boot with a half-edited block (CLAUDE.md §9)."""
        params = self._params
        polarity = str(params.get("edge_polarity", "dark_to_light")).lower()
        if polarity not in POLARITIES:
            raise DetectionError(f"Unknown edge_polarity: {polarity!r}")
        selection = str(params.get("edge_selection", "strongest")).lower()
        if selection not in SELECTIONS:
            raise DetectionError(f"Unknown edge_selection: {selection!r}")
        return {
            "coarse_threshold": int(params.get("coarse_threshold", 0) or 0),
            "blur": _odd(params.get("blur_kernel_size", 5)),
            "calipers": max(8, int(params.get("num_calipers", 72))),
            "width": max(1, int(params.get("caliper_width_px", 5))),
            "band": float(np.clip(float(params.get("search_band", 0.5)), 0.05, 0.95)),
            "polarity": polarity,
            "selection": selection,
            "strength": float(params.get("min_edge_strength", 10)),
            "tolerance": max(0.05, float(params.get("inlier_threshold_px", 3.0))),
            "min_diameter": float(params.get("min_hole_diameter_px", 0) or 0),
            "max_diameter": float(params.get("max_hole_diameter_px", 0) or 0),
        }

    def _accepts(self, fit: CaliperFit) -> bool:
        params = self._params
        if fit.inlier_ratio < float(params.get("min_inlier_ratio", 0.3)):
            return False
        if fit.rms > float(params.get("max_rms_px", 2.0)):
            return False
        diameter = 2.0 * fit.circle[2]
        min_diameter = float(params.get("min_hole_diameter_px", 0) or 0)
        max_diameter = float(params.get("max_hole_diameter_px", 0) or 0)
        if min_diameter > 0 and diameter < min_diameter:
            return False
        return not (max_diameter > 0 and diameter > max_diameter)

    def _analyse(self, image: np.ndarray) -> tuple[np.ndarray, CaliperFit | None]:
        """``(seed mask, final caliper pass or None)``."""
        settings = self._settings()
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            if settings["blur"] > 1:
                gray = cv2.GaussianBlur(gray, (settings["blur"],) * 2, 0)
            mask = _seed_mask(gray, settings["coarse_threshold"])
            seed = _seed(mask, settings["min_diameter"], settings["max_diameter"])
            if seed is None:
                return mask, None
            fit = _caliper_pass(gray, seed, settings["band"], settings)
            if fit.circle is not None:
                refined = _caliper_pass(gray, fit.circle, settings["band"] / 2.0, settings)
                if refined.circle is not None:
                    fit = refined
            return mask, fit
        except cv2.error as exc:
            raise DetectionError(f"Caliper circle fitting failed: {exc}") from exc


# --------------------------------------------------------------------- seed
def _seed_mask(gray: np.ndarray, threshold: int) -> np.ndarray:
    if threshold <= 0:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((_SEED_KERNEL, _SEED_KERNEL), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _seed(
    mask: np.ndarray, min_diameter: float, max_diameter: float
) -> tuple[float, float, float] | None:
    """Centroid and equivalent radius of the largest roughly round dark blob.

    The diameter gates apply loosely: a partly hidden bore's blob is smaller
    than the bore, so the seed may be down to half the minimum diameter; it
    can never be larger than the whole bore, so the maximum applies as-is.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, float, float] | None = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < _SEED_MIN_AREA_PX or area <= best_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0 or 4.0 * math.pi * area / (perimeter * perimeter) < _SEED_MIN_CIRCULARITY:
            continue
        diameter = 2.0 * math.sqrt(area / math.pi)
        if min_diameter > 0 and diameter < 0.5 * min_diameter:
            continue
        if max_diameter > 0 and diameter > max_diameter:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        best = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"], diameter / 2.0)
        best_area = area
    return best


# ----------------------------------------------------------------- calipers
def _caliper_pass(
    gray: np.ndarray,
    around: tuple[float, float, float],
    band: float,
    settings: dict[str, Any],
) -> CaliperFit:
    """Cast the calipers round *around* ``(cx, cy, r)`` and fit a circle."""
    cx, cy, radius = around
    calipers = settings["calipers"]
    inner = max(1.0, radius * (1.0 - band))
    outer = max(inner + 3.0, radius * (1.0 + band))
    empty = CaliperFit(origin=(cx, cy), band=(inner, outer), calipers=calipers)
    points = _edge_points(gray, cx, cy, inner, outer, settings)
    if len(points) < _MIN_FIT_POINTS:
        return CaliperFit(origin=(cx, cy), band=(inner, outer), calipers=calipers, points=points,
                          inliers=np.zeros(len(points), bool))

    tolerance = settings["tolerance"]
    circle = _ransac(points, around, tolerance, inner, outer)
    if circle is None:
        return empty
    for _ in range(_REFIT_ROUNDS):
        inliers = _residuals(points, circle) <= tolerance
        if inliers.sum() < _MIN_FIT_POINTS:
            return empty
        circle = _refine(points[inliers], circle)
        if circle is None:
            return empty
    residuals = _residuals(points, circle)
    inliers = residuals <= tolerance
    count = int(inliers.sum())
    if count < _MIN_FIT_POINTS:
        return empty
    return CaliperFit(
        origin=(cx, cy),
        band=(inner, outer),
        calipers=calipers,
        points=points,
        inliers=inliers,
        circle=circle,
        rms=float(np.sqrt(np.mean(residuals[inliers] ** 2))),
        inlier_ratio=count / calipers,
    )


def _edge_points(
    gray: np.ndarray, cx: float, cy: float, inner: float, outer: float, settings: dict[str, Any]
) -> np.ndarray:
    """One sub-pixel edge point per caliper that found an edge, ``(N, 2)``."""
    calipers, width = settings["calipers"], settings["width"]
    radii = np.arange(inner, outer + 1e-9, 1.0)
    if len(radii) < 3:
        return np.zeros((0, 2))
    angles = np.arange(calipers) * (2.0 * math.pi / calipers)
    ux, uy = np.cos(angles), np.sin(angles)
    across = np.arange(width) - (width - 1) / 2.0  # samples across the caliper's width

    # (calipers, width, radii): along the ray by radius, across it by offset
    xs = cx + ux[:, None, None] * radii[None, None, :] - uy[:, None, None] * across[None, :, None]
    ys = cy + uy[:, None, None] * radii[None, None, :] + ux[:, None, None] * across[None, :, None]
    sampled = cv2.remap(
        gray,
        xs.reshape(calipers * width, -1).astype(np.float32),
        ys.reshape(calipers * width, -1).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    profiles = sampled.reshape(calipers, width, -1).astype(np.float32).mean(axis=1)
    gradient = np.gradient(profiles, axis=1)
    if settings["polarity"] == "either":
        gradient = np.abs(gradient)

    peaks = np.zeros(gradient.shape, bool)
    middle = gradient[:, 1:-1]
    peaks[:, 1:-1] = (
        (middle >= gradient[:, :-2]) & (middle > gradient[:, 2:]) & (middle >= settings["strength"])
    )
    found = np.nonzero(peaks.any(axis=1))[0]
    if not len(found):
        return np.zeros((0, 2))
    if settings["selection"] == "first":
        index = np.argmax(peaks, axis=1)[found]
    else:
        index = np.argmax(np.where(peaks, gradient, -np.inf), axis=1)[found]

    before = gradient[found, index - 1]
    at = gradient[found, index]
    after = gradient[found, index + 1]
    curvature = before - 2.0 * at + after
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(np.abs(curvature) > 1e-9, 0.5 * (before - after) / curvature, 0.0)
    distance = radii[0] + index + np.clip(shift, -0.5, 0.5)
    points = np.column_stack([cx + ux[found] * distance, cy + uy[found] * distance])

    height, width_px = gray.shape[:2]  # an edge "found" on replicated border is not a rim
    inside = (
        (points[:, 0] >= 0) & (points[:, 0] <= width_px - 1)
        & (points[:, 1] >= 0) & (points[:, 1] <= height - 1)
    )
    return points[inside]


# --------------------------------------------------------------- circle fit
def _residuals(points: np.ndarray, circle: tuple[float, float, float]) -> np.ndarray:
    cx, cy, radius = circle
    return np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius)


def _ransac(
    points: np.ndarray,
    guess: tuple[float, float, float],
    tolerance: float,
    min_radius: float,
    max_radius: float,
) -> tuple[float, float, float] | None:
    """The circle through three edge points (or *guess* itself) with the most
    points within *tolerance*; ties go to the smaller total residual. Circles
    outside the search band's radii are not hypotheses — three nearly
    collinear points otherwise propose a huge circle that "fits" a straight
    edge."""
    rng = np.random.default_rng(_RANSAC_SEED)
    count = len(points)
    triplets = rng.random((_RANSAC_SAMPLES, count)).argsort(axis=1)[:, :3]
    a, b, c = points[triplets[:, 0]], points[triplets[:, 1]], points[triplets[:, 2]]
    ax, ay, bx, by, qx, qy = a[:, 0], a[:, 1], b[:, 0], b[:, 1], c[:, 0], c[:, 1]
    d = 2.0 * (ax * (by - qy) + bx * (qy - ay) + qx * (ay - by))
    with np.errstate(divide="ignore", invalid="ignore"):
        a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, qx * qx + qy * qy
        centre_x = (a2 * (by - qy) + b2 * (qy - ay) + c2 * (ay - by)) / d
        centre_y = (a2 * (qx - bx) + b2 * (ax - qx) + c2 * (bx - ax)) / d
        radius = np.hypot(ax - centre_x, ay - centre_y)
    centre_x = np.append(centre_x, guess[0])
    centre_y = np.append(centre_y, guess[1])
    radius = np.append(radius, guess[2])
    valid = (
        np.isfinite(centre_x) & np.isfinite(centre_y) & np.isfinite(radius)
        & (radius >= min_radius) & (radius <= max_radius)
    )
    if not valid.any():
        return None
    centre_x, centre_y, radius = centre_x[valid], centre_y[valid], radius[valid]
    residuals = np.abs(
        np.hypot(points[None, :, 0] - centre_x[:, None], points[None, :, 1] - centre_y[:, None])
        - radius[:, None]
    )
    within = residuals <= tolerance
    support = within.sum(axis=1)
    spread = np.where(within, residuals, 0.0).sum(axis=1)
    best = np.lexsort((spread, -support))[0]
    return float(centre_x[best]), float(centre_y[best]), float(radius[best])


def _refine(
    points: np.ndarray, guess: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    """Geometric least squares (Gauss-Newton on point-to-rim distance) from
    *guess*. Geometric, not algebraic: an algebraic fit is biased toward a
    smaller circle on a short arc, which is exactly the partly visible bore."""
    cx, cy, radius = guess
    x, y = points[:, 0], points[:, 1]
    for _ in range(_GAUSS_NEWTON_STEPS):
        dx, dy = x - cx, y - cy
        distance = np.maximum(np.hypot(dx, dy), 1e-9)
        jacobian = np.column_stack([-dx / distance, -dy / distance, -np.ones_like(distance)])
        step = np.linalg.lstsq(jacobian, radius - distance, rcond=None)[0]
        cx, cy, radius = cx + step[0], cy + step[1], radius + step[2]
        if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(radius)) or radius <= 0:
            return None
        if float(np.abs(step).max()) < 1e-4:
            break
    return float(cx), float(cy), float(radius)

"""Local-contrast dark-hole detector (``dark_hole``).

Written for the production images: a small dark bore on bright, strongly
textured, unevenly lit sheet metal — and a bore that is often only *partly*
visible, because its rim runs behind a slot edge or the part sits slightly
off position. What is left in that case is an arc, not a disc.

Why the classical ``opencv`` strategy misses those
--------------------------------------------------
* a **global** threshold cannot separate the bore from the large shadowed
  areas and specular bands elsewhere in the frame — one exposure change and
  either everything or nothing passes;
* ``RETR_EXTERNAL`` discards the bore whenever it lies inside a larger dark
  region (it is then an *inner* contour, never reported);
* the ``4πA/P²`` roundness gate assumes a whole disc: a half-visible bore
  scores around 0.5 and is rejected before it is ever scored, and on a small
  ragged blob the perimeter term is noisy even for a complete hole.

Pipeline
--------
1. **Channel** — under a red/IR ring light the green channel dominates
   ``BGR2GRAY`` while carrying mostly noise, so ``channel: auto`` picks the
   channel with the widest spread (falls back to standard grey).
2. **Black-hat** — ``close(gray, K) − gray`` with ``K`` sized just above the
   largest expected bore. The result measures *how much darker than its own
   surroundings* each pixel is, so illumination gradients, shadowed regions
   and glare bands — all larger than ``K`` — drop out and only small dark
   features survive. This is what makes one parameter set hold across
   exposure and position changes.
3. **Threshold** — ``max(min_contrast, Otsu)`` on that response, then a small
   open/close to drop speckle and seal rims.
4. **Candidates** — the outline of every blob, *including* blobs nested inside
   a stamped outline, gated on fitted diameter and on ``min_fill_ratio``
   (0.35 ≈ a third of the bore visible) instead of on roundness. A candidate
   is measured from its outer outline, so a specular glint inside the bore
   cannot split it in two.
5. **Circle fit** — algebraic least-squares fit, refit after trimming the
   worst residuals, so the straight chord of a clipped bore does not drag the
   centre off the real one. The reported centre and diameter are therefore
   those of the *whole* bore even when only an arc is visible — which keeps
   the position judgement and the mm calibration meaningful.
6. **Score** — bore-vs-metal contrast, fit residual and arc coverage.
   Coverage is reported as ``Hole.circularity``: 1.0 = the whole rim is
   visible, ~0.5 = half of it.

All parameters come from the ``dark_hole`` block of detection.json.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import ClassVar

import cv2
import numpy as np

from core.vision.detection_result import DetectionResult, Hole
from core.vision.detector_base import HoleDetector
from core.utilities.exceptions import DetectionError

_CHANNELS = {"blue": 0, "green": 1, "red": 2}
_CHANNEL_CHOICES = frozenset({"auto", "gray", *_CHANNELS})
_ANGLE_BINS = 36                 # 10° buckets for arc coverage
_CONTRAST_SPAN = 100.0           # grey levels above min_contrast that score a full 1.0
_TRIM_FRACTION = 0.35            # share of outline points dropped per refit round
_TRIM_ROUNDS = 2
_MIN_FIT_POINTS = 8
_MAX_KERNEL_PX = 31              # above this the background closing is downscaled
_MAX_BACKGROUND_SCALE = 8


@dataclass(frozen=True)
class Candidate:
    """One blob the detector looked at: accepted, or dropped with the reason."""

    hole: Hole
    area: float
    fill: float                 # visible share of the fitted bore (1.0 = whole disc)
    fit_error: float            # RMS rim deviation / radius
    contrast: float             # bore-vs-metal darkness, 0..1
    rejected_because: str = ""  # empty when the candidate was accepted


def _rejection(
    diameter: float,
    min_diameter: float,
    max_diameter: float,
    fit_error: float,
    max_fit_error: float,
    fill: float,
    min_fill: float,
) -> str:
    """Why this candidate is not a bore, in the operator's words ("" = it is)."""
    if diameter < min_diameter:
        return f"diameter {diameter:.1f} px below min {min_diameter:.0f}"
    if diameter > max_diameter:
        return f"diameter {diameter:.1f} px above max {max_diameter:.0f}"
    if fit_error > max_fit_error:
        return f"rim not circular enough ({fit_error:.2f} > {max_fit_error:.2f})"
    if fill < min_fill:
        return f"only {fill:.0%} of the bore visible (min {min_fill:.0%})"
    if fill > 1.6:
        return f"blob much bigger than its circle fit (fill {fill:.2f})"
    return ""


def _odd(value: int) -> int:
    """Kernel sizes must be odd and >= 1."""
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


class DarkHoleDetector(HoleDetector):
    """Black-hat + circle-fit strategy; handles partially visible bores."""

    name: ClassVar[str] = "dark_hole"
    thread_safe: ClassVar[bool] = True

    def detect(self, image: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        _, candidates = self._scan(image)
        holes = [
            candidate.hole for candidate in candidates if not candidate.rejected_because
        ]
        holes.sort(key=lambda hole: hole.confidence, reverse=True)
        return DetectionResult(
            holes=holes, processing_ms=(time.perf_counter() - started) * 1000.0
        )

    def explain(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], list["Candidate"]]:
        """The same pass as :meth:`detect`, but returning the stage images and
        *every* candidate — including rejected ones and the reason.

        Used by ``tools/hole_debug.py`` to tune a station against real photos.
        It deliberately shares the production code path, so what the tool shows
        is what the line will do.
        """
        return self._scan(image)

    # ------------------------------------------------------------------ pass
    def _scan(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], list["Candidate"]]:
        params = self._params
        try:
            min_diameter = float(params.get("min_hole_diameter_px", 12))
            max_diameter = float(params.get("max_hole_diameter_px", 80))
            if min_diameter <= 0 or max_diameter <= min_diameter:
                raise DetectionError(
                    "dark_hole: max_hole_diameter_px must be greater than "
                    "min_hole_diameter_px (both > 0)"
                )
            min_fill = float(params.get("min_fill_ratio", 0.35))
            max_fit_error = max(0.01, float(params.get("max_fit_error", 0.25)))

            gray = self._to_gray(image, str(params.get("channel", "auto")).lower())
            blur = _odd(params.get("blur_kernel_size", 3))
            if blur > 1:
                gray = cv2.GaussianBlur(gray, (blur, blur), 0)

            response = self._blackhat(gray, max_diameter)
            mask = self._binarize(
                response,
                float(params.get("min_contrast", 18)),
                bool(params.get("use_otsu", True)),
            )
            mask = self._clean(mask, _odd(params.get("morphology_kernel_size", 3)))
            stages = {"gray": gray, "response": response, "mask": mask}

            # RETR_LIST, not RETR_EXTERNAL: a stamped bracket outline is a closed
            # dark loop and the bore sits *inside* it — as an inner contour it
            # would never be reported (this is what the opencv strategy misses).
            # CHAIN_APPROX_SIMPLE keeps every point of a curved rim but collapses
            # straight runs to their endpoints: large stamped edges become cheap
            # to gate, and the chord of a clipped bore stops out-voting its arc.
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # a bore at the smallest allowed size, seen at the smallest allowed
            # visible fraction, still has to clear this
            min_area = math.pi * (min_diameter / 2.0) ** 2 * min_fill * 0.8

            candidates: list[Candidate] = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or len(contour) < _MIN_FIT_POINTS:
                    continue  # surface speckle: too many to be worth reporting
                # cheap size pre-gate: the circle fit below is the expensive part
                # and most blobs on a textured surface die right here
                (ex, ey), enclosing_radius = cv2.minEnclosingCircle(contour)
                if not 0.5 * min_diameter <= 2.0 * enclosing_radius <= 1.5 * max_diameter:
                    continue

                points = contour.reshape(-1, 2).astype(np.float64)
                cx, cy, radius, fit_error = _fit_circle(points, (ex, ey, enclosing_radius))
                diameter = 2.0 * radius
                fill = area / (math.pi * radius * radius)
                contrast = self._contrast_score(
                    gray, contour, cx, cy, radius, float(params.get("min_contrast", 18))
                )
                coverage = self._arc_coverage(points, cx, cy, radius)
                fit_quality = 1.0 - min(1.0, fit_error / max_fit_error)
                confidence = min(
                    1.0, 0.45 * contrast + 0.25 * fit_quality + 0.30 * coverage
                )

                candidates.append(
                    Candidate(
                        hole=Hole(
                            x_px=float(cx),
                            y_px=float(cy),
                            diameter_px=float(diameter),
                            circularity=float(coverage),
                            confidence=float(confidence),
                        ),
                        area=float(area),
                        fill=float(fill),
                        fit_error=float(fit_error),
                        contrast=float(contrast),
                        rejected_because=_rejection(
                            diameter, min_diameter, max_diameter,
                            fit_error, max_fit_error, fill, min_fill,
                        ),
                    )
                )

            candidates.sort(key=lambda item: item.hole.confidence, reverse=True)
            return stages, candidates
        except DetectionError:
            raise
        except (cv2.error, ValueError, AttributeError) as exc:
            raise DetectionError(f"dark_hole detection failed: {exc}") from exc

    # --------------------------------------------------------------- stages
    @staticmethod
    def _to_gray(image: np.ndarray, channel: str) -> np.ndarray:
        """Single-channel view of *image* (see ``channel`` in detection.json)."""
        if channel not in _CHANNEL_CHOICES:
            raise DetectionError(
                f"dark_hole: unknown channel {channel!r} "
                f"(auto, gray, red, green, blue)"
            )
        if image.ndim == 2:
            return image
        if channel in _CHANNELS:
            return image[:, :, _CHANNELS[channel]].copy()
        if channel == "auto":
            # one optimised pass; numpy per-channel .std() on a non-contiguous
            # slice costs more than the rest of the pipeline put together
            spreads = [float(value) for value in cv2.meanStdDev(image)[1].ravel()]
            widest = int(np.argmax(spreads))
            # only override plain grey when one channel clearly carries the
            # signal (red/IR lighting), otherwise stay with the standard mix
            if spreads[widest] > 1.25 * float(np.median(spreads)):
                return image[:, :, widest].copy()
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _blackhat(gray: np.ndarray, max_diameter: float) -> np.ndarray:
        """Darker-than-surroundings response; suppresses anything bigger than a bore.

        The closing that estimates the background needs a kernel wider than the
        largest bore, which is expensive at production resolution — so it is
        computed on a downscaled copy (a background estimate does not need full
        detail) and stretched back. At 1280×1024 that is the difference between
        ~100 ms and a few ms per frame.
        """
        size = _odd(int(max_diameter * 1.6) + 1)
        scale = 1
        while size // scale > _MAX_KERNEL_PX and scale < _MAX_BACKGROUND_SCALE:
            scale *= 2

        if scale == 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        else:
            height, width = gray.shape
            small = cv2.resize(
                gray,
                (max(1, width // scale), max(1, height // scale)),
                interpolation=cv2.INTER_AREA,
            )
            kernel_size = _odd(max(3, size // scale))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,) * 2)
            background = cv2.resize(
                cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
        return cv2.subtract(background, gray)

    @staticmethod
    def _binarize(response: np.ndarray, min_contrast: float, use_otsu: bool) -> np.ndarray:
        """Threshold the response at ``max(min_contrast, Otsu)``.

        The floor keeps a hole-free frame from producing candidates out of
        surface texture — Otsu always splits *something*.
        """
        level = max(1.0, min_contrast)
        if use_otsu:
            otsu_level, _ = cv2.threshold(
                response, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            level = max(level, float(otsu_level))
        _, mask = cv2.threshold(response, level, 255, cv2.THRESH_BINARY)
        return mask

    @staticmethod
    def _clean(mask: np.ndarray, kernel_size: int) -> np.ndarray:
        """Drop speckle and seal rims.

        Interior gaps are deliberately *not* filled here: a stamped bracket
        outline is a closed loop, and filling it would merge the bore with the
        whole bracket. A glare speck inside the bore is handled later instead —
        candidates are measured from their outer contour, which ignores it.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --------------------------------------------------------------- scoring
    @staticmethod
    def _contrast_score(
        gray: np.ndarray,
        contour: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
        min_contrast: float,
    ) -> float:
        """How much darker than the surrounding metal the bore actually is.

        Scored from ``min_contrast`` (0.0) to ``min_contrast + _CONTRAST_SPAN``
        (1.0). The span matters: a through-hole returns no light and typically
        sits 100+ grey levels below the metal, while a stain, a shallow dent or
        a dark fleck of surface texture manages 40-70. Scoring both as "dark
        enough" is what lets texture pass as a bore.
        """
        height, width = gray.shape
        pad = int(radius * 2.2) + 3
        x0, y0 = max(0, int(cx) - pad), max(0, int(cy) - pad)
        x1, y1 = min(width, int(cx) + pad), min(height, int(cy) + pad)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return 0.0

        bore = np.zeros(crop.shape, np.uint8)
        cv2.drawContours(bore, [contour], -1, 255, cv2.FILLED, offset=(-x0, -y0))
        centre = (int(cx) - x0, int(cy) - y0)
        ring = np.zeros(crop.shape, np.uint8)
        cv2.circle(ring, centre, max(2, int(radius * 2.0)), 255, -1)
        cv2.circle(ring, centre, max(1, int(radius * 1.2)), 0, -1)
        cv2.bitwise_and(ring, cv2.bitwise_not(bore), dst=ring)  # metal only

        if not cv2.countNonZero(bore) or not cv2.countNonZero(ring):
            return 0.0
        difference = cv2.mean(crop, ring)[0] - cv2.mean(crop, bore)[0]
        return float(np.clip((difference - min_contrast) / _CONTRAST_SPAN, 0.0, 1.0))

    @staticmethod
    def _arc_coverage(
        points: np.ndarray, cx: float, cy: float, radius: float
    ) -> float:
        """Share of the fitted rim actually present: 1.0 whole bore, ~0.5 half."""
        dx = points[:, 0] - cx
        dy = points[:, 1] - cy
        distance = np.hypot(dx, dy)
        tolerance = max(1.5, 0.22 * radius)
        on_rim = np.abs(distance - radius) <= tolerance
        if not on_rim.any():
            return 0.0

        angles = np.arctan2(dy, dx)
        buckets = ((angles + math.pi) / (2.0 * math.pi) * _ANGLE_BINS).astype(int)
        buckets %= _ANGLE_BINS
        covered = np.zeros(_ANGLE_BINS, dtype=bool)
        covered[buckets[on_rim]] = True

        # The outline is ordered, and a straight run of rim was compressed to
        # its two endpoints — bridge those so a whole bore still reads 1.0. The
        # long jump across a clipped bore's chord is excluded by length, which
        # is what keeps a half bore reading ~0.5 instead of 1.0.
        bridge_limit = 0.75 * radius
        count = len(points)
        for index in range(count):
            following = (index + 1) % count
            if not (on_rim[index] and on_rim[following]):
                continue
            if math.hypot(*(points[following] - points[index])) > bridge_limit:
                continue
            start, step = buckets[index], int(buckets[following] - buckets[index]) % _ANGLE_BINS
            if step > _ANGLE_BINS // 2:  # wrapped the other way round
                start, step = buckets[following], _ANGLE_BINS - step
            for offset in range(1, step):
                covered[(start + offset) % _ANGLE_BINS] = True

        return float(covered.sum()) / _ANGLE_BINS


# --------------------------------------------------------------------------- #
# Circle fitting
# --------------------------------------------------------------------------- #
def _algebraic_circle(points: np.ndarray) -> tuple[float, float, float] | None:
    """Kåsa least-squares circle through *points*, or None if degenerate.

    Solves ``x² + y² = D·x + E·y + F`` in mean-centred coordinates (for
    conditioning); centre is ``(D/2, E/2)`` and ``r² = F + cx² + cy²``.
    """
    if len(points) < _MIN_FIT_POINTS:
        return None
    origin = points.mean(axis=0)
    x = points[:, 0] - origin[0]
    y = points[:, 1] - origin[1]
    design = np.column_stack((x, y, np.ones(len(points))))
    try:
        solution, *_ = np.linalg.lstsq(design, x * x + y * y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = solution[0] / 2.0, solution[1] / 2.0
    squared = solution[2] + cx * cx + cy * cy
    if not np.isfinite(squared) or squared <= 0.0:
        return None
    return float(cx + origin[0]), float(cy + origin[1]), float(math.sqrt(squared))


def _fit_circle(
    points: np.ndarray, fallback: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    """Circle through a blob outline, robust to a clipped bore's straight chord.

    Fits once, then twice drops the worst ``_TRIM_FRACTION`` of residuals —
    that is the chord and any ragged rim — and refits on what remains, so the
    centre and radius describe the *whole* bore even when only an arc shows.

    Returns:
        ``(cx, cy, radius, error)``; *error* is the RMS residual of the kept
        rim points relative to the radius (0 = perfect arc). Falls back to the
        minimum enclosing circle with ``error = 1.0`` when no fit is possible.
    """
    fallback_x, fallback_y, fallback_r = fallback
    if len(points) < _MIN_FIT_POINTS or fallback_r <= 0:
        return fallback_x, fallback_y, fallback_r, 1.0

    fitted = _algebraic_circle(points)
    if fitted is None:
        return fallback_x, fallback_y, fallback_r, 1.0

    kept = points
    cx, cy, radius = fitted
    # a chord can be a third of the outline of a half-visible bore, so one
    # trimming round leaves some of it in — two rounds shake it off while a
    # complete rim (all residuals tiny) survives either way
    for _ in range(_TRIM_ROUNDS):
        residual = np.abs(np.hypot(kept[:, 0] - cx, kept[:, 1] - cy) - radius)
        limit = float(np.quantile(residual, 1.0 - _TRIM_FRACTION))
        subset = kept[residual <= limit]
        if len(subset) < _MIN_FIT_POINTS:
            break
        refitted = _algebraic_circle(subset)
        if refitted is None:
            break
        kept, (cx, cy, radius) = subset, refitted

    if not (0.4 * fallback_r <= radius <= 2.5 * fallback_r):
        return fallback_x, fallback_y, fallback_r, 1.0  # fit ran away

    kept_residual = np.abs(np.hypot(kept[:, 0] - cx, kept[:, 1] - cy) - radius)
    error = float(math.sqrt(float(np.mean(kept_residual**2))) / radius)
    return cx, cy, radius, error

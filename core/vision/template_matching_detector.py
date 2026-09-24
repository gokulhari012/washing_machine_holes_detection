"""Template-matching hole detector.

Configure with a grayscale crop of a known-good hole (``template_path``); at
runtime ``cv2.matchTemplate`` correlates it over the frame and every peak
scoring above ``match_threshold`` becomes a candidate. Robust when hole
appearance is stable and lighting is controlled; insensitive to hole *shape*
assumptions, which is what makes it the fallback when a bore is too irregular
for ``dark_hole``'s circle fit.

Pipeline
--------
1. grayscale
2. for each entry in ``scales``, the template resized by that factor is
   correlated over the frame — a single scale is brittle, because the bore's
   apparent size moves with part height and camera standoff, and a 10 %
   mismatch collapses a normalised correlation score well below any usable
   threshold. The scale set is searched jointly, not per-scale independently:
   a hole is reported once, at whichever scale fitted it best.
3. ``method`` picks the correlation; ``TM_SQDIFF_NORMED`` is inverted to
   "higher is better" so one comparison serves all three.
4. greedy peak extraction with **cross-scale** non-maximum suppression: the
   global best peak over every scale map is accepted, then every map is
   suppressed around that image position, so one bore matched at three
   neighbouring scales yields one hole rather than three.
5. size gate on the matched template's own diameter
   (``min_hole_diameter_px``/``max_hole_diameter_px``, either 0 = no gate) —
   with several scales in play this is what stops a 3x-scaled match on a
   background feature being reported as a hole.

``circularity`` is reported as 1.0: this strategy never fits a shape, so it
has no roundness measurement to offer. ``diameter_px`` is the matched
template's mean side, which is a real measurement only insofar as the
template was cropped tight to the bore — crop it tight.

Note on ``method``: ``TM_CCORR_NORMED`` does not subtract the mean, so on a
bright, low-contrast plate it scores near 1.0 almost everywhere and is
effectively unusable; ``TM_CCOEFF_NORMED`` is the default for that reason.
All parameters come from the ``template_matching`` block of detection.json.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, ClassVar

import cv2
import numpy as np

from core.vision.detection_result import DetectionResult, Hole
from core.vision.detector_base import HoleDetector
from core.utilities.exceptions import DetectionError

_METHODS = {
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,  # inverted below: lower = better
}

#: Score written into a suppressed region — low enough that it can never win a
#: later ``np.max``, finite so the arithmetic around it stays well-behaved.
_SUPPRESSED = -1.0e9

#: Two accepted peaks must sit at least this many *matched diameters* apart.
#: 1.0 encodes "two bores cannot overlap": anything closer is the same hole
#: matched again at a neighbouring offset or scale.
_NMS_DIAMETER_FACTOR = 1.0

#: Upper bound on entries in ``scales`` — each is a full matchTemplate pass,
#: and a config asking for hundreds would stall the inspection cycle.
_MAX_SCALES = 25

DEFAULT_MAX_MATCHES = 10


def parse_scales(raw: Any) -> list[float]:
    """Normalise the ``scales`` parameter into a sorted list of positive floats.

    Accepts a JSON list (``[0.9, 1.0, 1.1]``) or the comma-separated string the
    Detection page's field produces (``"0.9, 1.0, 1.1"``). Empty or
    unparseable degrades to ``[1.0]`` — the template at its authored size —
    because a typo in a tuning field must not take the station's detector out
    of service.
    """
    if raw is None or raw == "":
        return [1.0]
    if isinstance(raw, (int, float)):
        raw = [raw]
    elif isinstance(raw, str):
        raw = [piece for piece in raw.replace(";", ",").split(",") if piece.strip()]
    values: list[float] = []
    for item in raw:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(round(value, 4))
    if not values:
        return [1.0]
    return sorted(set(values))[:_MAX_SCALES]


def template_mean_side(template_path: str) -> float:
    """Mean side of the template image in px — the ``diameter_px`` a match
    reports at scale 1.0.

    The Detection page's Auto Sweep uses it to turn the drawn ROI's size into
    the *scale range worth searching* (ROI diameter / this), which is the one
    parameter of this strategy nobody can guess by eye. It lives here rather
    than in the page so the UI layer does not have to open image files.

    Raises:
        DetectionError: no path, or the file cannot be read as an image.
    """
    if not template_path:
        raise DetectionError(
            "Template matching has no 'template_path' configured — choose a "
            "template image before sweeping"
        )
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None or template.size == 0:
        raise DetectionError(f"Cannot read template image: {template_path}")
    height, width = template.shape[:2]
    return (height + width) / 2.0


class TemplateMatchingDetector(HoleDetector):
    """Normalised cross-correlation against a golden hole template."""

    name: ClassVar[str] = "template_matching"
    thread_safe: ClassVar[bool] = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._template: np.ndarray | None = None
        self._template_key: tuple[str, float, int] | None = None
        self._scaled: list[np.ndarray] = []  # template resized by each entry in _scales
        self._scales: list[float] = [1.0]
        super().__init__(params)

    # ----------------------------------------------------------- configure
    def configure(self, params: dict[str, Any]) -> None:
        """Load the template and pre-build its scale pyramid.

        Both are cached on the file's identity (path + mtime + size) and on
        the scale list, so the Detection page's Auto Sweep — which
        reconfigures one instance once per trial — re-reads the file only
        when it actually changes, and a template re-cropped on disk under an
        unchanged filename is still picked up.
        """
        super().configure(params)
        template_path = str(self._params.get("template_path", "") or "")
        scales = parse_scales(self._params.get("scales"))

        if not template_path:
            self._template = None
            self._template_key = None
            self._scaled = []
            self._scales = scales
            return

        key = self._file_key(template_path)
        if key != self._template_key or self._template is None:
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise DetectionError(f"Cannot read template image: {template_path}")
            if template.size == 0:
                raise DetectionError(f"Template image is empty: {template_path}")
            self._template = template
            self._template_key = key
            self._scaled = []  # force a rebuild against the new template

        if not self._scaled or scales != self._scales:
            self._scales = scales
            self._scaled = self._build_pyramid(self._template, scales)

    @staticmethod
    def _file_key(path: str) -> tuple[str, float, int]:
        """Identity of the template file — path, mtime and size.

        A missing file yields a sentinel key that cannot collide with a real
        stat, so the next ``configure`` retries the read (and raises a useful
        error) instead of silently reusing a stale template.
        """
        try:
            stat = os.stat(path)
        except OSError:
            return (path, -1.0, -1)
        return (path, stat.st_mtime, stat.st_size)

    @staticmethod
    def _build_pyramid(template: np.ndarray, scales: list[float]) -> list[np.ndarray]:
        """Resize *template* by each scale, dropping any that round away to nothing."""
        height, width = template.shape[:2]
        built: list[np.ndarray] = []
        for scale in scales:
            if abs(scale - 1.0) < 1e-6:
                built.append(template)
                continue
            new_w, new_h = int(round(width * scale)), int(round(height * scale))
            if new_w < 2 or new_h < 2:
                continue  # scaled past usefulness; the other scales still stand
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            built.append(cv2.resize(template, (new_w, new_h), interpolation=interpolation))
        return built

    # -------------------------------------------------------------- detect
    def detect(self, image: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        threshold = float(self._params.get("match_threshold", 0.8))
        max_matches = max(1, int(self._params.get("max_matches", DEFAULT_MAX_MATCHES)))
        min_diameter, max_diameter = self._diameter_gate()

        maps = self._score_maps(image)
        holes: list[Hole] = []
        # One extra round per size-gated reject, so a rejected match does not
        # consume a slot a real hole further down the ranking could have used.
        for _ in range(max_matches * 2):
            if len(holes) >= max_matches:
                break
            peak = self._best_peak(maps)
            if peak is None:
                break
            score, map_index, x, y = peak
            if score < threshold:
                break
            template_h, template_w = maps[map_index][1].shape[:2]
            center_x = x + template_w / 2.0
            center_y = y + template_h / 2.0
            diameter = (template_w + template_h) / 2.0
            self._suppress(maps, center_x, center_y, diameter)
            if min_diameter and diameter < min_diameter:
                continue
            if max_diameter and diameter > max_diameter:
                continue
            holes.append(
                Hole(
                    x_px=center_x,
                    y_px=center_y,
                    diameter_px=diameter,
                    circularity=1.0,  # not measured by this strategy
                    confidence=float(min(1.0, max(0.0, score))),
                )
            )

        holes.sort(key=lambda hole: hole.confidence, reverse=True)
        return DetectionResult(
            holes=holes,
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )

    def debug_stages(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Every location whose correlation clears ``match_threshold``, at any
        scale — the "what is it reacting to right now" view, before the
        non-maximum suppression and the size gate thin it down to holes.

        A score map is smaller than the frame (it is indexed by the template's
        top-left corner), so each scale's map is pasted into a frame-sized
        canvas offset by half that scale's template — putting the response
        over the bore it matched rather than up and to the left of it — and
        the canvases are combined by taking the per-pixel best.
        """
        threshold = float(self._params.get("match_threshold", 0.8))
        combined = self._combined_response(image)
        return {"mask": ((combined >= threshold) * 255).astype(np.uint8)}

    # ------------------------------------------------------------ internal
    def _diameter_gate(self) -> tuple[float, float]:
        """``(min, max)`` matched-diameter gate; 0 on either side disables it.

        Optional — and absent from a template block written before multi-scale
        matching existed — because at a single scale every match has the
        template's own size, so there is nothing for a gate to separate.
        """
        min_diameter = max(0.0, float(self._params.get("min_hole_diameter_px", 0) or 0))
        max_diameter = max(0.0, float(self._params.get("max_hole_diameter_px", 0) or 0))
        if min_diameter and max_diameter and max_diameter <= min_diameter:
            raise DetectionError(
                "template_matching: max_hole_diameter_px must be greater than "
                "min_hole_diameter_px (or 0 to disable the gate)"
            )
        return min_diameter, max_diameter

    def _score_maps(self, image: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """``(score_map, scaled_template)`` for every scale that fits in *image*.

        Raises:
            DetectionError: no template configured, unknown method, or every
                scale is larger than the frame.
        """
        if self._template is None:
            raise DetectionError(
                "Template matching selected but 'template_path' is not configured"
            )
        method_name = str(self._params.get("method", "TM_CCOEFF_NORMED"))
        if method_name not in _METHODS:
            raise DetectionError(f"Unknown template matching method: {method_name!r}")
        method = _METHODS[method_name]

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            height, width = gray.shape[:2]
            maps: list[tuple[np.ndarray, np.ndarray]] = []
            for template in self._scaled:
                template_h, template_w = template.shape[:2]
                if template_h > height or template_w > width:
                    continue  # this scale does not fit; smaller ones may
                scores = cv2.matchTemplate(gray, template, method)
                if method == cv2.TM_SQDIFF_NORMED:
                    scores = 1.0 - scores  # normalise to "higher is better"
                maps.append((scores.astype(np.float32, copy=True), template))
        except cv2.error as exc:
            raise DetectionError(f"Template matching failed: {exc}") from exc

        if not maps:
            raise DetectionError(
                "Image smaller than template at every configured scale (template "
                f"{self._template.shape[1]}x{self._template.shape[0]} px, "
                f"scales {self._scales}) — re-crop the template or add a smaller scale"
            )
        return maps

    def _combined_response(self, image: np.ndarray) -> np.ndarray:
        """Per-pixel best score across every scale, in the frame's own basis."""
        maps = self._score_maps(image)
        height, width = image.shape[:2]
        combined = np.full((height, width), _SUPPRESSED, np.float32)
        for scores, template in maps:
            template_h, template_w = template.shape[:2]
            offset_x, offset_y = template_w // 2, template_h // 2
            score_h, score_w = scores.shape[:2]
            window = combined[offset_y : offset_y + score_h, offset_x : offset_x + score_w]
            np.maximum(
                window, scores[: window.shape[0], : window.shape[1]], out=window
            )
        return combined

    @staticmethod
    def _best_peak(
        maps: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[float, int, int, int] | None:
        """Global best ``(score, map_index, x, y)`` across every scale map."""
        best: tuple[float, int, int, int] | None = None
        for index, (scores, _template) in enumerate(maps):
            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(scores)
            if best is None or max_val > best[0]:
                best = (float(max_val), index, int(max_loc[0]), int(max_loc[1]))
        return best

    @staticmethod
    def _suppress(
        maps: list[tuple[np.ndarray, np.ndarray]],
        center_x: float,
        center_y: float,
        diameter: float,
    ) -> None:
        """Blank every scale map around the image position just accepted.

        Each map is indexed by its own template's top-left corner, so the
        image-space exclusion box is translated into each map's frame by
        subtracting that template's half-size — which is what makes the
        suppression cross-scale rather than per-map, and is why one bore
        matched at three neighbouring scales reports as one hole.
        """
        radius = max(1.0, diameter * _NMS_DIAMETER_FACTOR)
        for scores, template in maps:
            template_h, template_w = template.shape[:2]
            score_h, score_w = scores.shape[:2]
            x_lo = int(math.floor(center_x - radius - template_w / 2.0))
            x_hi = int(math.ceil(center_x + radius - template_w / 2.0))
            y_lo = int(math.floor(center_y - radius - template_h / 2.0))
            y_hi = int(math.ceil(center_y + radius - template_h / 2.0))
            x_lo, y_lo = max(0, x_lo), max(0, y_lo)
            x_hi, y_hi = min(score_w - 1, x_hi), min(score_h - 1, y_hi)
            if x_lo <= x_hi and y_lo <= y_hi:
                scores[y_lo : y_hi + 1, x_lo : x_hi + 1] = _SUPPRESSED

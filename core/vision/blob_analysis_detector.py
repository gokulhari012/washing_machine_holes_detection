"""Blob analysis: the largest dark, round blob is the hole.

Ported from the lab script ``development_files/testing_own/holes_detecion.py``
(``detect_hole``), a center-hole detector tuned by hand with four sliders —
threshold, min area, min circularity, blur. The pipeline is kept exactly as
that script runs it, so the numbers found there carry straight over:

1. grayscale -> Gaussian blur (``blur_kernel_size``, made odd)
2. inverse binary threshold at ``threshold`` — every pixel darker than it is
   foreground
3. morphology: open, then close, with a square ``morphology_kernel_size``
   kernel, made odd (the script hard-coded 5) — open removes speckle, close
   fills pin-holes in the bore
4. external contours (``CHAIN_APPROX_NONE``, as the script used, since
   circularity is measured on the contour's own perimeter)
5. gates per contour: area >= ``min_area_px``, circularity (4πA/P²) >=
   ``min_circularity``, and the diameter gates every strategy shares
   (``min/max_hole_diameter_px``, 0 = off — the script had none; they exist
   so Auto Sweep can scope the search from a drawn ROI)
6. the **largest-area** survivor is the hole. Its centre and diameter come
   from ``cv2.fitEllipse`` (>= 5 contour points), else the centroid and the
   minimum enclosing circle.

Exactly one hole per frame
--------------------------
The script answers "where is *the* hole", and so does this strategy: it
reports the one blob step 6 picks, or nothing. That is deliberate. The
pipeline judges the highest-*confidence* hole (``select_hole``), and the
confidence here is the blob's circularity — so reporting every survivor
would make the pipeline judge the *roundest* blob, not the *largest* one the
script picks, and the station would disagree with the lab on the same frame.
The Debug view shows every blob the mask holds, which is the script's "Mask"
view. A camera expecting more than one hole needs ``template_blob_analysis``
with "Regions Searched" at that count (one hole per matched region), or a
different strategy.

``confidence`` is the blob's circularity, clamped to 1.0: the script's only
measure of how hole-like a blob is. ``common.confidence_threshold`` therefore
acts as a second circularity floor on top of ``min_circularity``.
"""

from __future__ import annotations

import math
import time
from typing import Any, ClassVar

import cv2
import numpy as np

from core.utilities.exceptions import DetectionError
from core.vision.detection_result import DetectionResult, Hole
from core.vision.detector_base import HoleDetector, shrink_kernel

#: Pixel-sized parameters beyond the diameter gates — see downscaled_params.
_KERNEL_KEYS = ("blur_kernel_size", "morphology_kernel_size")


def _odd(value: Any) -> int:
    """Kernel sizes must be odd and >= 1 (the script rounded up the same way)."""
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


class BlobAnalysisDetector(HoleDetector):
    """Threshold, clean up, and keep the largest blob that is round enough."""

    name: ClassVar[str] = "blob_analysis"
    thread_safe: ClassVar[bool] = True  # stateless per call

    @classmethod
    def downscaled_params(cls, params: dict[str, Any], factor: float) -> dict[str, Any]:
        """Diameter gates and both kernels, plus ``min_area_px`` — an *area*,
        so it shrinks by the factor squared."""
        scaled = super().downscaled_params(params, factor)
        for key in _KERNEL_KEYS:
            if key in scaled:
                scaled[key] = shrink_kernel(scaled[key], factor)
        if "min_area_px" in scaled:
            scaled["min_area_px"] = float(scaled["min_area_px"] or 0) / (factor * factor)
        return scaled

    def detect(self, image: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        try:
            mask = self._mask(image)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            hole = self._largest_round_blob(contours)
        except (cv2.error, ValueError, TypeError) as exc:
            raise DetectionError(f"Blob analysis failed: {exc}") from exc
        return DetectionResult(
            holes=[hole] if hole is not None else [],
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )

    def debug_stages(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """The cleaned threshold mask — every blob the gates then judge."""
        try:
            return {"mask": self._mask(image)}
        except (cv2.error, ValueError, TypeError) as exc:
            raise DetectionError(f"Blob analysis failed: {exc}") from exc

    # ------------------------------------------------------------- internal
    def _mask(self, image: np.ndarray) -> np.ndarray:
        """gray -> blur -> inverse threshold -> open -> close."""
        params = self._params
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blur = _odd(params.get("blur_kernel_size", 7))
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
        _, mask = cv2.threshold(
            gray, int(params.get("threshold", 40)), 255, cv2.THRESH_BINARY_INV
        )
        # Odd, like the blur: an even square kernel has an off-centre anchor,
        # and open+close with one shifts the whole blob — 2 px diagonally on a
        # ÷2 frame, where a typed 5 px kernel becomes 2 px.
        size = _odd(params.get("morphology_kernel_size", 5))
        kernel = np.ones((size, size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _largest_round_blob(self, contours: Any) -> Hole | None:
        params = self._params
        min_area = float(params.get("min_area_px", 500) or 0)
        min_circularity = float(params.get("min_circularity", 0.8))
        min_diameter = float(params.get("min_hole_diameter_px", 0) or 0)
        max_diameter = float(params.get("max_hole_diameter_px", 0) or 0)

        best: Hole | None = None
        best_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            # <= best_area: only a strictly larger blob replaces the current
            # one (the script's rule), which also rules out zero-area contours.
            if area < min_area or area <= best_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue

            cx, cy, diameter = _centre_and_diameter(contour)
            if min_diameter > 0 and diameter < min_diameter:
                continue
            if max_diameter > 0 and diameter > max_diameter:
                continue

            best_area = area
            circularity = min(1.0, circularity)
            best = Hole(
                x_px=float(cx),
                y_px=float(cy),
                diameter_px=float(diameter),
                circularity=float(circularity),
                confidence=float(circularity),
            )
        return best


def _centre_and_diameter(contour: np.ndarray) -> tuple[float, float, float]:
    """Ellipse fit when the contour has enough points, else the centroid and
    the minimum enclosing circle — the script's fallback order."""
    if len(contour) >= 5:
        (cx, cy), (minor, major), _angle = cv2.fitEllipse(contour)
        return cx, cy, (minor + major) / 2.0
    moments = cv2.moments(contour)
    (ex, ey), radius = cv2.minEnclosingCircle(contour)
    if moments["m00"] > 0:
        ex, ey = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    return ex, ey, 2.0 * radius

"""Classical OpenCV hole detector.

Contour/shape stage ported from ``scripts/dark_contour_lab.py``'s
``find_dark_blobs`` (an ellipse fit instead of a plain enclosing circle, plus
an aspect-ratio gate) — that script's docstring calls itself a lab for tuning
this exact pipeline "before [it gets] ported into core/vision".

Pipeline
--------
1. grayscale → Gaussian blur (``blur_kernel_size``)
2. inverse binary threshold — fixed (``detection_threshold``) or adaptive —
   holes are darker than the sheet metal
3. morphology (``morphology_operation/kernel/iterations``) to clean speckle
4. contours (``contour_retrieval_mode``) → size pre-gate → roundness gate
   (``min_circularity``, 4πA/P²)
5. shape fit: ``cv2.fitEllipse`` on contours with ≥5 points (falls back to
   the minimum enclosing circle below that) — the ellipse's minor/major axes
   give a diameter that holds up under perspective skew, where a plain
   enclosing circle overestimates it
6. size gate on that diameter, then an aspect-ratio gate
   (``min_aspect_ratio`` = minor/major) that rejects scratches and shadow
   streaks a round-hole gate alone would miss
7. confidence per surviving candidate:

       0.45·circularity + 0.35·contrast + 0.20·rim-edge-support

   - *contrast*: interior mean vs. surrounding annulus mean (a real hole is
     much darker than the metal around it)
   - *rim edge support*: fraction of the candidate's rim lying on Canny edges
     (``edge_threshold_low/high``) — rejects smudges and shadows with soft
     boundaries

All parameters come from the ``opencv`` block of detection.json.
"""

from __future__ import annotations

import math
import time
from typing import Any, ClassVar

import cv2
import numpy as np

from core.vision.detection_result import DetectionResult, Hole
from core.vision.detector_base import HoleDetector
from core.utilities.exceptions import DetectionError

_CONTOUR_MODES = {
    "external": cv2.RETR_EXTERNAL,
    "list": cv2.RETR_LIST,
    "tree": cv2.RETR_TREE,
}
_MORPH_OPS = {
    "close": cv2.MORPH_CLOSE,
    "open": cv2.MORPH_OPEN,
    "none": None,
}


def _odd(value: int) -> int:
    """Kernel sizes must be odd and >= 1."""
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


class OpenCVHoleDetector(HoleDetector):
    """Threshold + contour analysis strategy (the production default)."""

    name: ClassVar[str] = "opencv"
    thread_safe: ClassVar[bool] = True

    def detect(self, image: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        try:
            mask, edges, blurred = self._pipeline(image)

            contour_mode = _CONTOUR_MODES.get(
                str(self._params.get("contour_retrieval_mode", "external")).lower(),
                cv2.RETR_EXTERNAL,
            )
            contours, _ = cv2.findContours(mask, contour_mode, cv2.CHAIN_APPROX_SIMPLE)

            holes = self._holes_from_contours(contours, blurred, edges)
            holes.sort(key=lambda hole: hole.confidence, reverse=True)
            return DetectionResult(
                holes=holes,
                processing_ms=(time.perf_counter() - started) * 1000.0,
            )
        except DetectionError:
            raise
        except (cv2.error, ValueError, AttributeError) as exc:
            raise DetectionError(f"OpenCV detection failed: {exc}") from exc

    def debug_stages(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Threshold mask + Canny edge map, for the Detection page's debug overlay."""
        mask, edges, _blurred = self._pipeline(image)
        return {"mask": mask, "edges": edges}

    # ------------------------------------------------------------- internal
    def _pipeline(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """gray → blur → threshold → morphology, plus a Canny edge map.

        Returns:
            ``(mask, edges, blurred)``.
        """
        params = self._params
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        kernel_size = _odd(params.get("blur_kernel_size", 5))
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        if params.get("adaptive_threshold", False):
            mask = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                blockSize=51, C=10,
            )
        else:
            threshold = int(params.get("detection_threshold", 60))
            _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

        morph_name = str(params.get("morphology_operation", "close")).lower()
        if morph_name not in _MORPH_OPS:
            raise DetectionError(f"Unknown morphology_operation: {morph_name!r}")
        morph_op = _MORPH_OPS[morph_name]
        if morph_op is not None:
            morph_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (_odd(params.get("morphology_kernel_size", 5)),) * 2,
            )
            mask = cv2.morphologyEx(
                mask, morph_op, morph_kernel,
                iterations=max(1, int(params.get("morphology_iterations", 1))),
            )

        edges = cv2.Canny(
            blurred,
            int(params.get("edge_threshold_low", 50)),
            int(params.get("edge_threshold_high", 150)),
        )
        return mask, edges, blurred

    def _holes_from_contours(
        self, contours: Any, blurred: np.ndarray, edges: np.ndarray
    ) -> list[Hole]:
        """Size/roundness/aspect-ratio gates + confidence scoring per contour."""
        params = self._params
        min_diameter = float(params.get("min_hole_diameter_px", 20))
        max_diameter = float(params.get("max_hole_diameter_px", 200))
        min_circularity = float(params.get("min_circularity", 0.7))
        min_aspect_ratio = float(params.get("min_aspect_ratio", 0.35))
        min_area = math.pi * (min_diameter / 2.0) ** 2 * 0.5  # fast pre-gate

        holes: list[Hole] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue

            if len(contour) >= 5:
                (cx, cy), (minor, major), _angle = cv2.fitEllipse(contour)
                if minor > major:
                    minor, major = major, minor
            else:
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                minor = major = 2.0 * radius

            diameter = (major + minor) / 2.0
            if not min_diameter <= diameter <= max_diameter:
                continue

            aspect_ratio = minor / major if major > 0 else 0.0
            if aspect_ratio < min_aspect_ratio:
                continue  # too elongated to be a hole (scratch / shadow streak)

            moments = cv2.moments(contour)
            if moments["m00"] > 0:  # centroid beats the fit centre on ragged rims
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]

            radius = diameter / 2.0
            contrast = self._contrast_score(blurred, cx, cy, radius)
            edge_support = self._rim_edge_score(edges, cx, cy, radius)
            confidence = min(
                1.0, 0.45 * circularity + 0.35 * contrast + 0.20 * edge_support
            )

            holes.append(
                Hole(
                    x_px=float(cx),
                    y_px=float(cy),
                    diameter_px=float(diameter),
                    circularity=float(min(1.0, circularity)),
                    confidence=float(confidence),
                )
            )
        return holes
    @staticmethod
    def _contrast_score(gray: np.ndarray, cx: float, cy: float, radius: float) -> float:
        """Interior-vs-annulus darkness, scaled so ≥100 grey levels → 1.0."""
        height, width = gray.shape
        pad = int(radius * 1.6) + 2
        x0, y0 = max(0, int(cx) - pad), max(0, int(cy) - pad)
        x1, y1 = min(width, int(cx) + pad), min(height, int(cy) + pad)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return 0.0
        centre = (int(cx) - x0, int(cy) - y0)

        inner_mask = np.zeros(crop.shape, np.uint8)
        cv2.circle(inner_mask, centre, max(1, int(radius * 0.7)), 255, -1)
        ring_mask = np.zeros(crop.shape, np.uint8)
        cv2.circle(ring_mask, centre, int(radius * 1.5), 255, -1)
        cv2.circle(ring_mask, centre, int(radius * 1.1), 0, -1)

        if not cv2.countNonZero(inner_mask) or not cv2.countNonZero(ring_mask):
            return 0.0
        inner_mean = cv2.mean(crop, inner_mask)[0]
        ring_mean = cv2.mean(crop, ring_mask)[0]
        return float(np.clip((ring_mean - inner_mean) / 100.0, 0.0, 1.0))

    @staticmethod
    def _rim_edge_score(edges: np.ndarray, cx: float, cy: float, radius: float) -> float:
        """Fraction of the expected rim length supported by Canny edge pixels."""
        rim_band = np.zeros(edges.shape, np.uint8)
        cv2.circle(rim_band, (int(cx), int(cy)), max(1, int(radius)), 255, 5)
        overlap = cv2.countNonZero(cv2.bitwise_and(rim_band, edges))
        expected_rim = math.pi * 2.0 * radius
        if expected_rim <= 0:
            return 0.0
        return float(np.clip(overlap / expected_rim, 0.0, 1.0))

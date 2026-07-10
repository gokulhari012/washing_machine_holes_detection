"""Template-matching hole detector.

Configure with a grayscale crop of a known-good hole (``template_path``); at
runtime ``cv2.matchTemplate`` + non-maximum suppression returns every location
scoring above ``match_threshold``. Robust when hole appearance is stable and
lighting is controlled; insensitive to hole *shape* assumptions.
"""

from __future__ import annotations

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
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,  # note: lower = better for SQDIFF
}
MAX_MATCHES = 10


class TemplateMatchingDetector(HoleDetector):
    """Normalised cross-correlation against a golden hole template."""

    name: ClassVar[str] = "template_matching"
    thread_safe: ClassVar[bool] = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._template: np.ndarray | None = None
        super().__init__(params)

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        template_path = str(self._params.get("template_path", "") or "")
        self._template = None
        if template_path:
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise DetectionError(f"Cannot read template image: {template_path}")
            self._template = template

    def detect(self, image: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        if self._template is None:
            raise DetectionError(
                "Template matching selected but 'template_path' is not configured"
            )

        method_name = str(self._params.get("method", "TM_CCOEFF_NORMED"))
        if method_name not in _METHODS:
            raise DetectionError(f"Unknown template matching method: {method_name!r}")
        method = _METHODS[method_name]
        threshold = float(self._params.get("match_threshold", 0.8))

        try:
            gray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            )
            template_h, template_w = self._template.shape
            if gray.shape[0] < template_h or gray.shape[1] < template_w:
                raise DetectionError("Image smaller than template")

            scores = cv2.matchTemplate(gray, self._template, method)
            if method == cv2.TM_SQDIFF_NORMED:
                scores = 1.0 - scores  # normalise to "higher is better"

            holes: list[Hole] = []
            working = scores.copy()
            for _ in range(MAX_MATCHES):
                _, best_score, _, best_loc = cv2.minMaxLoc(working)
                if best_score < threshold:
                    break
                x, y = best_loc
                holes.append(
                    Hole(
                        x_px=x + template_w / 2.0,
                        y_px=y + template_h / 2.0,
                        diameter_px=(template_w + template_h) / 2.0,
                        circularity=1.0,  # not measured by this strategy
                        confidence=float(min(1.0, best_score)),
                    )
                )
                # non-maximum suppression: blank one template footprint
                x0, y0 = max(0, x - template_w // 2), max(0, y - template_h // 2)
                working[y0 : y + template_h, x0 : x + template_w] = -1.0

            holes.sort(key=lambda hole: hole.confidence, reverse=True)
            return DetectionResult(
                holes=holes,
                processing_ms=(time.perf_counter() - started) * 1000.0,
            )
        except DetectionError:
            raise
        except cv2.error as exc:
            raise DetectionError(f"Template matching failed: {exc}") from exc

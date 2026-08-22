"""Strategy context for hole detection + shared overlay rendering.

``VisionEngine`` owns the active :class:`HoleDetector`, applies the
``detection.json`` configuration (hot-swappable at runtime via
``apply_config``), filters candidates by the common confidence threshold, and
serialises ``detect()`` for strategies that declare themselves not
thread-safe (the classical detectors run 4 images in parallel).
"""

from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np

from core.logging import get_logger
from core.utilities.enums import DetectorType, LogSource
from core.utilities.exceptions import ConfigurationError
from core.vision.dark_hole_detector import DarkHoleDetector
from core.vision.detection_result import DetectionResult
from core.vision.detector_base import HoleDetector
from core.vision.opencv_hole_detector import OpenCVHoleDetector
from core.vision.template_matching_detector import TemplateMatchingDetector
from core.vision.yolo_hole_detector import YoloHoleDetector

logger = get_logger(LogSource.VISION)

_REGISTRY: dict[DetectorType, type[HoleDetector]] = {
    DetectorType.OPENCV: OpenCVHoleDetector,
    DetectorType.TEMPLATE_MATCHING: TemplateMatchingDetector,
    DetectorType.YOLO: YoloHoleDetector,
    DetectorType.DARK_HOLE: DarkHoleDetector,
}


class VisionEngine:
    """Facade the rest of the application talks to for detection."""

    def __init__(self, detection_config: dict[str, Any]) -> None:
        self._swap_lock = threading.Lock()      # protects strategy replacement
        self._detect_lock = threading.Lock()    # used only for non-thread-safe strategies
        self._detector: HoleDetector | None = None
        self._common: dict[str, Any] = {}
        self.apply_config(detection_config)

    # ------------------------------------------------------------- configure
    def apply_config(self, detection_config: dict[str, Any]) -> None:
        """(Re)build the active strategy from a full detection.json dict.

        Raises:
            ConfigurationError: unknown detector name.
            DetectionError: strategy rejected its parameters (bad template...).
        """
        active_name = str(detection_config.get("active_detector", "opencv")).lower()
        try:
            detector_type = DetectorType(active_name)
        except ValueError as exc:
            raise ConfigurationError(f"Unknown active_detector: {active_name!r}") from exc

        params = dict(detection_config.get(detector_type.value, {}))
        detector = _REGISTRY[detector_type](params)

        with self._swap_lock:
            self._detector = detector
            self._common = dict(detection_config.get("common", {}))
        logger.info("Vision engine strategy -> %s", detector_type.value)

    # ------------------------------------------------------------ properties
    @property
    def active_detector_name(self) -> str:
        with self._swap_lock:
            assert self._detector is not None
            return self._detector.name

    @property
    def confidence_threshold(self) -> float:
        return float(self._common.get("confidence_threshold", 0.6))

    @property
    def expected_hole_count(self) -> int:
        return int(self._common.get("expected_hole_count", 1))

    @property
    def position_tolerance_mm(self) -> float:
        """<= 0 disables the position tolerance check."""
        return float(self._common.get("position_tolerance_mm", 0.0))

    # ---------------------------------------------------------------- detect
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run the active strategy; candidates below the common confidence
        threshold are dropped so callers only ever see viable holes.

        Raises:
            DetectionError
        """
        with self._swap_lock:
            assert self._detector is not None
            detector = self._detector

        if detector.thread_safe:
            result = detector.detect(image)
        else:
            with self._detect_lock:
                result = detector.detect(image)

        threshold = self.confidence_threshold
        result.holes = [hole for hole in result.holes if hole.confidence >= threshold]
        return result


# --------------------------------------------------------------------------- #
# Overlay rendering (dashboard panels, calibration live test, saved NG images)
# --------------------------------------------------------------------------- #
_GREEN = (80, 220, 80)
_YELLOW = (60, 200, 240)
_RED = (70, 70, 230)


def draw_detection_overlay(
    frame: np.ndarray,
    result: DetectionResult | None,
    label: str = "",
) -> np.ndarray:
    """Return a BGR copy of *frame* annotated with the detection outcome."""
    out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()

    if result is not None:
        for hole in result.holes[1:]:  # secondary candidates
            cv2.circle(
                out, (int(hole.x_px), int(hole.y_px)),
                max(3, int(hole.diameter_px / 2)), _YELLOW, 1,
            )
        best = result.best
        if best is not None:
            cx, cy = int(best.x_px), int(best.y_px)
            radius = max(4, int(best.diameter_px / 2))
            cv2.circle(out, (cx, cy), radius, _GREEN, 2)
            cv2.line(out, (cx - radius - 8, cy), (cx + radius + 8, cy), _GREEN, 1)
            cv2.line(out, (cx, cy - radius - 8), (cx, cy + radius + 8), _GREEN, 1)
            cv2.putText(
                out,
                f"({best.x_px:.1f}, {best.y_px:.1f})  conf {best.confidence:.2f}",
                (max(4, cx - radius), max(18, cy - radius - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _GREEN, 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                out, "NO HOLE", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, _RED, 2, cv2.LINE_AA,
            )

    if label:
        cv2.putText(
            out, label, (12, out.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA,
        )
    return out

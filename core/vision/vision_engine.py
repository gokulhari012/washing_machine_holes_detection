"""Per-camera strategy context for hole detection + shared overlay rendering.

``VisionEngine`` owns one independent :class:`HoleDetector` instance **per
camera** — each camera may run a different algorithm with its own
parameters and its own common judgement thresholds (confidence, expected
hole count, position tolerance). ``apply_config`` hot-swaps every camera's
strategy at once from a full ``detection.json`` dict (atomic: a malformed
block for one camera leaves every camera's previous, working strategy in
place); ``apply_camera_config`` hot-swaps a single camera, for the Detection
page's per-camera "Save & Apply". ``detect()``/``debug_stages()`` take a
``camera_index`` and run that camera's strategy, serialised on a per-camera
lock only when that strategy declares itself not thread-safe (e.g. a GPU
model whose inference is not re-entrant) — classical CV strategies are
stateless per call and safely run 4 cameras' images in parallel with no
locking at all.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Any, Iterable

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


def migrate_legacy_detection_config(
    detection_config: dict[str, Any], camera_indices: Iterable[int]
) -> dict[str, Any]:
    """Upgrade a pre-per-camera ``detection.json`` (one flat block shared by
    every camera) into the current ``{"cameras": {"<index>": {...}}}`` shape
    by cloning that one block onto every camera index — preserves the old
    all-cameras-share-one-config behaviour exactly, so nothing changes for a
    station that has not yet re-tuned any camera individually.

    A no-op (returns *detection_config* unchanged) if it is already in the
    per-camera shape.
    """
    if "cameras" in detection_config:
        return detection_config
    return {"cameras": {str(index): copy.deepcopy(detection_config) for index in camera_indices}}


@dataclass
class _CameraDetector:
    detector: HoleDetector
    common: dict[str, Any]
    detect_lock: threading.Lock | None  # only set when the strategy is not thread-safe


class VisionEngine:
    """Facade the rest of the application talks to for detection."""

    def __init__(self, detection_config: dict[str, Any]) -> None:
        self._swap_lock = threading.Lock()  # protects the per-camera dict itself
        self._cameras: dict[int, _CameraDetector] = {}
        self.apply_config(detection_config)

    # ------------------------------------------------------------- configure
    def apply_config(self, detection_config: dict[str, Any]) -> None:
        """(Re)build every camera's strategy from a full detection.json dict.

        All-or-nothing: every camera's block is validated and built *before*
        any of them replace the live state, so a malformed block for one
        camera never leaves the others half-updated.

        Raises:
            ConfigurationError: no "cameras" block, or a camera's block names
                an unknown detector.
            DetectionError: a strategy rejected its parameters (bad template...).
        """
        cameras_config = detection_config.get("cameras")
        if not cameras_config:
            raise ConfigurationError("detection config has no 'cameras' block")

        built = {
            int(index): self._build_camera(camera_config)
            for index, camera_config in cameras_config.items()
        }
        with self._swap_lock:
            self._cameras = built
        logger.info(
            "Vision engine strategies -> %s",
            {index: camera.detector.name for index, camera in built.items()},
        )

    def apply_camera_config(self, camera_index: int, camera_config: dict[str, Any]) -> None:
        """Hot-swap a single camera's strategy (Detection page per-camera save).

        Raises:
            ConfigurationError: unknown detector name.
            DetectionError: strategy rejected its parameters.
        """
        camera = self._build_camera(camera_config)
        with self._swap_lock:
            self._cameras[camera_index] = camera
        logger.info("Vision engine strategy for camera %d -> %s", camera_index, camera.detector.name)

    @staticmethod
    def _build_camera(camera_config: dict[str, Any]) -> _CameraDetector:
        active_name = str(camera_config.get("active_detector", "opencv")).lower()
        try:
            detector_type = DetectorType(active_name)
        except ValueError as exc:
            raise ConfigurationError(f"Unknown active_detector: {active_name!r}") from exc

        params = dict(camera_config.get(detector_type.value, {}))
        detector = _REGISTRY[detector_type](params)
        common = dict(camera_config.get("common", {}))
        detect_lock = None if detector.thread_safe else threading.Lock()
        return _CameraDetector(detector=detector, common=common, detect_lock=detect_lock)

    def _camera(self, camera_index: int) -> _CameraDetector:
        with self._swap_lock:
            camera = self._cameras.get(camera_index)
        if camera is None:
            raise ConfigurationError(f"No detection configuration for camera {camera_index}")
        return camera

    # ------------------------------------------------------------ properties
    def active_detector_name(self, camera_index: int) -> str:
        return self._camera(camera_index).detector.name

    def confidence_threshold(self, camera_index: int) -> float:
        return float(self._camera(camera_index).common.get("confidence_threshold", 0.6))

    def expected_hole_count(self, camera_index: int) -> int:
        return int(self._camera(camera_index).common.get("expected_hole_count", 1))

    def position_tolerance_mm(self, camera_index: int) -> float:
        """<= 0 disables the position tolerance check."""
        return float(self._camera(camera_index).common.get("position_tolerance_mm", 0.0))

    # ---------------------------------------------------------------- detect
    def detect(self, image: np.ndarray, camera_index: int) -> DetectionResult:
        """Run *camera_index*'s strategy; candidates below its own confidence
        threshold are dropped so callers only ever see viable holes.

        Raises:
            ConfigurationError: no detector configured for this camera.
            DetectionError
        """
        camera = self._camera(camera_index)
        if camera.detect_lock is None:
            result = camera.detector.detect(image)
        else:
            with camera.detect_lock:
                result = camera.detector.detect(image)

        threshold = float(camera.common.get("confidence_threshold", 0.6))
        result.holes = [hole for hole in result.holes if hole.confidence >= threshold]
        return result

    def debug_stages(self, image: np.ndarray, camera_index: int) -> dict[str, np.ndarray]:
        """Intermediate mask/edges from *camera_index*'s strategy; ``{}`` when
        it has none to show (see :meth:`HoleDetector.debug_stages`).

        Raises:
            ConfigurationError: no detector configured for this camera.
            DetectionError
        """
        return self._camera(camera_index).detector.debug_stages(image)


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


_EDGE_CYAN = (230, 210, 60)
_CONTOUR_ORANGE = (0, 150, 255)


def draw_debug_overlay(frame: np.ndarray, stages: dict[str, np.ndarray]) -> np.ndarray:
    """Composite a detector's intermediate ``mask``/``edges`` onto *frame*.

    Every contour the mask currently offers is drawn (not just the ones that
    survive the detector's gates) — this is the "what is it reacting to right
    now" view, complementary to :func:`draw_detection_overlay`'s "what did it
    decide" view. ``stages`` is whatever :meth:`HoleDetector.debug_stages`
    returned; missing keys are simply skipped.
    """
    out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()

    edges = stages.get("edges")
    if edges is not None:
        out[edges > 0] = _EDGE_CYAN

    mask = stages.get("mask")
    contour_count = 0
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, _CONTOUR_ORANGE, 1)
        contour_count = len(contours)

    if mask is None and edges is None:
        cv2.putText(
            out, "No debug view for this strategy", (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            out, f"{contour_count} contour(s)", (12, out.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA,
        )
    return out

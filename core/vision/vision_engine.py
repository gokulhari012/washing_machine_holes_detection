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

Each camera's ``common`` block may also set ``normalize_image`` (default
off): when true, ``detect``/``debug_stages`` run the frame through
:func:`core.vision.normalization.normalize_image` (a min-max contrast
stretch) before handing it to the strategy. It is a per-camera choice, not a
global one, because turning it on changes what a strategy's own contrast
thresholds mean.
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
from core.vision.normalization import normalize_image
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
    # The raw per-camera block this was built from, kept verbatim so the
    # Detection page can render what the engine is *running* rather than what
    # detection.json says: a machine-model switch hot-swaps strategies
    # without persisting (see MachineModelService.apply_profile), so the file
    # and the live engine legitimately disagree after one.
    config: dict[str, Any]


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
        return _CameraDetector(
            detector=detector,
            common=common,
            detect_lock=detect_lock,
            config=copy.deepcopy(camera_config),
        )

    def _camera(self, camera_index: int) -> _CameraDetector:
        with self._swap_lock:
            camera = self._cameras.get(camera_index)
        if camera is None:
            raise ConfigurationError(f"No detection configuration for camera {camera_index}")
        return camera

    # ------------------------------------------------------------ properties
    def camera_config(self, camera_index: int) -> dict[str, Any] | None:
        """The per-camera detection block currently live for *camera_index*.

        A deep copy, so a caller editing it cannot reach into the running
        strategy. Returns None for a camera the engine has no strategy for.
        Read this — not detection.json — anywhere the UI shows what the
        engine is actually using: ``apply_config``/``apply_camera_config``
        are "preview, don't persist" entry points, so after a machine-model
        switch or a Detection-page "Test" the file is the older document.
        """
        with self._swap_lock:
            camera = self._cameras.get(camera_index)
        return copy.deepcopy(camera.config) if camera is not None else None

    def active_detector_name(self, camera_index: int) -> str:
        return self._camera(camera_index).detector.name

    def confidence_threshold(self, camera_index: int) -> float:
        return float(self._camera(camera_index).common.get("confidence_threshold", 0.6))

    def expected_hole_count(self, camera_index: int) -> int:
        return int(self._camera(camera_index).common.get("expected_hole_count", 1))

    def position_tolerance_mm(self, camera_index: int) -> float:
        """<= 0 disables the position tolerance check."""
        return float(self._camera(camera_index).common.get("position_tolerance_mm", 0.0))

    def normalize_enabled(self, camera_index: int) -> bool:
        """Whether *camera_index* runs frames through :func:`normalize_image`
        before its strategy sees them. Off unless ``common.normalize_image``
        is set — see the module docstring on why this isn't on by default."""
        return bool(self._camera(camera_index).common.get("normalize_image", False))

    # ---------------------------------------------------------------- detect
    def detect(self, image: np.ndarray, camera_index: int) -> DetectionResult:
        """Run *camera_index*'s strategy; candidates below its own confidence
        threshold are dropped so callers only ever see viable holes.

        Raises:
            ConfigurationError: no detector configured for this camera.
            DetectionError
        """
        camera = self._camera(camera_index)
        if camera.common.get("normalize_image", False):
            image = normalize_image(image)
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
        camera = self._camera(camera_index)
        if camera.common.get("normalize_image", False):
            image = normalize_image(image)
        return camera.detector.debug_stages(image)


# --------------------------------------------------------------------------- #
# Overlay rendering (dashboard panels, calibration live test, saved NG images)
# --------------------------------------------------------------------------- #
_GREEN = (80, 220, 80)
_YELLOW = (60, 200, 240)
_RED = (70, 70, 230)
_WHITE = (235, 235, 235)
_OUTLINE = (0, 0, 0)

# Annotation size is proportional to the frame, not fixed in pixels: the
# station's 12-20 MP frames are shown fitted into a few-hundred-pixel panel
# (and saved as full-resolution PNGs), so a fixed 1-2 px stroke and a 0.55
# font shrink to invisible. _OVERLAY_REFERENCE_PX is the shorter image side
# at which the base sizes below are drawn 1:1; larger frames scale up.
_OVERLAY_REFERENCE_PX = 480.0
_BASE_STROKE = 3
_BASE_FONT = 0.6


def _overlay_scale(image: np.ndarray) -> float:
    return max(1.0, min(image.shape[:2]) / _OVERLAY_REFERENCE_PX)


def _put_outlined_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """Text with a dark outline, so it stays legible on bright and dark metal alike.

    *origin* is the baseline-left point, clamped so the text never runs off
    the frame (a hole near an edge would otherwise lose its label).
    """
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    margin = thickness + 2
    x = min(max(margin, origin[0]), max(margin, image.shape[1] - width - margin))
    y = min(max(height + margin, origin[1]), max(height + margin, image.shape[0] - baseline - margin))
    for stroke, ink in ((thickness + max(2, thickness), _OUTLINE), (thickness, color)):
        cv2.putText(
            image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, ink, stroke, cv2.LINE_AA
        )


def draw_detection_overlay(
    frame: np.ndarray,
    result: DetectionResult | None,
    label: str = "",
) -> np.ndarray:
    """Return a BGR copy of *frame* annotated with the detection outcome.

    Every candidate carries its confidence: the judged hole (green) as
    ``conf 0.87`` above its pixel position, secondary candidates (yellow) as
    the bare value — so an operator can see *why* one hole won over another.
    Stroke width and text size follow the frame size (see
    :data:`_OVERLAY_REFERENCE_PX`).
    """
    out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    scale = _overlay_scale(out)
    stroke = max(_BASE_STROKE, round(_BASE_STROKE * scale))
    font = _BASE_FONT * scale
    text_stroke = max(1, round(scale))
    line_gap = int(28 * scale)

    if result is not None:
        for hole in result.holes[1:]:  # secondary candidates
            cx, cy = int(hole.x_px), int(hole.y_px)
            radius = max(3, int(hole.diameter_px / 2))
            cv2.circle(out, (cx, cy), radius, _YELLOW, max(2, stroke * 2 // 3), cv2.LINE_AA)
            # Below the circle: the judged hole's label sits above its own, so
            # two nearby candidates don't write over each other.
            _put_outlined_text(
                out, f"{hole.confidence:.2f}", (cx - radius, cy + radius + stroke + line_gap),
                _YELLOW, font * 0.9, text_stroke,
            )
        best = result.best
        if best is not None:
            cx, cy = int(best.x_px), int(best.y_px)
            radius = max(4, int(best.diameter_px / 2))
            arm = radius + int(12 * scale)
            cv2.circle(out, (cx, cy), radius, _GREEN, stroke, cv2.LINE_AA)
            cv2.line(out, (cx - arm, cy), (cx + arm, cy), _GREEN, max(2, stroke // 2), cv2.LINE_AA)
            cv2.line(out, (cx, cy - arm), (cx, cy + arm), _GREEN, max(2, stroke // 2), cv2.LINE_AA)
            text_x = cx - radius
            text_y = cy - radius - stroke - int(10 * scale)
            _put_outlined_text(
                out, f"conf {best.confidence:.2f}", (text_x, text_y - line_gap),
                _GREEN, font * 1.15, text_stroke + 1,
            )
            _put_outlined_text(
                out, f"({best.x_px:.1f}, {best.y_px:.1f}) px", (text_x, text_y),
                _GREEN, font, text_stroke,
            )
        else:
            _put_outlined_text(
                out, "NO HOLE", (int(12 * scale), int(40 * scale)),
                _RED, font * 1.6, text_stroke + 1,
            )

    if label:
        _put_outlined_text(
            out, label, (int(12 * scale), out.shape[0] - int(12 * scale)),
            _WHITE, font, text_stroke,
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

    scale = _overlay_scale(out)
    font = _BASE_FONT * scale
    text_stroke = max(1, round(scale))

    edges = stages.get("edges")
    if edges is not None:
        out[edges > 0] = _EDGE_CYAN

    mask = stages.get("mask")
    contour_count = 0
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, _CONTOUR_ORANGE, max(1, round(scale)))
        contour_count = len(contours)

    if mask is None and edges is None:
        _put_outlined_text(
            out, "No debug view for this strategy", (int(12 * scale), int(40 * scale)),
            (210, 210, 210), font, text_stroke,
        )
    else:
        _put_outlined_text(
            out, f"{contour_count} contour(s)", (int(12 * scale), out.shape[0] - int(12 * scale)),
            _WHITE, font, text_stroke,
        )
    return out

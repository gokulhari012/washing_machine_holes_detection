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

Downscale
---------
``common.downscale_enabled`` + ``common.downscale_factor`` (default off, 2.0)
shrink the frame by that factor on each side before the strategy sees it —
the station's 12-20 MP frames make detection the slow part of a cycle, and
most bores are far larger than the precision the PLC needs. Two rules make it
safe to switch on without re-tuning anything:

- **results come back in full-resolution pixels.** Hole centres, diameters
  and search regions are mapped back onto the frame that was passed in, so
  ``x_px``, the calibration's px->mm mapping, the overlay and the PLC position
  are exactly what they would be without it — only less precise.
- **parameters keep their full-resolution meaning.** The strategy is built
  from its block converted by :meth:`HoleDetector.downscaled_params` (diameter
  gates, kernel sizes, template scales divided by the factor), so "Min Hole
  Diameter 80 px" still means 80 px of the camera's own image.

A factor below 1 would be an *up*scale and is treated as off (with a
warning) rather than raised, because ``VisionEngine`` is built at startup and
a typo in a tuning field must not stop the station booting.
``debug_stages`` returns its images scaled back up to the input's size, since
``draw_debug_overlay`` composites them pixel-for-pixel onto the full frame.

NG re-run
---------
``common.ng_rerun_enabled`` + ``common.ng_rerun_count`` are judged here only
in the sense that the engine stores and serves them
(:meth:`VisionEngine.ng_rerun_count`); the re-run itself — a fresh capture
and a fresh detection — is the pipeline's job (``InspectionService``), since
re-detecting the same frame would only ever repeat the same answer.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, replace
from typing import Any, Iterable

import cv2
import numpy as np

from core.logging import get_logger
from core.utilities.enums import DetectorType, LogSource
from core.utilities.exceptions import ConfigurationError
from core.vision.dark_hole_detector import DarkHoleDetector
from core.vision.detection_result import DetectionResult, SearchRegion
from core.vision.detector_base import HoleDetector
from core.vision.normalization import normalize_image
from core.vision.opencv_hole_detector import OpenCVHoleDetector
from core.vision.template_matching_detector import TemplateMatchingDetector
from core.vision.template_opencv_detector import TemplateOpenCVDetector
from core.vision.yolo_hole_detector import YoloHoleDetector

logger = get_logger(LogSource.VISION)

_REGISTRY: dict[DetectorType, type[HoleDetector]] = {
    DetectorType.OPENCV: OpenCVHoleDetector,
    DetectorType.TEMPLATE_MATCHING: TemplateMatchingDetector,
    DetectorType.YOLO: YoloHoleDetector,
    DetectorType.DARK_HOLE: DarkHoleDetector,
    DetectorType.TEMPLATE_OPENCV: TemplateOpenCVDetector,
}

#: Largest accepted downscale factor — beyond this a bore is a handful of pixels.
MAX_DOWNSCALE_FACTOR = 16.0
DEFAULT_DOWNSCALE_FACTOR = 2.0

#: Most re-runs an NG may trigger: each is a full capture plus detection.
MAX_NG_RERUNS = 10


def _downscale_factor(common: dict[str, Any]) -> float:
    """The effective factor for one camera's ``common`` block; 1.0 = off."""
    if not common.get("downscale_enabled", False):
        return 1.0
    raw = common.get("downscale_factor", DEFAULT_DOWNSCALE_FACTOR)
    try:
        factor = float(raw)
    except (TypeError, ValueError):
        factor = 0.0
    if factor < 1.0:
        logger.warning("downscale_factor %r is not >= 1 - downscale is off for this camera", raw)
        return 1.0
    return min(factor, MAX_DOWNSCALE_FACTOR)


def _ng_rerun_count(common: dict[str, Any]) -> int:
    """How many extra capture+detect runs an NG gets; 0 = off."""
    if not common.get("ng_rerun_enabled", False):
        return 0
    try:
        count = int(common.get("ng_rerun_count", 1))
    except (TypeError, ValueError):
        return 0
    return max(0, min(count, MAX_NG_RERUNS))


def _shrink(image: np.ndarray, factor: float) -> tuple[np.ndarray, float, float]:
    """*image* reduced by *factor* per side, plus the exact x/y ratios used.

    The sizes are rounded to whole pixels, so the true ratio on each axis is
    returned rather than *factor* — mapping back by *factor* itself would put
    a hole near the far edge of a 5471 px frame a pixel or two off.
    """
    if factor <= 1.0:
        return image, 1.0, 1.0
    height, width = image.shape[:2]
    small_w = max(1, int(round(width / factor)))
    small_h = max(1, int(round(height / factor)))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return small, width / small_w, height / small_h


def _grow_result(result: DetectionResult, sx: float, sy: float) -> None:
    """Map a result found on a shrunk frame back onto the original, in place.

    A pixel *centre* at ``u`` on the small frame covers ``[u*s, (u+1)*s)`` on
    the original, whose centre is ``(u + 0.5)*s - 0.5``; region corners are
    edges, not centres, and scale plainly.
    """
    mean = (sx + sy) / 2.0
    result.holes = [
        replace(
            hole,
            x_px=(hole.x_px + 0.5) * sx - 0.5,
            y_px=(hole.y_px + 0.5) * sy - 0.5,
            diameter_px=hole.diameter_px * mean,
        )
        for hole in result.holes
    ]
    result.regions = [
        SearchRegion(
            x=int(round(region.x * sx)),
            y=int(round(region.y * sy)),
            width=int(round(region.width * sx)),
            height=int(round(region.height * sy)),
            score=region.score,
        )
        for region in result.regions
    ]


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
    downscale: float = 1.0  # effective factor; 1.0 = full resolution


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

        detector_cls = _REGISTRY[detector_type]
        common = dict(camera_config.get("common", {}))
        params = detector_cls.params_from_camera_config(camera_config)
        downscale = _downscale_factor(common)
        if downscale > 1.0:
            params = detector_cls.downscaled_params(params, downscale)
        detector = detector_cls(params)
        detect_lock = None if detector.thread_safe else threading.Lock()
        return _CameraDetector(
            detector=detector,
            common=common,
            detect_lock=detect_lock,
            config=copy.deepcopy(camera_config),
            downscale=downscale,
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

    def downscale_factor(self, camera_index: int) -> float:
        """Per-side shrink *camera_index* detects at; 1.0 when downscale is
        off (see the module docstring)."""
        return self._camera(camera_index).downscale

    def ng_rerun_count(self, camera_index: int) -> int:
        """Extra capture+detect runs an NG on *camera_index* gets before it
        is reported; 0 when "Required another run" is off."""
        return _ng_rerun_count(self._camera(camera_index).common)

    # ---------------------------------------------------------------- detect
    def detect(self, image: np.ndarray, camera_index: int) -> DetectionResult:
        """Run *camera_index*'s strategy; candidates below its own confidence
        threshold are dropped so callers only ever see viable holes.

        Coordinates are always in *image*'s own pixels, downscaled or not.

        Raises:
            ConfigurationError: no detector configured for this camera.
            DetectionError
        """
        camera = self._camera(camera_index)
        small, sx, sy = _shrink(image, camera.downscale)
        if camera.common.get("normalize_image", False):
            small = normalize_image(small)
        if camera.detect_lock is None:
            result = camera.detector.detect(small)
        else:
            with camera.detect_lock:
                result = camera.detector.detect(small)
        if small is not image:
            _grow_result(result, sx, sy)

        threshold = float(camera.common.get("confidence_threshold", 0.6))
        result.holes = [hole for hole in result.holes if hole.confidence >= threshold]
        return result

    def debug_stages(self, image: np.ndarray, camera_index: int) -> dict[str, np.ndarray]:
        """Intermediate mask/edges from *camera_index*'s strategy; ``{}`` when
        it has none to show (see :meth:`HoleDetector.debug_stages`). Always
        *image*-sized, downscaled or not.

        Raises:
            ConfigurationError: no detector configured for this camera.
            DetectionError
        """
        camera = self._camera(camera_index)
        small, _sx, _sy = _shrink(image, camera.downscale)
        if camera.common.get("normalize_image", False):
            small = normalize_image(small)
        stages = camera.detector.debug_stages(small)
        if small is image:
            return stages
        height, width = image.shape[:2]
        return {
            name: cv2.resize(stage, (width, height), interpolation=cv2.INTER_NEAREST)
            for name, stage in stages.items()
        }


# --------------------------------------------------------------------------- #
# Overlay rendering (dashboard panels, calibration live test, saved NG images)
# --------------------------------------------------------------------------- #
_GREEN = (80, 220, 80)
_YELLOW = (60, 200, 240)
_RED = (70, 70, 230)
_WHITE = (235, 235, 235)
_OUTLINE = (0, 0, 0)
_ROI_COLOR = (255, 170, 0)   # BGR: azure — the analysed region's frame
_ROI_CENTER = (255, 0, 255)  # BGR: magenta — the (0, 0) mm reference point
_REGION_COLOR = (0, 140, 255)  # BGR: orange — where a two-stage strategy searched

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
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Return a BGR copy of *frame* annotated with the detection outcome.

    Every candidate carries its confidence: the judged hole (green) as
    ``conf 0.87`` above its pixel position, secondary candidates (yellow) as
    the bare value — so an operator can see *why* one hole won over another.
    Stroke width and text size follow the frame size (see
    :data:`_OVERLAY_REFERENCE_PX`).

    ``roi`` — ``(x0, y0, x1, y1)`` in *frame*'s pixels, as
    ``core.camera.roi_rect`` returns it — means *frame* is the whole
    (uncropped) image and *result* was detected in that region of it: the
    region is outlined, its centre (the point every reported x_mm/y_mm is
    measured from) is marked — unlabelled, so it never writes over a hole's
    labels — and the holes are shifted into place. The px
    label keeps the hole's own ROI-relative coordinates, the ones stored.

    ``result.regions`` (``template_opencv``'s template matches) are outlined
    in orange with their match score just above the box, under the holes — so a "NO HOLE" with a
    box drawn means the template was found but held no hole, and one without
    means the template itself was not found.
    """
    out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    scale = _overlay_scale(out)
    stroke = max(_BASE_STROKE, round(_BASE_STROKE * scale))
    font = _BASE_FONT * scale
    text_stroke = max(1, round(scale))
    line_gap = int(28 * scale)

    ox = oy = 0
    if roi is not None:
        x0, y0, x1, y1 = roi
        ox, oy = x0, y0
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), _ROI_COLOR, stroke, cv2.LINE_AA)
        _put_outlined_text(
            out, "ROI", (x0 + stroke + 4, y0 - stroke - 4), _ROI_COLOR, font, text_stroke
        )
        mx, my = (x0 + x1) // 2, (y0 + y1) // 2
        arm = int(18 * scale)
        thin = max(2, stroke // 2)
        cv2.line(out, (mx - arm, my), (mx + arm, my), _ROI_CENTER, thin, cv2.LINE_AA)
        cv2.line(out, (mx, my - arm), (mx, my + arm), _ROI_CENTER, thin, cv2.LINE_AA)
        cv2.circle(out, (mx, my), max(3, arm // 4), _ROI_CENTER, thin, cv2.LINE_AA)
        # The mark is deliberately unlabelled: a hole is usually near the
        # centre, and a caption here wrote over the hole's own labels.

    if result is not None:
        for region in result.regions:
            rx, ry = region.x + ox, region.y + oy
            cv2.rectangle(
                out, (rx, ry), (rx + region.width - 1, ry + region.height - 1),
                _REGION_COLOR, max(2, stroke * 2 // 3), cv2.LINE_AA,
            )
            # Just above the top edge: inside the box it wrote over the judged
            # hole's conf/px labels, which sit above the hole within the match.
            _put_outlined_text(
                out, f"Template {region.score:.2f}",
                (rx, ry - stroke - 4),
                _REGION_COLOR, font * 0.9, text_stroke,
            )
        for hole in result.holes[1:]:  # secondary candidates
            cx, cy = int(hole.x_px) + ox, int(hole.y_px) + oy
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
            cx, cy = int(best.x_px) + ox, int(best.y_px) + oy
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

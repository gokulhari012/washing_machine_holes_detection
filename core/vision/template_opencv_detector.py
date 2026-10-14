"""Two-stage detector: template matching finds the region, OpenCV finds the hole.

``template_opencv`` answers "is there a hole *inside the feature the template
shows*", which neither stage answers alone:

- ``template_matching`` on its own reports the template's centre as the hole,
  so its position is only as good as how tightly the template was cropped and
  it measures nothing about the bore itself;
- ``opencv`` on its own searches the whole ROI, so any dark round blob in the
  frame — a fixing hole, a shadow, a stamped recess — competes with the real
  one.

Here the template is cropped around a distinctive feature (a boss, a bracket,
the area round the drain hole), and the hole is measured by the OpenCV
contour pipeline **only inside the matched area**. A hole outside every
match is never reported, however round it is.

Pipeline
--------
1. :meth:`TemplateMatchingDetector.find_matches` over the whole frame —
   exactly the ``template_matching`` stage, with that block's parameters
   (template, threshold, method, scales, max matches, size gate on the
   *matched area*).
2. The best ``max_regions`` matches (default 1) become search regions, each
   grown by ``region_padding_px`` on every side and clamped to the frame.
3. :class:`OpenCVHoleDetector` runs on each region's crop, with the
   ``opencv`` block's parameters, and every hole it finds is shifted back
   into frame coordinates.
4. A hole found twice (two padded regions overlapping) is reported once.

No match means no hole: an empty result, i.e. NG — the template not being
there is a legitimate "this is not the part we expected", not an algorithm
failure. A missing template image is a ``DetectionError`` at detect time,
the same as for ``template_matching`` (see that module and CLAUDE.md §9).

Parameters
----------
The two stages read the camera's **own** ``template_matching`` and ``opencv``
blocks (see :meth:`params_from_camera_config`) — the ones the Detection page
already edits for those strategies — so tuning either stage on its own and
then switching to ``template_opencv`` needs no copying. The
``template_opencv`` block holds only what is specific to the combination:
``max_regions`` and ``region_padding_px``.

A hole's ``confidence`` is the OpenCV stage's own score. The template score
is a *gate* (``match_threshold``), not a factor: multiplying the two would
push a well-found hole in a well-matched region under
``common.confidence_threshold`` for no reason anyone could see. Each match is
kept on the result as a :class:`SearchRegion` (with its score) and drawn on
the overlay, so the operator can see where the template was found.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np

from core.utilities.exceptions import DetectionError
from core.vision.detection_result import DetectionResult, Hole, SearchRegion
from core.vision.detector_base import HoleDetector, shrink_length
from core.vision.opencv_hole_detector import OpenCVHoleDetector
from core.vision.template_matching_detector import TemplateMatchingDetector

DEFAULT_MAX_REGIONS = 1

#: A region smaller than this on either side cannot hold a contour worth fitting.
_MIN_REGION_PX = 3


class TemplateOpenCVDetector(HoleDetector):
    """Template match locates the region; OpenCV contours find the hole in it."""

    name: ClassVar[str] = "template_opencv"
    # Both stages are stateless per call (see their own modules).
    thread_safe: ClassVar[bool] = True

    #: keys of the nested stage blocks in this detector's params
    TEMPLATE_STAGE: ClassVar[str] = TemplateMatchingDetector.name
    HOLE_STAGE: ClassVar[str] = OpenCVHoleDetector.name

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        # Built before super().__init__, which calls configure().
        self._locator = TemplateMatchingDetector()
        self._finder = OpenCVHoleDetector()
        super().__init__(params)

    # ------------------------------------------------------- engine hooks
    @classmethod
    def params_from_camera_config(cls, camera_config: dict[str, Any]) -> dict[str, Any]:
        """This strategy's own block, plus the camera's ``template_matching``
        and ``opencv`` blocks nested under those names as the two stages."""
        params = dict(camera_config.get(cls.name, {}))
        params[cls.TEMPLATE_STAGE] = dict(camera_config.get(cls.TEMPLATE_STAGE, {}))
        params[cls.HOLE_STAGE] = dict(camera_config.get(cls.HOLE_STAGE, {}))
        return params

    @classmethod
    def downscaled_params(cls, params: dict[str, Any], factor: float) -> dict[str, Any]:
        """Each stage converted by its own rules, plus the region padding."""
        scaled = super().downscaled_params(params, factor)
        scaled[cls.TEMPLATE_STAGE] = TemplateMatchingDetector.downscaled_params(
            dict(params.get(cls.TEMPLATE_STAGE, {})), factor
        )
        scaled[cls.HOLE_STAGE] = OpenCVHoleDetector.downscaled_params(
            dict(params.get(cls.HOLE_STAGE, {})), factor
        )
        if "region_padding_px" in scaled:
            scaled["region_padding_px"] = shrink_length(scaled["region_padding_px"], factor)
        return scaled

    # ---------------------------------------------------------- configure
    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        self._locator.configure(dict(self._params.get(self.TEMPLATE_STAGE, {})))
        self._finder.configure(dict(self._params.get(self.HOLE_STAGE, {})))

    # ------------------------------------------------------------- detect
    def detect(self, image: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        regions = self._regions(image)
        holes: list[Hole] = []
        for region in regions:
            crop = image[region.y : region.y + region.height, region.x : region.x + region.width]
            found = self._finder.detect(crop)
            holes.extend(
                replace(hole, x_px=hole.x_px + region.x, y_px=hole.y_px + region.y)
                for hole in found.holes
            )
        return DetectionResult(
            holes=_without_duplicates(holes),
            processing_ms=(time.perf_counter() - started) * 1000.0,
            regions=regions,
        )

    def debug_stages(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """The OpenCV stage's mask and edges, inside the matched regions only.

        Frame-sized, black outside every region — that is precisely the part
        of the frame this strategy never looks at, so showing anything there
        would suggest it reacts to features it ignores. With no match at all
        both are blank: switch Active Detector to ``template_matching`` to see
        what the template stage responds to.
        """
        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        edges = np.zeros((height, width), np.uint8)
        for region in self._regions(image):
            y0, y1 = region.y, region.y + region.height
            x0, x1 = region.x, region.x + region.width
            stages = self._finder.debug_stages(image[y0:y1, x0:x1])
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], stages["mask"])
            edges[y0:y1, x0:x1] = np.maximum(edges[y0:y1, x0:x1], stages["edges"])
        return {"mask": mask, "edges": edges}

    # ----------------------------------------------------------- internal
    def _regions(self, image: np.ndarray) -> list[SearchRegion]:
        """The best ``max_regions`` template matches, padded and clamped.

        Raises:
            DetectionError: the template stage failed (no template, template
                larger than the frame, ...), prefixed so the operator knows
                which of the two stages is misconfigured.
        """
        try:
            matches = self._locator.find_matches(image)
        except DetectionError as exc:
            raise DetectionError(f"template_opencv — template stage: {exc}") from exc

        max_regions = max(1, int(self._params.get("max_regions", DEFAULT_MAX_REGIONS)))
        padding = max(0, int(round(float(self._params.get("region_padding_px", 0) or 0))))
        height, width = image.shape[:2]
        regions: list[SearchRegion] = []
        for match in matches[:max_regions]:
            x0, y0 = max(0, match.x - padding), max(0, match.y - padding)
            x1 = min(width, match.x + match.width + padding)
            y1 = min(height, match.y + match.height + padding)
            if x1 - x0 < _MIN_REGION_PX or y1 - y0 < _MIN_REGION_PX:
                continue
            regions.append(SearchRegion(x0, y0, x1 - x0, y1 - y0, match.score))
        return regions


def _without_duplicates(holes: list[Hole]) -> list[Hole]:
    """Best first, dropping any hole centred inside a better one already kept.

    Only overlapping (padded) regions can produce a duplicate: the template
    stage's own suppression keeps two matches at least a template apart.
    """
    kept: list[Hole] = []
    for hole in sorted(holes, key=lambda candidate: candidate.confidence, reverse=True):
        if all(
            (hole.x_px - other.x_px) ** 2 + (hole.y_px - other.y_px) ** 2
            > (max(hole.diameter_px, other.diameter_px) / 2.0) ** 2
            for other in kept
        ):
            kept.append(hole)
    return kept

"""Two-stage detector: template matching finds the region, caliper circle fitting finds the hole.

``template_caliper_based_circle_fitting`` is ``template_opencv`` with
:class:`~core.vision.caliper_circle_detector.CaliperCircleDetector` as the
hole stage: the template is cropped around a distinctive feature, and inside
each matched area the calipers measure the bore's rim and a robust circle fit
gives its centre. The seed blob that aims the calipers is then chosen inside
the match only, so a bigger dark round feature elsewhere in the ROI can no
longer pull the calipers onto itself — the one way ``caliper_based_circle_fitting``
on its own goes wrong.

Caliper fitting reports one hole per image it is given, so here that is one
hole per matched region: "Regions Searched" (``max_regions``) is also how many
holes this strategy can report. A bore whose rim runs past the padded match is
measured from the part of the rim inside it — the fit is robust to that, but
"Region Padding" is the setting that gives the calipers the whole rim back.

The pipeline and the ``max_regions``/``region_padding_px`` block are shared
with ``template_opencv`` and documented in
:mod:`core.vision.template_region_detector`. Stage 2 reads the camera's own
``caliper_based_circle_fitting`` block, and a hole's confidence is that stage's
score (the inlier ratio).
"""

from __future__ import annotations

from typing import ClassVar

from core.vision.caliper_circle_detector import CaliperCircleDetector
from core.vision.detector_base import HoleDetector
from core.vision.template_region_detector import TemplateRegionDetector


class TemplateCaliperCircleDetector(TemplateRegionDetector):
    """Template match locates the region; caliper circle fitting finds the hole in it."""

    name: ClassVar[str] = "template_caliper_based_circle_fitting"
    HOLE_DETECTOR: ClassVar[type[HoleDetector]] = CaliperCircleDetector
    HOLE_STAGE: ClassVar[str] = CaliperCircleDetector.name
    DEBUG_KEYS: ClassVar[tuple[str, ...]] = ("mask", "edges")

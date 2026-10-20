"""Two-stage detector: template matching finds the region, blob analysis finds the hole.

``template_blob_analysis`` is ``template_opencv`` with
:class:`~core.vision.blob_analysis_detector.BlobAnalysisDetector` as the hole
stage: the template is cropped around a distinctive feature, and inside each
matched area the largest dark blob that is round enough is the hole. A blob
outside every match is never reported, however large or round it is — which
is exactly the weakness of ``blob_analysis`` on its own, where any bigger
dark round feature in the ROI wins.

Blob analysis reports one hole per image it is given, so here that is one
hole per matched region: "Regions Searched" (``max_regions``) is also how
many holes this strategy can report.

The pipeline and the ``max_regions``/``region_padding_px`` block are shared
with ``template_opencv`` and documented in
:mod:`core.vision.template_region_detector`. Stage 2 reads the camera's own
``blob_analysis`` block, and a hole's confidence is that stage's score (the
blob's circularity).
"""

from __future__ import annotations

from typing import ClassVar

from core.vision.blob_analysis_detector import BlobAnalysisDetector
from core.vision.detector_base import HoleDetector
from core.vision.template_region_detector import TemplateRegionDetector


class TemplateBlobAnalysisDetector(TemplateRegionDetector):
    """Template match locates the region; blob analysis finds the hole in it."""

    name: ClassVar[str] = "template_blob_analysis"
    HOLE_DETECTOR: ClassVar[type[HoleDetector]] = BlobAnalysisDetector
    HOLE_STAGE: ClassVar[str] = BlobAnalysisDetector.name
    DEBUG_KEYS: ClassVar[tuple[str, ...]] = ("mask",)

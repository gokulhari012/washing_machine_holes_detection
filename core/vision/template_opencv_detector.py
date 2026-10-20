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

The pipeline, the ``max_regions``/``region_padding_px`` block and the rule
that a hole's confidence is the OpenCV stage's own score are shared with
``template_blob_analysis`` and documented once, in
:mod:`core.vision.template_region_detector`. Stage 2 reads the camera's own
``opencv`` block.
"""

from __future__ import annotations

from typing import ClassVar

from core.vision.detector_base import HoleDetector
from core.vision.opencv_hole_detector import OpenCVHoleDetector
from core.vision.template_region_detector import TemplateRegionDetector


class TemplateOpenCVDetector(TemplateRegionDetector):
    """Template match locates the region; OpenCV contours find the hole in it."""

    name: ClassVar[str] = "template_opencv"
    HOLE_DETECTOR: ClassVar[type[HoleDetector]] = OpenCVHoleDetector
    HOLE_STAGE: ClassVar[str] = OpenCVHoleDetector.name
    DEBUG_KEYS: ClassVar[tuple[str, ...]] = ("mask", "edges")

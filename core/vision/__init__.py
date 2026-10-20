"""Vision layer: strategy interface, detectors, engine, result types."""

from core.vision.blob_analysis_detector import BlobAnalysisDetector
from core.vision.dark_hole_detector import DarkHoleDetector
from core.vision.detection_result import DetectionResult, Hole, SearchRegion
from core.vision.detector_base import HoleDetector
from core.vision.normalization import normalize_image
from core.vision.opencv_hole_detector import OpenCVHoleDetector
from core.vision.template_blob_analysis_detector import TemplateBlobAnalysisDetector
from core.vision.template_matching_detector import TemplateMatchingDetector
from core.vision.template_opencv_detector import TemplateOpenCVDetector
from core.vision.template_region_detector import TemplateRegionDetector
from core.vision.vision_engine import (
    VisionEngine,
    draw_debug_overlay,
    draw_detection_overlay,
    migrate_legacy_detection_config,
)
from core.vision.yolo_hole_detector import YoloHoleDetector

__all__ = [
    "BlobAnalysisDetector",
    "DarkHoleDetector",
    "DetectionResult",
    "Hole",
    "HoleDetector",
    "OpenCVHoleDetector",
    "SearchRegion",
    "TemplateBlobAnalysisDetector",
    "TemplateMatchingDetector",
    "TemplateOpenCVDetector",
    "TemplateRegionDetector",
    "VisionEngine",
    "YoloHoleDetector",
    "draw_debug_overlay",
    "draw_detection_overlay",
    "migrate_legacy_detection_config",
    "normalize_image",
]

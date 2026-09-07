"""Vision layer: strategy interface, detectors, engine, result types."""

from core.vision.dark_hole_detector import DarkHoleDetector
from core.vision.detection_result import DetectionResult, Hole
from core.vision.detector_base import HoleDetector
from core.vision.opencv_hole_detector import OpenCVHoleDetector
from core.vision.template_matching_detector import TemplateMatchingDetector
from core.vision.vision_engine import (
    VisionEngine,
    draw_debug_overlay,
    draw_detection_overlay,
    migrate_legacy_detection_config,
)
from core.vision.yolo_hole_detector import YoloHoleDetector

__all__ = [
    "DarkHoleDetector",
    "DetectionResult",
    "Hole",
    "HoleDetector",
    "OpenCVHoleDetector",
    "TemplateMatchingDetector",
    "VisionEngine",
    "YoloHoleDetector",
    "draw_debug_overlay",
    "draw_detection_overlay",
    "migrate_legacy_detection_config",
]

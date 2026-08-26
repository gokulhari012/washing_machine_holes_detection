"""Hole-detector strategy interface.

Swapping the detection algorithm (classical OpenCV, template matching, a
trained YOLO model, ...) means implementing this class and registering it in
``vision_engine._REGISTRY`` — no UI, PLC or service code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from core.vision.detection_result import DetectionResult


class HoleDetector(ABC):
    """Contract every detection strategy must fulfil."""

    #: strategy identifier, matches DetectorType values
    name: ClassVar[str] = "base"

    #: False when detect() must be serialised by the caller (e.g. GPU models
    #: whose inference is not re-entrant). Classical CV strategies are
    #: stateless per call and safely run 4 images in parallel.
    thread_safe: ClassVar[bool] = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params: dict[str, Any] = {}
        if params:
            self.configure(params)

    def configure(self, params: dict[str, Any]) -> None:
        """Adopt strategy-specific parameters (one block of detection.json).

        Subclasses override to validate/pre-compute and must call super().
        """
        self._params.update(params)

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Find hole candidates in *image* (BGR or grayscale ndarray).

        Returns:
            DetectionResult with candidates sorted best-first (may be empty).

        Raises:
            DetectionError: the algorithm itself failed (bad input, missing
                model/template) — distinct from "no hole found".
        """

    def debug_stages(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Intermediate images (e.g. ``mask``, ``edges``) for a debug overlay
        showing what the algorithm currently reacts to.

        Base implementation: ``{}`` — no debug view. Strategies built on a
        threshold mask / edge map override this (see ``OpenCVHoleDetector``,
        ``DarkHoleDetector``); strategies with no such intermediate
        representation (template matching, YOLO) leave it as-is.
        """
        return {}

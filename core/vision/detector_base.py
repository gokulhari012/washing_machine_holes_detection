"""Hole-detector strategy interface.

Swapping the detection algorithm (classical OpenCV, template matching, a
trained YOLO model, ...) means implementing this class and registering it in
``vision_engine._REGISTRY`` — no UI, PLC or service code changes.

Two class-level hooks let ``VisionEngine`` build a strategy without knowing
anything about it:

- :meth:`HoleDetector.params_from_camera_config` picks the strategy's
  parameters out of one camera's detection block. Almost every strategy reads
  the one block named like itself; a composite strategy (``template_opencv``)
  overrides it to gather the blocks of the stages it runs.
- :meth:`HoleDetector.downscaled_params` re-expresses those parameters for a
  frame shrunk by the camera's ``common.downscale_factor``, so every pixel
  parameter keeps meaning what it means at full resolution and switching
  downscale on never needs a re-tune. The base version converts the
  ``min/max_hole_diameter_px`` gates every strategy shares; a strategy with
  further pixel parameters (kernel sizes, template scales) extends it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from core.vision.detection_result import DetectionResult

#: The size gates every strategy carries, in pixels of the analysed frame.
DIAMETER_GATE_KEYS = ("min_hole_diameter_px", "max_hole_diameter_px")


def shrink_length(value: Any, factor: float) -> float:
    """A pixel length on a frame shrunk by *factor*. 0 stays 0, so a gate
    that is switched off (0 = disabled) stays switched off."""
    return float(value or 0) / factor


def shrink_kernel(value: Any, factor: float) -> int:
    """A kernel size on a frame shrunk by *factor*, never below 1 px.

    Detectors still round it up to odd themselves, exactly as they do for a
    hand-typed value.
    """
    return max(1, int(round(float(value or 1) / factor)))


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

    # ------------------------------------------------------- engine hooks
    @classmethod
    def params_from_camera_config(cls, camera_config: dict[str, Any]) -> dict[str, Any]:
        """This strategy's parameters, out of one camera's detection block.

        Default: the block named exactly like the strategy (``"opencv"`` for
        ``opencv``). Always a copy, so the detector can never reach back into
        the configuration it was built from.
        """
        return dict(camera_config.get(cls.name, {}))

    @classmethod
    def downscaled_params(cls, params: dict[str, Any], factor: float) -> dict[str, Any]:
        """*params* re-expressed for a frame shrunk by *factor* (> 1).

        Base version: the shared diameter gates. Override — and call this —
        when a strategy has other parameters measured in pixels.
        """
        scaled = dict(params)
        for key in DIAMETER_GATE_KEYS:
            if key in scaled:
                scaled[key] = shrink_length(scaled[key], factor)
        return scaled

    # ---------------------------------------------------------- strategy
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

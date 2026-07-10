"""YOLO deep-learning hole detector (optional plugin).

Works with any Ultralytics-format model trained on hole images. The heavy
dependency is imported lazily on first use, so stations running the classical
detector never need it installed:

    pip install ultralytics

then set in detection.json:  "active_detector": "yolo",
                             "yolo": {"model_path": "models/holes.pt", ...}
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import numpy as np

from core.vision.detection_result import DetectionResult, Hole
from core.vision.detector_base import HoleDetector
from core.utilities.exceptions import DetectionError


class YoloHoleDetector(HoleDetector):
    """Ultralytics YOLO inference strategy."""

    name: ClassVar[str] = "yolo"
    # single model instance; inference re-entrancy is not guaranteed, so the
    # engine serialises detect() calls for this strategy
    thread_safe: ClassVar[bool] = False

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._model = None
        self._loaded_path = ""
        super().__init__(params)

    def configure(self, params: dict[str, Any]) -> None:
        super().configure(params)
        if str(self._params.get("model_path", "")) != self._loaded_path:
            self._model = None  # force reload with the new weights

    def detect(self, image: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        model = self._ensure_model()
        try:
            results = model.predict(
                image,
                conf=float(self._params.get("confidence", 0.5)),
                iou=float(self._params.get("iou_threshold", 0.45)),
                verbose=False,
            )
            wanted_class = int(self._params.get("class_id", 0))

            holes: list[Hole] = []
            for box in results[0].boxes:
                if int(box.cls.item()) != wanted_class:
                    continue
                x0, y0, x1, y1 = (float(v) for v in box.xyxy[0].tolist())
                holes.append(
                    Hole(
                        x_px=(x0 + x1) / 2.0,
                        y_px=(y0 + y1) / 2.0,
                        diameter_px=((x1 - x0) + (y1 - y0)) / 2.0,
                        circularity=1.0,  # not measured by this strategy
                        confidence=float(box.conf.item()),
                    )
                )

            holes.sort(key=lambda hole: hole.confidence, reverse=True)
            return DetectionResult(
                holes=holes,
                processing_ms=(time.perf_counter() - started) * 1000.0,
            )
        except DetectionError:
            raise
        except Exception as exc:  # inference errors are model/driver specific
            raise DetectionError(f"YOLO inference failed: {exc}") from exc

    # ------------------------------------------------------------- internal
    def _ensure_model(self):
        if self._model is not None:
            return self._model

        model_path = str(self._params.get("model_path", "") or "")
        if not model_path:
            raise DetectionError("YOLO selected but 'model_path' is not configured")
        try:
            from ultralytics import YOLO  # heavy import, deferred on purpose
        except ImportError as exc:
            raise DetectionError(
                "YOLO detector requires the 'ultralytics' package "
                "(pip install ultralytics)"
            ) from exc
        try:
            self._model = YOLO(model_path)
        except Exception as exc:
            raise DetectionError(f"Cannot load YOLO model {model_path}: {exc}") from exc
        self._loaded_path = model_path
        return self._model

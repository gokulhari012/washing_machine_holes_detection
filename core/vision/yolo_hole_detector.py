"""YOLO deep-learning hole detector (optional plugin).

Works with any Ultralytics-format model trained on hole images. The heavy
dependency is imported lazily on first use, so stations running a classical
detector never need it installed:

    pip install ultralytics

then set in detection.json:  "active_detector": "yolo",
                             "yolo": {"model_path": "models/holes.pt", ...}

Parameters (the ``yolo`` block)
-------------------------------
``model_path``      weights (.pt / .onnx). Required.
``confidence``      per-box confidence the model must reach. This is the
                    model's own gate; ``common.confidence_threshold`` then
                    filters again in ``VisionEngine.detect``, so the
                    *effective* floor is whichever is higher.
``iou_threshold``   NMS IoU — how much two boxes may overlap before the
                    weaker is dropped.
``class_id``        which trained class counts as a hole; **-1 accepts every
                    class**, which is what a single-class model wants and
                    what stops a model whose one class happens to be id 3
                    silently detecting nothing.
``device``          "" (let Ultralytics choose), "cpu", "0", "cuda:0". On a
                    station with no CUDA this stays empty.
``imgsz``           inference size in px; 0 keeps the model's own default.
                    Must match how the model was trained to be worth setting.
``max_detections``  cap on boxes returned per frame; 0 keeps the default.
``min_hole_diameter_px`` / ``max_hole_diameter_px``
                    box-size gate in pixels of the analysed frame; either 0
                    disables it. A trained model usually needs no size gate —
                    this is here for the case where it also fires on a
                    similar-looking feature at a very different scale.

``circularity`` is reported as 1.0 and ``diameter_px`` is the box's mean
side: a detector returns an axis-aligned box, not a fitted shape, so neither
a roundness measurement nor a true bore diameter is available from it.

Threading: one model instance per detector, and Ultralytics inference is not
guaranteed re-entrant, so ``thread_safe`` is False and ``VisionEngine``
serialises that camera's ``detect()`` calls. Two cameras configured for YOLO
get two independent instances (and two loaded models) on purpose — sharing
one across the parallel capture mode's threads is exactly what the
per-camera lock could not protect.
"""

from __future__ import annotations

import os
import time
from typing import Any, ClassVar

import cv2
import numpy as np

from core.vision.detection_result import DetectionResult, Hole
from core.vision.detector_base import HoleDetector
from core.utilities.exceptions import DetectionError

#: ``class_id`` value meaning "every class the model predicts".
ANY_CLASS = -1


class YoloHoleDetector(HoleDetector):
    """Ultralytics YOLO inference strategy."""

    name: ClassVar[str] = "yolo"
    # single model instance; inference re-entrancy is not guaranteed, so the
    # engine serialises detect() calls for this strategy
    thread_safe: ClassVar[bool] = False

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._model = None
        self._model_key: tuple[str, float, int, str] | None = None
        super().__init__(params)

    # ----------------------------------------------------------- configure
    def configure(self, params: dict[str, Any]) -> None:
        """Adopt new parameters, reloading the model only when it actually changed.

        The reload key is the weights file's identity (path + mtime + size)
        *and* the device, so retraining over the same filename is picked up,
        while re-tuning ``confidence``/``iou_threshold`` — the cheap knobs an
        operator actually turns — keeps the loaded model in memory instead of
        paying a multi-second reload per edit.
        """
        super().configure(params)
        if self._model is not None and self._current_key() != self._model_key:
            self._model = None  # force a reload with the new weights/device
            self._model_key = None

    # -------------------------------------------------------------- detect
    def detect(self, image: np.ndarray) -> DetectionResult:
        started = time.perf_counter()
        min_diameter, max_diameter = self._diameter_gate()
        wanted_class = int(self._params.get("class_id", 0))

        holes: list[Hole] = []
        for x0, y0, x1, y1, confidence, class_id in self._predict(image):
            if wanted_class != ANY_CLASS and class_id != wanted_class:
                continue
            diameter = ((x1 - x0) + (y1 - y0)) / 2.0
            if min_diameter and diameter < min_diameter:
                continue
            if max_diameter and diameter > max_diameter:
                continue
            holes.append(
                Hole(
                    x_px=(x0 + x1) / 2.0,
                    y_px=(y0 + y1) / 2.0,
                    diameter_px=diameter,
                    circularity=1.0,  # not measured by this strategy
                    confidence=float(min(1.0, max(0.0, confidence))),
                )
            )

        holes.sort(key=lambda hole: hole.confidence, reverse=True)
        return DetectionResult(
            holes=holes,
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )

    def debug_stages(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Every box the model returned, as a filled mask — including the ones
        the class and size gates then reject.

        That is the point of the debug view: a model firing on the right
        feature but under a class id the config does not ask for, or at a
        size the gate excludes, looks identical to "found nothing" in the
        result view and obvious here.
        """
        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        for x0, y0, x1, y1, _confidence, _class_id in self._predict(image):
            cv2.rectangle(
                mask,
                (max(0, int(x0)), max(0, int(y0))),
                (min(width - 1, int(x1)), min(height - 1, int(y1))),
                255,
                thickness=cv2.FILLED,
            )
        return {"mask": mask}

    # ------------------------------------------------------------ internal
    def _diameter_gate(self) -> tuple[float, float]:
        """``(min, max)`` box-diameter gate; 0 on either side disables it."""
        min_diameter = max(0.0, float(self._params.get("min_hole_diameter_px", 0) or 0))
        max_diameter = max(0.0, float(self._params.get("max_hole_diameter_px", 0) or 0))
        if min_diameter and max_diameter and max_diameter <= min_diameter:
            raise DetectionError(
                "yolo: max_hole_diameter_px must be greater than "
                "min_hole_diameter_px (or 0 to disable the gate)"
            )
        return min_diameter, max_diameter

    def _predict(
        self, image: np.ndarray
    ) -> list[tuple[float, float, float, float, float, int]]:
        """Raw ``(x0, y0, x1, y1, confidence, class_id)`` for every box the
        model returns — before this detector's class and size gates.

        Raises:
            DetectionError: model missing/unloadable, or inference failed.
        """
        model = self._ensure_model()
        kwargs: dict[str, Any] = {
            "conf": float(self._params.get("confidence", 0.5)),
            "iou": float(self._params.get("iou_threshold", 0.45)),
            "verbose": False,
        }
        device = str(self._params.get("device", "") or "").strip()
        if device:
            kwargs["device"] = device
        imgsz = int(self._params.get("imgsz", 0) or 0)
        if imgsz > 0:
            kwargs["imgsz"] = imgsz
        max_detections = int(self._params.get("max_detections", 0) or 0)
        if max_detections > 0:
            kwargs["max_det"] = max_detections

        try:
            results = model.predict(image, **kwargs)
        except Exception as exc:  # inference errors are model/driver specific
            raise DetectionError(f"YOLO inference failed: {exc}") from exc

        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []
        try:
            return [
                (
                    *(float(value) for value in box.xyxy[0].tolist()),
                    float(box.conf.item()),
                    int(box.cls.item()),
                )
                for box in boxes
            ]
        except Exception as exc:  # a task whose results carry no usable boxes
            raise DetectionError(f"Cannot read YOLO detections: {exc}") from exc

    def _current_key(self) -> tuple[str, float, int, str]:
        """Identity of the weights currently *configured* — path, mtime, size
        and device. A missing file yields a sentinel that cannot match a real
        stat, so the next load retries and raises a useful error rather than
        reusing stale weights."""
        path = str(self._params.get("model_path", "") or "")
        device = str(self._params.get("device", "") or "").strip()
        try:
            stat = os.stat(path)
        except OSError:
            return (path, -1.0, -1, device)
        return (path, stat.st_mtime, stat.st_size, device)

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        model_path = str(self._params.get("model_path", "") or "")
        if not model_path:
            raise DetectionError("YOLO selected but 'model_path' is not configured")
        if not os.path.exists(model_path):
            raise DetectionError(f"YOLO model file not found: {model_path}")
        try:
            from ultralytics import YOLO  # heavy import, deferred on purpose
        except ImportError as exc:
            raise DetectionError(
                "YOLO detector requires the 'ultralytics' package "
                "(pip install ultralytics)"
            ) from exc
        try:
            model = YOLO(model_path)
        except Exception as exc:
            raise DetectionError(f"Cannot load YOLO model {model_path}: {exc}") from exc
        self._model = model
        self._model_key = self._current_key()
        return self._model

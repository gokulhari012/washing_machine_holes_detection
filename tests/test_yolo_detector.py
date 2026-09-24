"""yolo strategy: class/size gating, predict kwargs, model reuse, debug view.

Ultralytics is an optional dependency this station does not install (the
import is deliberately deferred so a classical-detector station never needs
it), so these tests stand a fake ``ultralytics`` module in ``sys.modules``.
That fake also *records* what ``predict`` was called with, which is the only
way to pin the parameters that exist purely to be forwarded — ``device``,
``imgsz``, ``max_detections`` — and to prove a re-tune does not reload the
weights.
"""

import sys
import types

import cv2
import numpy as np
import pytest

from core.utilities.exceptions import DetectionError
from core.vision.yolo_hole_detector import ANY_CLASS, YoloHoleDetector


class _Value:
    """Stands in for the torch tensors Ultralytics hangs off a Boxes row."""

    def __init__(self, value) -> None:
        self._value = value

    def tolist(self):
        return list(self._value)

    def item(self):
        return self._value


class _Box:
    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = [_Value(xyxy)]
        self.conf = _Value(conf)
        self.cls = _Value(cls)


class _Result:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


class FakeModel:
    """Records every predict() call and replays a scripted set of boxes."""

    def __init__(self, path) -> None:
        self.path = path
        self.calls: list[dict] = []
        self.boxes: list[_Box] = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [_Result(list(self.boxes))]


@pytest.fixture
def ultralytics(monkeypatch):
    """Install a fake ``ultralytics`` and hand back the models it builds."""
    built: list[FakeModel] = []

    def factory(path):
        model = FakeModel(path)
        built.append(model)
        return model

    module = types.ModuleType("ultralytics")
    module.YOLO = factory
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return built


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / "holes.pt"
    path.write_bytes(b"not really a model")
    return str(path)


FRAME = np.full((480, 640, 3), 120, np.uint8)


def detector(weights, **params) -> YoloHoleDetector:
    return YoloHoleDetector({"model_path": weights, **params})


def boxes_of(built) -> FakeModel:
    assert len(built) == 1, "expected exactly one model load"
    return built[0]


# ------------------------------------------------------------- detection
def test_boxes_become_holes(ultralytics, weights) -> None:
    instance = detector(weights)
    instance._ensure_model()
    boxes_of(ultralytics).boxes = [_Box([100.0, 200.0, 140.0, 260.0], 0.9, 0)]
    result = instance.detect(FRAME)
    assert len(result.holes) == 1
    hole = result.best
    assert hole.x_px == pytest.approx(120.0)
    assert hole.y_px == pytest.approx(230.0)
    assert hole.diameter_px == pytest.approx(50.0)  # mean of the box's sides
    assert hole.circularity == 1.0  # never measured by this strategy
    assert hole.confidence == pytest.approx(0.9)


def test_no_boxes_is_empty_not_error(ultralytics, weights) -> None:
    instance = detector(weights)
    instance._ensure_model()
    result = instance.detect(FRAME)
    assert result.holes == []
    assert result.found is False


def test_holes_are_sorted_best_first(ultralytics, weights) -> None:
    instance = detector(weights)
    instance._ensure_model()
    boxes_of(ultralytics).boxes = [
        _Box([0.0, 0.0, 40.0, 40.0], 0.4, 0),
        _Box([100.0, 100.0, 140.0, 140.0], 0.95, 0),
    ]
    holes = instance.detect(FRAME).holes
    assert [hole.confidence for hole in holes] == [pytest.approx(0.95), pytest.approx(0.4)]


def test_confidence_is_clamped(ultralytics, weights) -> None:
    instance = detector(weights)
    instance._ensure_model()
    boxes_of(ultralytics).boxes = [_Box([0.0, 0.0, 40.0, 40.0], 1.4, 0)]
    assert instance.detect(FRAME).best.confidence == 1.0


# ---------------------------------------------------------- class gating
def test_class_id_selects_one_class(ultralytics, weights) -> None:
    instance = detector(weights, class_id=2)
    instance._ensure_model()
    boxes_of(ultralytics).boxes = [
        _Box([0.0, 0.0, 40.0, 40.0], 0.9, 0),
        _Box([100.0, 100.0, 140.0, 140.0], 0.8, 2),
    ]
    holes = instance.detect(FRAME).holes
    assert len(holes) == 1
    assert holes[0].x_px == pytest.approx(120.0)


def test_any_class_accepts_every_class(ultralytics, weights) -> None:
    """A single-class model whose one class is not id 0 would otherwise
    silently detect nothing."""
    instance = detector(weights, class_id=ANY_CLASS)
    instance._ensure_model()
    boxes_of(ultralytics).boxes = [
        _Box([0.0, 0.0, 40.0, 40.0], 0.9, 3),
        _Box([100.0, 100.0, 140.0, 140.0], 0.8, 7),
    ]
    assert len(instance.detect(FRAME).holes) == 2


def test_class_id_defaults_to_zero(ultralytics, weights) -> None:
    instance = detector(weights)
    instance._ensure_model()
    boxes_of(ultralytics).boxes = [_Box([0.0, 0.0, 40.0, 40.0], 0.9, 1)]
    assert instance.detect(FRAME).holes == []


# ----------------------------------------------------------- size gating
def test_size_gate_rejects_out_of_range_boxes(ultralytics, weights) -> None:
    instance = detector(weights, min_hole_diameter_px=30, max_hole_diameter_px=80)
    instance._ensure_model()
    boxes_of(ultralytics).boxes = [
        _Box([0.0, 0.0, 10.0, 10.0], 0.9, 0),  # 10 px — too small
        _Box([100.0, 100.0, 150.0, 150.0], 0.8, 0),  # 50 px — kept
        _Box([200.0, 200.0, 400.0, 400.0], 0.7, 0),  # 200 px — too big
    ]
    holes = instance.detect(FRAME).holes
    assert [round(hole.diameter_px) for hole in holes] == [50]


def test_size_gate_is_off_by_default(ultralytics, weights) -> None:
    instance = detector(weights)
    instance._ensure_model()
    boxes_of(ultralytics).boxes = [
        _Box([0.0, 0.0, 4.0, 4.0], 0.9, 0),
        _Box([100.0, 100.0, 500.0, 460.0], 0.8, 0),
    ]
    assert len(instance.detect(FRAME).holes) == 2


def test_inverted_size_gate_raises(ultralytics, weights) -> None:
    instance = detector(weights, min_hole_diameter_px=200, max_hole_diameter_px=100)
    with pytest.raises(DetectionError, match="max_hole_diameter_px"):
        instance.detect(FRAME)


# ------------------------------------------------------- predict kwargs
def test_thresholds_are_forwarded(ultralytics, weights) -> None:
    instance = detector(weights, confidence=0.33, iou_threshold=0.22)
    instance.detect(FRAME)
    call = boxes_of(ultralytics).calls[-1]
    assert call["conf"] == pytest.approx(0.33)
    assert call["iou"] == pytest.approx(0.22)
    assert call["verbose"] is False


def test_optional_kwargs_are_omitted_when_unset(ultralytics, weights) -> None:
    """Passing imgsz=0 or device="" would override the model's own defaults
    with nonsense rather than leaving them alone."""
    detector(weights, device="", imgsz=0, max_detections=0).detect(FRAME)
    call = boxes_of(ultralytics).calls[-1]
    assert "device" not in call
    assert "imgsz" not in call
    assert "max_det" not in call


def test_optional_kwargs_are_forwarded_when_set(ultralytics, weights) -> None:
    detector(weights, device="cpu", imgsz=1280, max_detections=8).detect(FRAME)
    call = boxes_of(ultralytics).calls[-1]
    assert call["device"] == "cpu"
    assert call["imgsz"] == 1280
    assert call["max_det"] == 8


# -------------------------------------------------------- model lifecycle
def test_model_is_loaded_once_and_reused(ultralytics, weights) -> None:
    instance = detector(weights)
    for _ in range(3):
        instance.detect(FRAME)
    assert len(ultralytics) == 1
    assert len(ultralytics[0].calls) == 3


def test_retuning_thresholds_does_not_reload_the_model(ultralytics, weights) -> None:
    """What makes re-tuning on the Detection page bearable — a reload is
    seconds of torch start-up per edit."""
    instance = detector(weights)
    instance.detect(FRAME)
    for confidence in (0.4, 0.5, 0.6):
        instance.configure({"confidence": confidence})
        instance.detect(FRAME)
    assert len(ultralytics) == 1


def test_changing_the_device_reloads(ultralytics, weights) -> None:
    instance = detector(weights, device="cpu")
    instance.detect(FRAME)
    instance.configure({"device": "cuda:0"})
    instance.detect(FRAME)
    assert len(ultralytics) == 2


def test_retrained_weights_under_the_same_name_are_reloaded(
    ultralytics, weights, tmp_path
) -> None:
    import os

    instance = detector(weights)
    instance.detect(FRAME)
    with open(weights, "wb") as handle:
        handle.write(b"retrained, and longer than before")
    os.utime(weights, (0, 0))  # a distinctly different mtime
    instance.configure({"model_path": weights})
    instance.detect(FRAME)
    assert len(ultralytics) == 2


def test_missing_model_path_raises(ultralytics) -> None:
    with pytest.raises(DetectionError, match="model_path"):
        YoloHoleDetector({"model_path": ""}).detect(FRAME)


def test_model_file_not_found_raises_before_import(monkeypatch, tmp_path) -> None:
    """The clear error must win over whatever Ultralytics would say — and it
    must not even need Ultralytics installed to be produced."""
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)
    with pytest.raises(DetectionError, match="not found"):
        YoloHoleDetector({"model_path": str(tmp_path / "absent.pt")}).detect(FRAME)


def test_missing_ultralytics_names_the_package(monkeypatch, weights) -> None:
    monkeypatch.setitem(sys.modules, "ultralytics", None)  # forces ImportError
    with pytest.raises(DetectionError, match="ultralytics"):
        YoloHoleDetector({"model_path": weights}).detect(FRAME)


def test_inference_failure_degrades_to_detection_error(ultralytics, weights) -> None:
    instance = detector(weights)
    instance._ensure_model()

    def boom(image, **kwargs):
        raise RuntimeError("CUDA out of memory")

    boxes_of(ultralytics).predict = boom
    with pytest.raises(DetectionError, match="CUDA out of memory"):
        instance.detect(FRAME)


# ------------------------------------------------------------ debug view
def test_debug_mask_marks_every_box_including_rejected_ones(
    ultralytics, weights
) -> None:
    """A model firing under the wrong class id looks like 'found nothing' in
    the result view; the debug view is where it becomes visible."""
    instance = detector(weights, class_id=0, max_hole_diameter_px=80)
    instance._ensure_model()
    boxes_of(ultralytics).boxes = [
        _Box([100.0, 100.0, 150.0, 150.0], 0.9, 0),  # accepted
        _Box([300.0, 200.0, 350.0, 250.0], 0.8, 5),  # wrong class
        _Box([400.0, 300.0, 600.0, 460.0], 0.7, 0),  # over the size gate
    ]
    assert len(instance.detect(FRAME).holes) == 1

    mask = instance.debug_stages(FRAME)["mask"]
    assert mask.shape == FRAME.shape[:2]
    assert mask.dtype == np.uint8
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 3


def test_debug_mask_is_empty_without_boxes(ultralytics, weights) -> None:
    instance = detector(weights)
    instance._ensure_model()
    assert not instance.debug_stages(FRAME)["mask"].any()


def test_debug_mask_clips_boxes_to_the_frame(ultralytics, weights) -> None:
    """A box can run off the edge; writing it unclipped would throw."""
    instance = detector(weights)
    instance._ensure_model()
    boxes_of(ultralytics).boxes = [_Box([-50.0, -50.0, 900.0, 900.0], 0.9, 0)]
    assert instance.debug_stages(FRAME)["mask"].any()

"""Detection page wiring for every strategy in the Active Detector list.

The page holds one parameter form per strategy and shows the ones the
selected strategy runs (``_FORMS_SHOWN``) — so a strategy without a form, or a
form shown for the wrong strategy, silently edits parameters nothing uses, and
a parameter missing from ``_load``/``_collect`` silently reverts on every save. Neither
shows up in a detector's own unit tests, which is what these cover: pick a
strategy, fill its form, Save, and check both the running engine and the
persisted JSON got exactly what was on screen.
"""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from core.camera import CameraManager
from core.led import LedControllerSettings, LedManager, SimulatedLedClient
from core.utilities.config_manager import ConfigManager
from core.utilities.enums import DetectorType
from core.vision import DetectionResult, VisionEngine
from models.app_state import AppState
from services.camera_service import CameraService
from services.led_service import LedService
from ui.detection import DetectionPage

CAMERA_DOC = {
    "cameras": [
        {
            "index": 1, "name": "Cam 1", "driver": "simulated", "connection_id": "",
            "enabled": True, "exposure_us": 10000, "gain_db": 0.0, "gamma": 1.0,
            "brightness": 10, "led_channel": 0, "led_strobe": False, "fps": 4.0,
            "rotation": 0, "width": 640, "height": 480, "trigger_mode": "software",
            "roi": {"x": 0, "y": 0, "width": 0, "height": 0},
        },
    ],
}

DETECTION_DOC = {
    "cameras": {
        "1": {
            "active_detector": "opencv",
            "common": {
                "confidence_threshold": 0.6, "expected_hole_count": 1,
                "position_tolerance_mm": 0.0, "normalize_image": False,
            },
            "opencv": {"detection_threshold": 60, "min_hole_diameter_px": 20},
            "dark_hole": {"min_contrast": 18, "min_hole_diameter_px": 15},
            "template_matching": {
                "template_path": "", "match_threshold": 0.8,
                "method": "TM_CCOEFF_NORMED", "scales": [1.0], "max_matches": 10,
                "min_hole_diameter_px": 0, "max_hole_diameter_px": 0,
            },
            "yolo": {
                "model_path": "", "confidence": 0.5, "iou_threshold": 0.45,
                "class_id": 0, "device": "", "imgsz": 0, "max_detections": 0,
                "min_hole_diameter_px": 0, "max_hole_diameter_px": 0,
            },
        },
    },
}


class FakeRepo:
    def upsert(self, values: dict) -> None:
        pass

    def delete_by_index(self, index: int) -> None:
        pass


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _silence_modals(monkeypatch):
    """Save reports success with a modal, which would block the run."""
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))


@pytest.fixture
def page(tmp_path, qt_app):
    config_dir = tmp_path / "config"
    (config_dir / "defaults").mkdir(parents=True)
    (config_dir / "camera.json").write_text(json.dumps(CAMERA_DOC))
    (config_dir / "detection.json").write_text(json.dumps(DETECTION_DOC))
    (config_dir / "defaults" / "detection.json").write_text(json.dumps(DETECTION_DOC))

    config = ConfigManager(config_dir)
    database = SimpleNamespace(camera_configs=FakeRepo(), plc_config=FakeRepo())
    led_service = LedService(
        LedManager(SimulatedLedClient(), LedControllerSettings()), config
    )
    cameras = CameraService(
        CameraManager(CAMERA_DOC["cameras"]), config, database, led_service
    )
    engine = VisionEngine(DETECTION_DOC)
    widget = DetectionPage(config, engine, cameras, AppState())
    return SimpleNamespace(widget=widget, engine=engine, config=config, tmp_path=tmp_path)


def stored(page) -> dict:
    return page.config.load("detection")["cameras"]["1"]


def template_file(tmp_path) -> str:
    image = np.full((80, 80), 190, np.uint8)
    cv2.circle(image, (40, 40), 30, 25, -1)
    path = tmp_path / "golden_hole.png"
    cv2.imwrite(str(path), image)
    return str(path)


# ---------------------------------------------------------- form wiring
def test_every_strategy_has_its_own_form(page) -> None:
    assert set(page.widget._forms) == {d.value for d in DetectorType}


@pytest.mark.parametrize("strategy", [d.value for d in DetectorType])
def test_selecting_a_strategy_shows_its_own_form(page, strategy) -> None:
    """Exactly the forms that strategy runs are on screen — a single-stage
    strategy shows its own and nothing else."""
    widget = page.widget
    widget._strategy.setCurrentText(strategy)
    shown = widget._visible_forms()
    assert strategy in shown
    if strategy != "template_opencv":
        assert shown == [strategy]


# ------------------------------------------------- template_matching form
def test_template_matching_round_trips_through_save_and_load(page) -> None:
    widget = page.widget
    path = template_file(page.tmp_path)
    widget._strategy.setCurrentText("template_matching")
    widget._tm_path.setText(path)
    widget._tm_threshold.setValue(0.65)
    widget._tm_method.setCurrentText("TM_SQDIFF_NORMED")
    widget._tm_scales.setText("0.8, 1.0, 1.25")
    widget._tm_max_matches.setValue(4)
    widget._tm_min_diameter.setValue(40)
    widget._tm_max_diameter.setValue(160)
    widget._on_save()

    block = stored(page)["template_matching"]
    assert block == {
        "template_path": path,
        "match_threshold": 0.65,
        "method": "TM_SQDIFF_NORMED",
        "scales": [0.8, 1.0, 1.25],
        "max_matches": 4,
        "min_hole_diameter_px": 40,
        "max_hole_diameter_px": 160,
    }
    assert stored(page)["active_detector"] == "template_matching"
    assert page.engine.active_detector_name(1) == "template_matching"

    widget._load()  # the form must come back showing what was saved
    assert widget._tm_scales.text() == "0.8, 1, 1.25"
    assert widget._tm_max_matches.value() == 4
    assert widget._tm_min_diameter.value() == 40
    assert widget._tm_method.currentText() == "TM_SQDIFF_NORMED"


def test_typed_scales_are_normalised_before_they_are_persisted(page) -> None:
    """detection.json always holds a clean list, whatever was typed."""
    widget = page.widget
    widget._strategy.setCurrentText("template_matching")
    widget._tm_path.setText(template_file(page.tmp_path))
    widget._tm_scales.setText("1.25 ; 0.8,, junk, -2, 1.0")
    widget._on_save()
    assert stored(page)["template_matching"]["scales"] == [0.8, 1.0, 1.25]


def test_a_template_detector_actually_runs_after_save(page) -> None:
    """Save hot-swaps the engine, so the next detect() uses the new strategy."""
    widget = page.widget
    widget._strategy.setCurrentText("template_matching")
    widget._tm_path.setText(template_file(page.tmp_path))
    widget._tm_threshold.setValue(0.7)
    widget._tm_scales.setText("1.0")
    widget._confidence.setValue(0.5)
    widget._on_save()

    frame = np.full((300, 400), 190, np.uint8)
    cv2.circle(frame, (210, 140), 30, 25, np.int32(-1))
    result = page.engine.detect(frame, 1)
    assert result.best is not None
    assert abs(result.best.x_px - 210) <= 4
    assert abs(result.best.y_px - 140) <= 4


def test_template_debug_view_reaches_the_engine(page) -> None:
    """Previously this strategy returned {} and the Debug view said it had
    nothing to show."""
    widget = page.widget
    widget._strategy.setCurrentText("template_matching")
    widget._tm_path.setText(template_file(page.tmp_path))
    widget._tm_scales.setText("1.0")
    widget._on_save()

    frame = np.full((300, 400), 190, np.uint8)
    cv2.circle(frame, (210, 140), 30, 25, np.int32(-1))
    stages = page.engine.debug_stages(frame, 1)
    assert "mask" in stages
    assert stages["mask"].shape == frame.shape
    assert stages["mask"].any()


# ------------------------------------------------------------- yolo form
def test_yolo_round_trips_through_save_and_load(page) -> None:
    widget = page.widget
    widget._strategy.setCurrentText("yolo")
    widget._yolo_path.setText("models/holes.pt")
    widget._yolo_conf.setValue(0.35)
    widget._yolo_iou.setValue(0.55)
    widget._yolo_class.setValue(-1)
    widget._yolo_device.setCurrentText("cpu")
    widget._yolo_imgsz.setValue(1280)
    widget._yolo_max_det.setValue(12)
    widget._yolo_min_diameter.setValue(25)
    widget._yolo_max_diameter.setValue(300)
    widget._on_save()

    assert stored(page)["yolo"] == {
        "model_path": "models/holes.pt",
        "confidence": 0.35,
        "iou_threshold": 0.55,
        "class_id": -1,
        "device": "cpu",
        "imgsz": 1280,
        "max_detections": 12,
        "min_hole_diameter_px": 25,
        "max_hole_diameter_px": 300,
    }

    widget._load()
    assert widget._yolo_class.value() == -1
    assert widget._yolo_device.currentText() == "cpu"
    assert widget._yolo_imgsz.value() == 1280
    assert widget._yolo_max_det.value() == 12
    assert widget._yolo_max_diameter.value() == 300


def test_yolo_class_field_reaches_any_class(page) -> None:
    """-1 means 'every class'; a spin box floored at 0 could not express it."""
    assert page.widget._yolo_class.minimum() == -1
    page.widget._yolo_class.setValue(-1)
    assert page.widget._yolo_class.value() == -1


def test_a_missing_model_is_a_detect_time_error_not_a_build_time_one(page) -> None:
    """Selecting yolo with no weights saves, and fails when it runs.

    Deliberate, and the same contract template_matching has had: a detector is
    *constructed* at startup and inside ``apply_config``'s all-or-nothing
    hot-swap, so raising there for a missing file would turn a tuning mistake
    into a station that will not boot — and would take the other three
    cameras' working strategies down with it. The misconfiguration surfaces
    where an operator is standing: "Test on Camera" reports it immediately.
    """
    widget = page.widget
    widget._strategy.setCurrentText("yolo")
    widget._yolo_path.setText("")
    widget._on_save()
    assert stored(page)["active_detector"] == "yolo"
    assert page.engine.active_detector_name(1) == "yolo"

    from core.utilities.exceptions import DetectionError

    with pytest.raises(DetectionError, match="model_path"):
        page.engine.detect(np.full((300, 400), 190, np.uint8), 1)


# ------------------------------------------------------- sweep availability
def test_sweep_button_follows_the_selected_strategy(page) -> None:
    widget = page.widget
    for strategy, enabled in [
        ("opencv", True),
        ("dark_hole", True),
        ("template_matching", True),
        ("yolo", False),
        ("template_opencv", False),
    ]:
        widget._strategy.setCurrentText(strategy)
        assert widget._sweep_btn.isEnabled() is enabled, strategy
    widget._strategy.setCurrentText("yolo")
    assert "crop" in widget._sweep_btn.toolTip()
    widget._strategy.setCurrentText("template_opencv")
    assert "one strategy at a time" in widget._sweep_btn.toolTip()


# ------------------------------------------------- template_opencv form
def test_template_opencv_shows_its_own_form_over_both_stage_forms(page) -> None:
    """Its stages are the camera's template_matching and opencv blocks, so
    those two forms are on screen with it, retitled as the stages."""
    widget = page.widget
    widget._strategy.setCurrentText("template_opencv")
    assert set(widget._visible_forms()) == {"template_opencv", "template_matching", "opencv"}
    assert widget._forms["template_matching"].title().startswith("Stage 1")
    assert widget._forms["opencv"].title().startswith("Stage 2")

    widget._strategy.setCurrentText("opencv")  # titles go back on their own
    assert widget._forms["opencv"].title() == "OpenCV Parameters"
    assert widget._forms["template_matching"].title() == "Template Matching Parameters"


def test_template_opencv_round_trips_through_save_and_load(page) -> None:
    widget = page.widget
    path = template_file(page.tmp_path)
    widget._strategy.setCurrentText("template_opencv")
    widget._tm_path.setText(path)
    widget._cv_threshold.setValue(77)
    widget._to_regions.setValue(2)
    widget._to_padding.setValue(15)
    widget._on_save()

    saved = stored(page)
    assert saved["active_detector"] == "template_opencv"
    assert saved["template_opencv"] == {"max_regions": 2, "region_padding_px": 15}
    assert saved["template_matching"]["template_path"] == path  # stage 1
    assert saved["opencv"]["detection_threshold"] == 77  # stage 2
    assert page.engine.active_detector_name(1) == "template_opencv"

    widget._to_regions.setValue(1)
    widget._to_padding.setValue(0)
    widget._load()
    assert widget._to_regions.value() == 2
    assert widget._to_padding.value() == 15


# ---------------------------------------------- downscale / NG re-run
def test_an_older_detection_json_loads_with_both_switched_off(page) -> None:
    """The fixture's common block predates both keys."""
    widget = page.widget
    assert widget._downscale.isChecked() is False
    assert widget._downscale_factor.isEnabled() is False
    assert widget._rerun.isChecked() is False
    assert widget._rerun_count.isEnabled() is False


def test_downscale_and_rerun_round_trip_and_reach_the_engine(page) -> None:
    widget = page.widget
    widget._downscale.setChecked(True)
    widget._downscale_factor.setValue(3.0)
    widget._rerun.setChecked(True)
    widget._rerun_count.setValue(4)
    widget._on_save()

    common = stored(page)["common"]
    assert common["downscale_enabled"] is True
    assert common["downscale_factor"] == 3.0
    assert common["ng_rerun_enabled"] is True
    assert common["ng_rerun_count"] == 4
    assert page.engine.downscale_factor(1) == 3.0
    assert page.engine.ng_rerun_count(1) == 4

    widget._downscale.setChecked(False)
    widget._rerun.setChecked(False)
    widget._load()
    assert widget._downscale.isChecked() and widget._downscale_factor.value() == 3.0
    assert widget._rerun.isChecked() and widget._rerun_count.value() == 4


def test_switching_off_keeps_the_value_for_next_time(page) -> None:
    widget = page.widget
    widget._downscale.setChecked(True)
    widget._downscale_factor.setValue(4.0)
    widget._downscale.setChecked(False)
    assert widget._downscale_factor.isEnabled() is False
    widget._on_save()
    common = stored(page)["common"]
    assert common["downscale_enabled"] is False
    assert common["downscale_factor"] == 4.0
    assert page.engine.downscale_factor(1) == 1.0  # off means off


def test_an_ng_verdict_says_what_a_real_cycle_would_do(page) -> None:
    """Test is one capture; the verdict says a cycle would re-run instead."""
    widget = page.widget
    widget._last_frame = np.zeros((40, 40), np.uint8)
    widget._last_result = DetectionResult(holes=[])
    widget._last_camera_index = 1
    widget._rerun.setChecked(True)
    widget._rerun_count.setValue(2)
    assert "up to 2 more time(s)" in widget._verdict_label.text()
    widget._rerun.setChecked(False)
    assert "more time(s)" not in widget._verdict_label.text()
    assert widget._verdict_label.text().startswith("NG")


# ------------------------------------------------ yolo training tooling
def test_training_button_is_disabled_without_the_service(page) -> None:
    """The page is built without a YoloTrainingService here, so the button
    says why rather than being absent — parameters still edit normally."""
    assert page.widget._train_btn.isEnabled() is False
    assert "not wired" in page.widget._train_btn.toolTip()


def test_training_button_lives_on_the_yolo_form_only(page) -> None:
    """It is reachable only while yolo is the selected strategy, because it
    is the one strategy that needs a trained file to do anything."""
    widget = page.widget
    assert widget._train_btn.parent() is widget._forms["yolo"]

    widget._strategy.setCurrentText("opencv")
    assert "yolo" not in widget._visible_forms()
    widget._strategy.setCurrentText("yolo")
    assert widget._visible_forms() == ["yolo"]


def test_training_button_is_enabled_when_the_service_is_wired(tmp_path, qt_app) -> None:
    from services.yolo_training_service import YoloTrainingService
    from ui.detection import DetectionPage

    config_dir = tmp_path / "config2"
    (config_dir / "defaults").mkdir(parents=True)
    (config_dir / "camera.json").write_text(json.dumps(CAMERA_DOC))
    (config_dir / "detection.json").write_text(json.dumps(DETECTION_DOC))
    (config_dir / "defaults" / "detection.json").write_text(json.dumps(DETECTION_DOC))
    config = ConfigManager(config_dir)
    database = SimpleNamespace(camera_configs=FakeRepo(), plc_config=FakeRepo())
    led_service = LedService(
        LedManager(SimulatedLedClient(), LedControllerSettings()), config
    )
    cameras = CameraService(
        CameraManager(CAMERA_DOC["cameras"]), config, database, led_service
    )
    widget = DetectionPage(
        config, VisionEngine(DETECTION_DOC), cameras, AppState(),
        None, YoloTrainingService(),
    )
    assert widget._train_btn.isEnabled() is True


def test_training_dialog_round_trips_its_settings(qt_app) -> None:
    """What the page persists into app_config.json and reloads next session."""
    from services.yolo_training_service import YoloTrainingService
    from ui.detection.yolo_training_dialog import YoloTrainingDialog

    saved = {
        "dataset_dir": "D:/data/cam2",
        "validation_dir": "D:/data/val",
        "output_dir": "D:/runs",
        "class_names": "hole, burr",
        "model": "yolo11m.pt",
        "epochs": 250,
        "imgsz": 1280,
        "device": "cpu",
    }
    dialog = YoloTrainingDialog(YoloTrainingService(), dict(saved))
    try:
        assert dialog.settings() == saved
        assert dialog.trained_weights == ""
    finally:
        dialog.deleteLater()


def test_training_dialog_survives_an_empty_settings_block(qt_app) -> None:
    """First run: app_config.json ships every key blank."""
    from services.yolo_training_service import YOLO_MODELS, YoloTrainingService
    from ui.detection.yolo_training_dialog import YoloTrainingDialog

    dialog = YoloTrainingDialog(YoloTrainingService(), {})
    try:
        settings = dialog.settings()
        assert settings["dataset_dir"] == ""
        assert settings["class_names"] == "hole"
        assert settings["model"] in YOLO_MODELS
    finally:
        dialog.deleteLater()


def test_training_dialog_summarises_the_chosen_dataset(qt_app, tmp_path) -> None:
    """The count appears as soon as a folder is typed, before Start is pressed."""
    from services.yolo_training_service import YoloTrainingService
    from ui.detection.yolo_training_dialog import YoloTrainingDialog

    folder = tmp_path / "ds"
    folder.mkdir()
    for index in range(3):
        (folder / f"f{index}.png").write_bytes(b"\x89PNG")
        (folder / f"f{index}.txt").write_text("0 .5 .5 .1 .1\n", encoding="utf-8")
    (folder / "extra.png").write_bytes(b"\x89PNG")  # deliberately unlabelled

    dialog = YoloTrainingDialog(YoloTrainingService(), {})
    try:
        dialog._dataset.setText(str(folder))
        text = dialog._dataset_status.text()
        assert "4 image(s), 3 labelled" in text
        assert "1 with no label file" in text
    finally:
        dialog.deleteLater()


def test_folder_class_list_wins_over_the_remembered_one(qt_app, tmp_path) -> None:
    """The remembered setting can be stale — the list in the dataset folder
    is what the images were actually labelled with, so it wins on open."""
    from services.yolo_training_service import YoloTrainingService
    from ui.detection.yolo_training_dialog import YoloTrainingDialog

    folder = tmp_path / "ds"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"\x89PNG")
    (folder / "a.txt").write_text("0 .5 .5 .1 .1\n", encoding="utf-8")
    (folder / "lables.txt").write_text("Holes\n", encoding="utf-8")

    dialog = YoloTrainingDialog(
        YoloTrainingService(), {"dataset_dir": str(folder), "class_names": "stale"}
    )
    try:
        assert dialog._classes.text() == "Holes"
        assert dialog.settings()["class_names"] == "Holes"
    finally:
        dialog.deleteLater()


def test_remembered_classes_survive_a_folder_with_no_list(qt_app, tmp_path) -> None:
    folder = tmp_path / "ds"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"\x89PNG")
    (folder / "a.txt").write_text("0 .5 .5 .1 .1\n", encoding="utf-8")

    from services.yolo_training_service import YoloTrainingService
    from ui.detection.yolo_training_dialog import YoloTrainingDialog

    dialog = YoloTrainingDialog(
        YoloTrainingService(), {"dataset_dir": str(folder), "class_names": "hole, burr"}
    )
    try:
        assert dialog._classes.text() == "hole, burr"
    finally:
        dialog.deleteLater()

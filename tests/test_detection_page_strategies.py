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
    from ui.detection import detection_page

    widget = page.widget
    widget._strategy.setCurrentText(strategy)
    shown = widget._visible_forms()
    assert set(shown) == set(detection_page._FORMS_SHOWN.get(strategy, (strategy,)))


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
        ("template_opencv", True),
        ("blob_analysis", True),
        ("template_blob_analysis", True),
    ]:
        widget._strategy.setCurrentText(strategy)
        assert widget._sweep_btn.isEnabled() is enabled, strategy
    widget._strategy.setCurrentText("yolo")
    assert "crop" in widget._sweep_btn.toolTip()
    for two_stage in ("template_opencv", "template_blob_analysis"):
        widget._strategy.setCurrentText(two_stage)
        assert widget._sweep_btn.toolTip().startswith("Stage 1"), two_stage


# ---------------------------------------------- two-stage strategy sweeps
@pytest.mark.parametrize(
    "strategy, hole_stage",
    [("template_opencv", "opencv"), ("template_blob_analysis", "blob_analysis")],
)
def test_stage_selector_only_shows_for_two_stage_strategies(page, strategy, hole_stage) -> None:
    widget = page.widget
    widget._strategy.setCurrentText("opencv")
    assert widget._sweep_stage.isHidden()
    assert widget._sweep_target("opencv") == "opencv"
    widget._strategy.setCurrentText("blob_analysis")
    assert widget._sweep_stage.isHidden()

    widget._strategy.setCurrentText(strategy)
    assert not widget._sweep_stage.isHidden()
    stages = [widget._sweep_stage.itemData(i) for i in range(widget._sweep_stage.count())]
    assert stages == ["template_matching", hole_stage]
    widget._sweep_stage.setCurrentIndex(1)
    assert widget._sweep_target(strategy) == hole_stage
    assert widget._sweep_btn.toolTip().startswith("Stage 2")

    # a reload (camera / model switch) keeps the chosen stage
    widget._update_sweep_availability(strategy)
    assert widget._sweep_target(strategy) == hole_stage


def _feature_frame(tmp_path):
    """A 400 px frame holding a grey bracket with a 40 px bore at (200, 200),
    plus a template cropped around the bracket — what template_opencv wants."""
    frame = np.full((400, 400), 180, np.uint8)
    cv2.rectangle(frame, (150, 150), (249, 249), 110, -1)
    cv2.circle(frame, (200, 200), 20, 20, -1)
    path = tmp_path / "bracket.png"
    cv2.imwrite(str(path), frame[140:260, 140:260])
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), str(path)


def _ready_for_stage_sweep(page, monkeypatch, stage: int, roi, strategy="template_opencv") -> None:
    widget = page.widget
    frame, template = _feature_frame(page.tmp_path)
    widget._strategy.setCurrentText(strategy)
    widget._tm_path.setText(template)
    widget._tm_threshold.setValue(0.8)
    widget._tm_scales.setText("1")
    widget._tm_min_diameter.setValue(0)
    widget._tm_max_diameter.setValue(0)
    widget._sweep_stage.setCurrentIndex(stage)
    widget._last_frame = frame
    monkeypatch.setattr(widget._view, "current_roi", lambda: roi)


@pytest.mark.parametrize("strategy", ["template_opencv", "template_blob_analysis"])
def test_stage_one_sweep_tunes_the_template_form(page, monkeypatch, strategy) -> None:
    widget = page.widget
    _ready_for_stage_sweep(page, monkeypatch, 0, (140, 140, 120, 120), strategy)
    widget._tm_method.setCurrentText("TM_SQDIFF_NORMED")

    widget._on_run_sweep()

    assert widget._sweep_rows, widget._sweep_status.text()
    assert widget._sweep_rows_strategy == "template_matching"
    best_params, best_hole = widget._sweep_rows[0]
    assert abs(best_hole.x_px - 200) < 5 and abs(best_hole.y_px - 200) < 5
    widget._on_apply_best_sweep()
    assert widget._tm_method.currentText() == best_params["method"]
    assert widget._strategy.currentText() == strategy  # stays the active detector


def test_stage_two_sweep_runs_opencv_inside_the_template_match(page, monkeypatch) -> None:
    from ui.detection import detection_page

    # a reduced grid keeps the test quick; the path is what is under test
    monkeypatch.setitem(detection_page._SWEEP_LEVELS, "opencv", {
        "detection_threshold": [60, 120],
        "blur_kernel_size": [3],
        "morphology_operation": ["close"],
        "morphology_kernel_size": [3],
        "morphology_iterations": [1],
        "min_circularity": [0.5],
        "min_aspect_ratio": [0.3],
        "contour_retrieval_mode": ["external", "list"],
    })
    widget = page.widget
    _ready_for_stage_sweep(page, monkeypatch, 1, (175, 175, 50, 50))
    seen_views = []
    real_grid = widget._run_sweep_grid

    def spy(combos, detector, views, roi, found):
        seen_views.extend((image.shape[:2], ox, oy) for image, ox, oy in views)
        return real_grid(combos, detector, views, roi, found)

    monkeypatch.setattr(widget, "_run_sweep_grid", spy)
    widget._on_run_sweep()

    # OpenCV saw the template match (120 px square at 140,140), not a ROI crop
    assert seen_views == [((120, 120), 140, 140)]
    assert widget._sweep_rows, widget._sweep_status.text()
    assert widget._sweep_rows_strategy == "opencv"
    best_params, best_hole = widget._sweep_rows[0]
    assert abs(best_hole.x_px - 200) < 3 and abs(best_hole.y_px - 200) < 3
    widget._cv_threshold.setValue(1)
    widget._on_apply_best_sweep()
    assert widget._cv_threshold.value() == best_params["detection_threshold"]


def test_stage_two_sweep_runs_blob_analysis_inside_the_template_match(page, monkeypatch) -> None:
    """template_blob_analysis Stage 2: blob analysis picks the *largest* blob
    in what it is given, so the crop it is tuned on matters even more than
    for OpenCV — it must be the match, as in a real cycle."""
    from ui.detection import detection_page

    monkeypatch.setitem(detection_page._SWEEP_LEVELS, "blob_analysis", {
        # 150 merges the bore into the 110-grey bracket: a square blob, which
        # the 0.6 floor accepts but which scores below the round bore at 60
        "threshold": [60, 150],
        "blur_kernel_size": [3],
        "morphology_kernel_size": [3],
        "min_circularity": [0.8, 0.6],
    })
    widget = page.widget
    _ready_for_stage_sweep(page, monkeypatch, 1, (175, 175, 50, 50), "template_blob_analysis")
    seen_views = []
    real_grid = widget._run_sweep_grid

    def spy(combos, detector, views, roi, found):
        seen_views.extend((image.shape[:2], ox, oy) for image, ox, oy in views)
        return real_grid(combos, detector, views, roi, found)

    monkeypatch.setattr(widget, "_run_sweep_grid", spy)
    widget._on_run_sweep()

    assert seen_views == [((120, 120), 140, 140)]  # the template match, not a ROI crop
    assert widget._sweep_rows, widget._sweep_status.text()
    assert widget._sweep_rows_strategy == "blob_analysis"
    best_params, best_hole = widget._sweep_rows[0]
    assert abs(best_hole.x_px - 200) < 3 and abs(best_hole.y_px - 200) < 3
    assert best_params["threshold"] == 60
    widget._ba_threshold.setValue(1)
    widget._on_apply_best_sweep()
    assert widget._ba_threshold.value() == 60
    assert widget._strategy.currentText() == "template_blob_analysis"


@pytest.mark.parametrize("strategy", ["template_opencv", "template_blob_analysis"])
def test_stage_two_sweep_without_a_match_says_tune_stage_one(page, monkeypatch, strategy) -> None:
    messages = []
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda _p, _t, text: messages.append(text))
    )
    widget = page.widget
    _ready_for_stage_sweep(page, monkeypatch, 1, (175, 175, 50, 50), strategy)
    widget._last_frame = np.full((400, 400, 3), 180, np.uint8)  # no bracket anywhere

    widget._on_run_sweep()

    assert widget._sweep_rows == []
    assert messages and "Stage 1" in messages[0]


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


# ------------------------------------------ blob_analysis / template_blob
def test_an_older_detection_json_loads_the_blob_forms_at_their_defaults(page) -> None:
    """The fixture predates both blocks — the lab script's slider defaults."""
    widget = page.widget
    assert widget._ba_threshold.value() == 40
    assert widget._ba_blur.value() == 7
    assert widget._ba_morph.value() == 5
    assert widget._ba_min_area.value() == 500
    assert widget._ba_circularity.value() == pytest.approx(0.8)
    assert widget._tb_regions.value() == 1
    assert widget._tb_padding.value() == 0


def test_blob_analysis_round_trips_through_save_and_load(page) -> None:
    widget = page.widget
    widget._strategy.setCurrentText("blob_analysis")
    widget._ba_threshold.setValue(55)
    widget._ba_blur.setValue(9)
    widget._ba_morph.setValue(3)
    widget._ba_min_area.setValue(1200)
    widget._ba_circularity.setValue(0.72)
    widget._ba_min_diameter.setValue(30)
    widget._ba_max_diameter.setValue(300)
    widget._on_save()

    expected = {
        "threshold": 55, "blur_kernel_size": 9, "morphology_kernel_size": 3,
        "min_area_px": 1200, "min_circularity": pytest.approx(0.72),
        "min_hole_diameter_px": 30, "max_hole_diameter_px": 300,
    }
    saved = stored(page)
    assert saved["active_detector"] == "blob_analysis"
    assert saved["blob_analysis"] == expected
    assert page.engine.active_detector_name(1) == "blob_analysis"
    assert page.engine.camera_config(1)["blob_analysis"] == expected

    widget._ba_threshold.setValue(1)
    widget._ba_min_area.setValue(0)
    widget._load()
    assert widget._ba_threshold.value() == 55
    assert widget._ba_min_area.value() == 1200


def test_template_blob_analysis_shows_its_own_form_over_both_stage_forms(page) -> None:
    widget = page.widget
    widget._strategy.setCurrentText("template_blob_analysis")
    assert set(widget._visible_forms()) == {
        "template_blob_analysis", "template_matching", "blob_analysis",
    }
    assert widget._forms["template_matching"].title().startswith("Stage 1")
    assert widget._forms["blob_analysis"].title().startswith("Stage 2")

    widget._strategy.setCurrentText("blob_analysis")  # titles go back on their own
    assert widget._forms["blob_analysis"].title() == "Blob Analysis Parameters"


def test_template_blob_analysis_round_trips_through_save_and_load(page) -> None:
    widget = page.widget
    path = template_file(page.tmp_path)
    widget._strategy.setCurrentText("template_blob_analysis")
    widget._tm_path.setText(path)
    widget._ba_threshold.setValue(66)
    widget._tb_regions.setValue(3)
    widget._tb_padding.setValue(12)
    widget._on_save()

    saved = stored(page)
    assert saved["active_detector"] == "template_blob_analysis"
    assert saved["template_blob_analysis"] == {"max_regions": 3, "region_padding_px": 12}
    assert saved["template_opencv"] == {"max_regions": 1, "region_padding_px": 0}  # its own
    assert saved["template_matching"]["template_path"] == path  # stage 1
    assert saved["blob_analysis"]["threshold"] == 66  # stage 2
    assert page.engine.active_detector_name(1) == "template_blob_analysis"

    widget._tb_regions.setValue(1)
    widget._load()
    assert widget._tb_regions.value() == 3
    assert widget._tb_padding.value() == 12


def test_no_hole_on_template_blob_analysis_says_the_template_was_not_found(page) -> None:
    widget = page.widget
    widget._strategy.setCurrentText("template_blob_analysis")
    assert "template itself was not found" in widget._where_searched(DetectionResult())


def test_blob_analysis_sweep_finds_the_hole_and_fills_the_form(page, monkeypatch) -> None:
    from ui.detection import detection_page

    monkeypatch.setitem(detection_page._SWEEP_LEVELS, "blob_analysis", {
        "threshold": [10, 60, 120],  # 10 is darker than the bore: finds nothing
        "blur_kernel_size": [1, 5],
        "morphology_kernel_size": [3],
        "min_circularity": [0.9, 0.7],
    })
    frame = np.full((400, 400), 180, np.uint8)
    cv2.circle(frame, (200, 210), 30, 20, -1)
    widget = page.widget
    widget._strategy.setCurrentText("blob_analysis")
    widget._ba_min_area.setValue(50_000)  # typed for a far bigger hole
    widget._last_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    roi = (160, 170, 80, 80)
    monkeypatch.setattr(widget._view, "current_roi", lambda: roi)

    widget._on_run_sweep()

    # the form's Min Area would have rejected every trial; the sweep derives
    # it from the ROI instead
    assert widget._sweep_rows, widget._sweep_status.text()
    assert widget._sweep_rows_strategy == "blob_analysis"
    best_params, best_hole = widget._sweep_rows[0]
    assert abs(best_hole.x_px - 200) < 2 and abs(best_hole.y_px - 210) < 2
    assert all(params["threshold"] != 10 for params, _hole in widget._sweep_rows)
    min_diameter, _max = widget._diameter_bounds_from_roi(roi)
    assert best_params["min_area_px"] == int(3.141592653589793 * (min_diameter / 2) ** 2)

    widget._on_apply_best_sweep()
    assert widget._ba_threshold.value() == best_params["threshold"]
    assert widget._ba_circularity.value() == pytest.approx(best_params["min_circularity"])
    assert widget._ba_min_area.value() == best_params["min_area_px"]
    assert widget._ba_min_diameter.value() == round(min_diameter)


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

"""Pages must repaint themselves when a machine model is applied (offscreen Qt).

The station switches model on a PLC register, and
``MachineModelService.apply_profile`` deliberately changes the running
cameras / detectors / calibrations *without* writing camera.json,
detection.json or the calibration database. Nothing else tells the
engineering pages their forms just went stale, so each one listens on
``AppState.active_machine_model_changed`` and re-reads the live state.

These drive the whole path the application uses - apply a profile, then
assert the widgets show the new model's numbers rather than the file's.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from core.calibration import CameraCalibration
from core.calibration.calibration_manager import CalibrationManager
from core.camera import CameraManager
from core.plc import PlcManager, RegisterMap, SimulatedPlc
from core.utilities.config_manager import ConfigManager
from core.vision import VisionEngine
from models.app_state import AppState
from services.camera_service import CameraService
from services.machine_model_service import MachineModelService
from services.plc_service import PlcService
from ui.calibration import CalibrationPage
from ui.camera import CameraPage
from ui.detection import DetectionPage

CAMERA_DOC = {
    "cameras": [
        {
            "index": 1, "name": "Cam 1", "driver": "simulated", "connection_id": "",
            "enabled": True, "exposure_us": 10000, "gain_db": 0.0, "gamma": 1.0,
            "brightness": 10, "fps": 4.0, "rotation": 0,
            "width": 1280, "height": 1024, "trigger_mode": "software",
            "roi": {"x": 0, "y": 0, "width": 0, "height": 0},
        },
    ]
}
DETECTION_DOC = {
    "cameras": {
        "1": {
            "active_detector": "opencv",
            "common": {
                "confidence_threshold": 0.6, "expected_hole_count": 1,
                "position_tolerance_mm": 0.0,
            },
            "opencv": {"detection_threshold": 60, "min_hole_diameter_px": 20},
            "dark_hole": {"min_contrast": 18, "min_hole_diameter_px": 15},
        },
    },
}
# Everything the profile changes, so one apply can be checked on every page.
PROFILE = {
    "id": 1, "name": "Model B", "plc_code": 7,
    "cameras": {
        "1": {"exposure_us": 33000, "roi": {"x": 5, "y": 6, "width": 7, "height": 8}}
    },
    "detection": {
        "cameras": {
            "1": {
                "active_detector": "dark_hole",
                "common": {
                    "confidence_threshold": 0.9, "expected_hole_count": 2,
                    "position_tolerance_mm": 1.5,
                },
                "opencv": {"detection_threshold": 60, "min_hole_diameter_px": 20},
                "dark_hole": {"min_contrast": 42, "min_hole_diameter_px": 15},
            }
        }
    },
    "calibration": {
        "1": CameraCalibration(
            camera_index=1, pixels_per_mm_x=4.0, pixels_per_mm_y=4.0,
            ref_point_mm=(2.0, 3.0),
        ).to_dict()
    },
}


class FakeRepo:
    def upsert(self, values: dict) -> None:
        pass

    def delete_by_index(self, index: int) -> None:
        pass


class FakeDatabaseService:
    def __init__(self) -> None:
        self.camera_configs = FakeRepo()
        self.plc_config = FakeRepo()


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def station(tmp_path, qt_app):
    """The composition root's object graph, cut down to what these pages need."""
    config_dir = tmp_path / "config"
    (config_dir / "defaults").mkdir(parents=True)
    (config_dir / "camera.json").write_text(json.dumps(CAMERA_DOC))
    (config_dir / "detection.json").write_text(json.dumps(DETECTION_DOC))
    (config_dir / "defaults" / "detection.json").write_text(json.dumps(DETECTION_DOC))
    (config_dir / "machine_models.json").write_text(json.dumps({"profiles": []}))
    plc_doc = {
        "connection": {"protocol": "simulated"},
        "registers": {
            "trigger": 100, "machine_number": 101, "heartbeat": 102,
            "result": 118, "vision_complete": 119, "model_select": 103,
            "camera_positions": {"1": {"x": 110, "y": 111}},
        },
    }
    (config_dir / "plc.json").write_text(json.dumps(plc_doc))

    config = ConfigManager(config_dir)
    database = FakeDatabaseService()
    register_map = RegisterMap.from_config(plc_doc)
    plc_manager = PlcManager(SimulatedPlc(register_map=register_map), register_map)
    plc_manager.connect()
    plc_service = PlcService(plc_manager, config, database)
    cameras = CameraService(CameraManager(CAMERA_DOC["cameras"]), config, database, plc_service)
    engine = VisionEngine(DETECTION_DOC)
    calibration = CalibrationManager(SimpleNamespace(get_all_active=lambda: {}))
    app_state = AppState()
    models = MachineModelService(config, cameras, engine, plc_service, calibration)
    return SimpleNamespace(
        config=config, cameras=cameras, engine=engine, calibration=calibration,
        app_state=app_state, models=models,
    )


def _apply(station, page) -> None:
    """Apply the profile exactly the way main.py / the Machine Models page do."""
    assert station.models.apply_profile(PROFILE) == []
    station.app_state.set_active_machine_model(PROFILE["name"], PROFILE["plc_code"])


def test_camera_page_shows_the_applied_exposure_and_roi(station) -> None:
    page = CameraPage(station.app_state, station.cameras)
    assert page._exposure.value() == 10000  # the file's baseline, before the switch

    _apply(station, page)

    assert page._exposure.value() == 33000
    assert [spin.value() for spin in page._roi_spins] == [5, 6, 7, 8]


def test_detection_page_shows_the_applied_strategy_and_thresholds(station) -> None:
    page = DetectionPage(
        station.config, station.engine, station.cameras, station.app_state
    )
    assert page._strategy.currentText() == "opencv"

    _apply(station, page)

    assert page._strategy.currentText() == "dark_hole"
    assert page._expected.value() == 2
    assert page._tolerance.value() == pytest.approx(1.5)
    assert page._dh_min_contrast.value() == 42
    # ...while detection.json still holds the manually maintained baseline.
    assert station.config.load("detection")["cameras"]["1"]["active_detector"] == "opencv"


def test_calibration_page_shows_the_applied_scale(station) -> None:
    page = CalibrationPage(
        station.cameras, station.calibration, station.engine, station.app_state
    )
    assert page._ppmm_x.value() == pytest.approx(0.0)  # no calibration yet

    _apply(station, page)

    assert page._ppmm_x.value() == pytest.approx(4.0)
    assert page._ref_x.value() == pytest.approx(2.0)
    assert page._ref_y.value() == pytest.approx(3.0)

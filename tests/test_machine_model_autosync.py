"""Saving on an engineering page updates the applied machine model at once.

The complaint this pins: a profile only picked up new values when someone
went back to the Machine Models page and pressed "Update Selected from
Current", so a station could run Model A while Model A's stored snapshot
still described the settings from before the last tuning session.

These drive the real pages' Save handlers through the same subscriptions the
composition root registers (``ConfigManager.subscribe`` for camera/detection,
``CalibrationManager.subscribe`` for calibration -> ``MachineModelService.
sync_active_profile``); the fixture reproduces that wiring, cut down to the
objects the pages need, and each test asserts on ``machine_models.json``
itself rather than on the service's return value.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

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
from ui.machine_models import MachineModelsPage

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
                "position_tolerance_mm": 0.0, "normalize_image": False,
            },
            "opencv": {"detection_threshold": 60, "min_hole_diameter_px": 20},
            "dark_hole": {"min_contrast": 18, "min_hole_diameter_px": 15},
        },
    },
}
PLC_DOC = {
    "connection": {"protocol": "simulated"},
    "registers": {
        "trigger": 100, "machine_number": 101, "heartbeat": 102,
        "result": 118, "vision_complete": 119, "model_select": 103,
        "camera_positions": {"1": {"x": 110, "y": 111}},
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
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _silence_modals(monkeypatch):
    """The pages report a successful Save with a modal, which would block."""
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))


@pytest.fixture
def station(tmp_path, qt_app):
    """The composition root's graph and its machine-model sync wiring."""
    config_dir = tmp_path / "config"
    (config_dir / "defaults").mkdir(parents=True)
    (config_dir / "camera.json").write_text(json.dumps(CAMERA_DOC))
    (config_dir / "detection.json").write_text(json.dumps(DETECTION_DOC))
    (config_dir / "defaults" / "detection.json").write_text(json.dumps(DETECTION_DOC))
    (config_dir / "machine_models.json").write_text(json.dumps({"profiles": []}))
    (config_dir / "plc.json").write_text(json.dumps(PLC_DOC))

    config = ConfigManager(config_dir)
    database = FakeDatabaseService()
    register_map = RegisterMap.from_config(PLC_DOC)
    plc_manager = PlcManager(SimulatedPlc(register_map=register_map), register_map)
    plc_manager.connect()
    plc_service = PlcService(plc_manager, config, database)
    manager = CameraManager(CAMERA_DOC["cameras"])
    cameras = CameraService(manager, config, database, plc_service)
    engine = VisionEngine(DETECTION_DOC)
    calibration = CalibrationManager(
        SimpleNamespace(get_all_active=lambda: {}, save=lambda row: 1)
    )
    app_state = AppState()
    models = MachineModelService(config, cameras, engine, plc_service, calibration)
    auth = SimpleNamespace(current_user=SimpleNamespace(username="admin"))

    # --- main.py's wiring, minus the acquisition workers a headless test
    # has no use for. Order matters for "camera": the rebuild runs first so
    # the snapshot is taken from the manager the save just produced.
    config.subscribe("camera", lambda doc: manager.rebuild(doc.get("cameras", [])))
    config.subscribe("camera", lambda _doc: models.sync_active_profile("cameras"))
    config.subscribe("detection", lambda _doc: models.sync_active_profile("detection"))
    calibration.subscribe(lambda _cal: models.sync_active_profile("calibration"))

    return SimpleNamespace(
        config=config, cameras=cameras, engine=engine, calibration=calibration,
        app_state=app_state, models=models, auth=auth,
    )


def _live_profile(station) -> dict:
    """Capture the current settings as a profile and make it the applied one."""
    profile = station.models.capture_current("Model A", 3, created_by="admin")
    assert station.models.apply_profile(profile) == []
    station.app_state.set_active_machine_model(profile["name"], profile["plc_code"])
    return profile


def _stored(station, profile: dict) -> dict:
    """Re-read the profile from machine_models.json, not from the cache."""
    for entry in station.config.load("machine_models", force_reload=True)["profiles"]:
        if entry["id"] == profile["id"]:
            return entry
    raise AssertionError(f"profile {profile['id']} is gone")


def test_camera_page_save_updates_the_applied_profile(station) -> None:
    profile = _live_profile(station)
    assert profile["cameras"]["1"]["exposure_us"] == 10000

    page = CameraPage(station.app_state, station.cameras)
    page._exposure.setValue(27000)
    page._on_save()

    assert _stored(station, profile)["cameras"]["1"]["exposure_us"] == 27000


def test_detection_page_save_updates_the_applied_profile(station) -> None:
    profile = _live_profile(station)
    assert profile["detection"]["cameras"]["1"]["active_detector"] == "opencv"

    page = DetectionPage(
        station.config, station.engine, station.cameras, station.app_state
    )
    page._strategy.setCurrentText("dark_hole")
    page._expected.setValue(3)
    page._on_save()

    stored = _stored(station, profile)["detection"]["cameras"]["1"]
    assert stored["active_detector"] == "dark_hole"
    assert stored["common"]["expected_hole_count"] == 3


def test_calibration_page_save_updates_the_applied_profile(station) -> None:
    profile = _live_profile(station)
    assert profile["calibration"] == {}

    page = CalibrationPage(
        station.cameras, station.calibration, station.engine, station.app_state
    )
    page._ppmm_x.setValue(6.25)
    page._ppmm_y.setValue(6.25)
    page._on_save()

    assert _stored(station, profile)["calibration"]["1"]["pixels_per_mm_x"] == 6.25


def test_saving_without_an_applied_profile_leaves_every_profile_alone(station) -> None:
    """No model is live — the state the station is in between startup and the
    PLC's first model_select read — so a Save has no profile to belong to and
    must not pick an arbitrary one to grow into."""
    first = station.models.capture_current("Model A", 3, created_by="admin")
    other = station.models.capture_current("Model B", 4, created_by="admin")
    station.models.delete(first["id"])  # takes the only claimed target with it
    assert station.models.active_profile_id is None

    page = CameraPage(station.app_state, station.cameras)
    page._exposure.setValue(27000)
    page._on_save()

    assert _stored(station, other)["cameras"]["1"]["exposure_us"] == 10000


def test_machine_models_page_repaints_when_a_save_syncs_the_profile(station) -> None:
    """The page's own tree must not keep showing the pre-save snapshot — an
    automatic sync is the one profile change no button on that page made."""
    profile = _live_profile(station)
    page = MachineModelsPage(station.models, station.auth, station.app_state)
    assert _detection_row(page, "Active Strategy") == "opencv"

    detection_page = DetectionPage(
        station.config, station.engine, station.cameras, station.app_state
    )
    detection_page._strategy.setCurrentText("dark_hole")
    detection_page._on_save()

    assert _detection_row(page, "Active Strategy") == "dark_hole"
    assert profile["name"] in page._status.text()


def _detection_row(page: MachineModelsPage, label: str) -> str:
    """Value of a row under Camera 1 -> Detection in the profile tree."""
    camera_item = page._tree.topLevelItem(0)
    for branch_row in range(camera_item.childCount()):
        branch = camera_item.child(branch_row)
        if branch.text(0) != "Detection":
            continue
        for leaf_row in range(branch.childCount()):
            leaf = branch.child(leaf_row)
            if leaf.text(0) == label:
                return leaf.text(1)
    raise AssertionError(f"no Detection row {label!r} in the profile tree")

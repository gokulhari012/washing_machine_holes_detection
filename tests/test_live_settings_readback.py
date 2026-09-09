"""Live settings must be readable back, or the UI shows the previous model.

``MachineModelService.apply_profile`` is deliberately "preview, don't
persist": it pushes camera ROI/exposure, detection parameters and
calibration into the running objects without writing camera.json,
detection.json or the calibration database. That is the right policy for the
manually maintained baseline, but it means a page that renders those files
describes the *previous* model after a switch. These tests pin the read-back
paths the Camera/Detection/Calibration pages use instead.
"""

import json

from core.calibration import CameraCalibration
from core.camera import CameraManager
from core.camera.camera_base import CameraSettings
from core.plc import PlcManager, RegisterMap, SimulatedPlc
from core.utilities.config_manager import ConfigManager
from core.vision import VisionEngine
from services.camera_service import CameraService
from services.machine_model_service import MachineModelService
from services.plc_service import PlcService

CAMERA_DOC = {
    "cameras": [
        {
            "index": 1, "name": "Cam 1", "driver": "simulated", "connection_id": "",
            "enabled": True, "exposure_us": 10000, "gain_db": 0.0, "gamma": 1.0,
            "brightness": 10, "fps": 4.0, "rotation": 0,
            "width": 1280, "height": 1024, "trigger_mode": "software",
            "roi": {"x": 0, "y": 0, "width": 0, "height": 0},
            "simulation": {"hole_radius_px": 30},
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


class FakeCameraConfigsRepo:
    def upsert(self, values: dict) -> None:
        pass

    def delete_by_index(self, index: int) -> None:
        pass


class FakeDatabaseService:
    def __init__(self) -> None:
        self.camera_configs = FakeCameraConfigsRepo()
        self.plc_config = FakeCameraConfigsRepo()


class FakeCalibrationManager:
    """get/apply_live/save only - enough to prove 'live, not persisted'."""

    def __init__(self) -> None:
        self.live: dict[int, CameraCalibration] = {}
        self.persisted: list[CameraCalibration] = []

    def get(self, camera_index: int):
        return self.live.get(camera_index)

    def apply_live(self, calibration: CameraCalibration) -> None:
        self.live[calibration.camera_index] = calibration

    def save(self, calibration: CameraCalibration) -> int:
        self.persisted.append(calibration)
        self.live[calibration.camera_index] = calibration
        return len(self.persisted)

    def apply_screw_compensation(
        self, enabled: bool, positions: dict[int, tuple[float, float]]
    ) -> None:
        self.screw_compensation = (enabled, positions)


def _build(tmp_path):
    config_dir = tmp_path / "config"
    (config_dir / "defaults").mkdir(parents=True)
    (config_dir / "camera.json").write_text(json.dumps(CAMERA_DOC))
    (config_dir / "detection.json").write_text(json.dumps(DETECTION_DOC))
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
    manager = CameraManager(CAMERA_DOC["cameras"])
    camera_service = CameraService(manager, config, database, plc_service)
    engine = VisionEngine(DETECTION_DOC)
    calibration = FakeCalibrationManager()
    models = MachineModelService(config, camera_service, engine, plc_service, calibration)
    return config, camera_service, engine, calibration, models


# ------------------------------------------------------------ CameraSettings
def test_camera_settings_round_trip_through_to_config() -> None:
    settings = CameraSettings.from_config(CAMERA_DOC["cameras"][0])
    assert CameraSettings.from_config(settings.to_config()) == settings


def test_to_config_keeps_driver_specific_blocks() -> None:
    """``extra`` carries the raw entry - a driver's own block must survive."""
    cfg = CameraSettings.from_config(CAMERA_DOC["cameras"][0]).to_config()
    assert cfg["simulation"] == {"hole_radius_px": 30}
    assert cfg["roi"] == {"x": 0, "y": 0, "width": 0, "height": 0}


# ------------------------------------------------------------- CameraService
def test_effective_configs_show_apply_live_overrides(tmp_path) -> None:
    _config, cameras, *_ = _build(tmp_path)
    cameras.apply_live(dict(CAMERA_DOC["cameras"][0], exposure_us=25000))

    assert cameras.get_configs()[0]["exposure_us"] == 10000            # file untouched
    assert cameras.get_effective_configs()[0]["exposure_us"] == 25000  # what is running
    assert cameras.effective_config(1)["exposure_us"] == 25000
    assert cameras.effective_config(99) is None


def test_effective_configs_fall_back_to_the_file_for_an_unbuilt_camera(tmp_path) -> None:
    """A camera in camera.json but not in the manager keeps its file entry."""
    config, cameras, *_ = _build(tmp_path)
    document = config.load("camera")
    document["cameras"].append(dict(CAMERA_DOC["cameras"][0], index=7, name="Cam 7"))
    config.save("camera", document)

    effective = {int(cfg["index"]): cfg for cfg in cameras.get_effective_configs()}
    assert effective[7]["name"] == "Cam 7"


# -------------------------------------------------------------- VisionEngine
def test_camera_config_reflects_the_live_hot_swap() -> None:
    engine = VisionEngine(DETECTION_DOC)
    engine.apply_camera_config(
        1, dict(DETECTION_DOC["cameras"]["1"], active_detector="dark_hole")
    )
    assert engine.camera_config(1)["active_detector"] == "dark_hole"
    assert engine.camera_config(99) is None


def test_camera_config_is_a_copy_the_caller_cannot_corrupt() -> None:
    engine = VisionEngine(DETECTION_DOC)
    engine.camera_config(1)["common"]["expected_hole_count"] = 99
    assert engine.expected_hole_count(1) == 1


# ------------------------------------------------------------------- applied
def test_applying_a_profile_is_visible_to_the_pages_but_not_the_files(tmp_path) -> None:
    """The whole point: after a model switch every page can read the new
    values, while camera.json / detection.json still hold the baseline."""
    config, cameras, engine, calibration, models = _build(tmp_path)
    profile = {
        "id": 1, "name": "Model B", "plc_code": 7,
        "cameras": {
            "1": {"exposure_us": 33000, "roi": {"x": 5, "y": 6, "width": 7, "height": 8}}
        },
        "detection": {
            "cameras": {
                "1": dict(
                    DETECTION_DOC["cameras"]["1"],
                    active_detector="dark_hole",
                    common={
                        "confidence_threshold": 0.9, "expected_hole_count": 2,
                        "position_tolerance_mm": 1.5,
                    },
                )
            }
        },
        "calibration": {
            "1": CameraCalibration(
                camera_index=1, pixels_per_mm_x=4.0, pixels_per_mm_y=4.0,
                ref_point_mm=(2.0, 3.0),
            ).to_dict()
        },
    }
    assert models.apply_profile(profile) == []

    effective = cameras.effective_config(1)
    assert effective["exposure_us"] == 33000
    assert effective["roi"] == {"x": 5, "y": 6, "width": 7, "height": 8}
    assert engine.camera_config(1)["active_detector"] == "dark_hole"
    assert engine.camera_config(1)["common"]["expected_hole_count"] == 2
    assert calibration.get(1).pixels_per_mm_x == 4.0

    # ...and nothing was persisted over the manually maintained baseline.
    assert config.load("camera")["cameras"][0]["exposure_us"] == 10000
    assert config.load("detection")["cameras"]["1"]["active_detector"] == "opencv"
    assert calibration.persisted == []


def test_a_legacy_flat_profile_is_still_readable_back(tmp_path) -> None:
    """A pre-per-camera profile is migrated on apply - the read-back path
    must show the migrated block, not None."""
    _config, _cameras, engine, _calibration, models = _build(tmp_path)
    flat = dict(DETECTION_DOC["cameras"]["1"], active_detector="dark_hole")
    models.apply_profile({"id": 1, "name": "Old", "plc_code": 3, "detection": flat})
    assert engine.camera_config(1)["active_detector"] == "dark_hole"

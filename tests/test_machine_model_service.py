"""MachineModelService: profile capture/apply, JSON-only persistence.

Camera application is exercised against a lightweight fake that mimics only
the two CameraService methods the service actually calls (get_configs,
apply_live) — validated the same way the real one is, via
CameraSettings.from_config, so a malformed merge would still be caught.
"""

import json

import pytest

from core.camera.camera_base import CameraSettings
from core.utilities.config_manager import ConfigManager
from core.utilities.exceptions import ConfigurationError, VisionSystemError
from core.vision import VisionEngine
from services.machine_model_service import MachineModelService

CAMERA_DOC = {
    "cameras": [
        {
            "index": 1, "name": "Cam 1", "driver": "simulated", "connection_id": "",
            "enabled": True, "exposure_us": 10000, "gain_db": 0.0, "gamma": 1.0,
            "brightness": 0, "width": 1280, "height": 1024, "trigger_mode": "software",
            "roi": {"x": 0, "y": 0, "width": 0, "height": 0},
        },
    ]
}
DETECTION_DOC = {
    "active_detector": "opencv",
    "common": {
        "confidence_threshold": 0.6, "expected_hole_count": 1, "position_tolerance_mm": 0.0,
    },
    "opencv": {
        "detection_threshold": 60, "blur_kernel_size": 5, "morphology_operation": "close",
        "morphology_kernel_size": 5, "morphology_iterations": 1, "min_hole_diameter_px": 20,
        "max_hole_diameter_px": 200, "min_circularity": 0.7,
        "edge_threshold_low": 50, "edge_threshold_high": 150,
    },
    "dark_hole": {
        "channel": "auto", "blur_kernel_size": 3, "min_contrast": 18, "use_otsu": True,
        "morphology_kernel_size": 3, "min_hole_diameter_px": 15, "max_hole_diameter_px": 120,
        "min_fill_ratio": 0.35, "max_fit_error": 0.25,
    },
}


class FakeCameraService:
    """Mimics the two CameraService methods MachineModelService calls."""

    def __init__(self, configs: list[dict]) -> None:
        self._configs = configs
        self.applied: list[dict] = []

    def get_configs(self) -> list[dict]:
        return [dict(c) for c in self._configs]

    def apply_live(self, camera_config: dict) -> None:
        CameraSettings.from_config(camera_config)  # raises ConfigurationError if malformed
        self.applied.append(camera_config)


class FakePlcService:
    """Mimics the two PlcService camera-position methods MachineModelService calls."""

    def __init__(self) -> None:
        self.positions: dict[int, tuple[int, int]] = {}
        self.rejects: set[int] = set()

    def read_camera_position(self, camera_index: int) -> tuple[int, int]:
        return self.positions.get(camera_index, (0, 0))

    def set_camera_position(self, camera_index: int, x: int, y: int) -> tuple[int, int]:
        if camera_index in self.rejects:
            raise VisionSystemError(f"no jog registers configured for camera {camera_index}")
        self.positions[camera_index] = (x, y)
        return x, y


def make_service(tmp_path):
    config_dir = tmp_path / "config"
    (config_dir / "defaults").mkdir(parents=True)
    (config_dir / "camera.json").write_text(json.dumps(CAMERA_DOC))
    (config_dir / "detection.json").write_text(json.dumps(DETECTION_DOC))
    (config_dir / "machine_models.json").write_text(json.dumps({"profiles": []}))

    config = ConfigManager(config_dir)
    cameras = FakeCameraService(CAMERA_DOC["cameras"])
    engine = VisionEngine(DETECTION_DOC)
    plc = FakePlcService()
    return MachineModelService(config, cameras, engine, plc), cameras, engine, plc


def test_capture_then_get_by_code_round_trip(tmp_path) -> None:
    service, _cameras, _engine, _plc = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    assert profile["plc_code"] == 3
    assert service.get_by_code(3)["name"] == "Model A"
    assert service.get_by_id(profile["id"]) is not None
    assert service.list_profiles() == [profile]


def test_duplicate_plc_code_rejected(tmp_path) -> None:
    service, _cameras, _engine, _plc = make_service(tmp_path)
    service.capture_current("Model A", 3, created_by="admin")
    with pytest.raises(ConfigurationError):
        service.capture_current("Model B", 3, created_by="admin")


def test_blank_name_rejected(tmp_path) -> None:
    service, _cameras, _engine, _plc = make_service(tmp_path)
    with pytest.raises(ConfigurationError):
        service.capture_current("   ", 1, created_by="admin")


def test_apply_profile_merges_tunable_fields_only(tmp_path) -> None:
    service, cameras, _engine, _plc = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    profile["cameras"]["1"]["roi"] = {"x": 10, "y": 20, "width": 300, "height": 200}
    profile["cameras"]["1"]["exposure_us"] = 25000

    warnings = service.apply_profile(profile)
    assert warnings == []
    assert len(cameras.applied) == 1
    applied = cameras.applied[0]
    assert applied["roi"] == {"x": 10, "y": 20, "width": 300, "height": 200}
    assert applied["exposure_us"] == 25000
    assert applied["driver"] == "simulated"  # identity field untouched by the profile
    assert applied["connection_id"] == ""


def test_apply_profile_skips_missing_camera_with_warning(tmp_path) -> None:
    service, cameras, _engine, _plc = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    profile["cameras"]["9"] = profile["cameras"].pop("1")

    warnings = service.apply_profile(profile)
    assert len(warnings) == 1
    assert "camera 9" in warnings[0]
    assert cameras.applied == []


def test_apply_profile_hot_swaps_detection(tmp_path) -> None:
    service, _cameras, engine, _plc = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    profile["detection"] = dict(DETECTION_DOC, active_detector="dark_hole")

    service.apply_profile(profile)
    assert engine.active_detector_name == "dark_hole"


def test_apply_profile_raises_on_bad_detection_block(tmp_path) -> None:
    service, _cameras, _engine, _plc = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    profile["detection"] = dict(DETECTION_DOC, active_detector="not_a_real_detector")
    with pytest.raises(VisionSystemError):
        service.apply_profile(profile)


def test_update_from_current_keeps_name_and_code(tmp_path) -> None:
    service, cameras, _engine, _plc = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    cameras._configs[0]["exposure_us"] = 99999

    updated = service.update_from_current(profile["id"], updated_by="admin2")
    assert updated["name"] == "Model A"
    assert updated["plc_code"] == 3
    assert updated["cameras"]["1"]["exposure_us"] == 99999


def test_rename_validates_and_persists(tmp_path) -> None:
    service, _cameras, _engine, _plc = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    service.rename(profile["id"], "Model A2", 4)
    assert service.get_by_code(4)["name"] == "Model A2"
    assert service.get_by_code(3) is None


def test_rename_rejects_code_already_used_by_another_profile(tmp_path) -> None:
    service, _cameras, _engine, _plc = make_service(tmp_path)
    profile_a = service.capture_current("Model A", 3, created_by="admin")
    service.capture_current("Model B", 4, created_by="admin")
    with pytest.raises(ConfigurationError):
        service.rename(profile_a["id"], "Model A", 4)


def test_delete_removes_profile(tmp_path) -> None:
    service, _cameras, _engine, _plc = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    service.delete(profile["id"])
    assert service.list_profiles() == []
    with pytest.raises(ConfigurationError):
        service.delete(profile["id"])


def test_save_camera_position_persists_on_profile(tmp_path) -> None:
    service, _cameras, _engine, _plc = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")

    updated = service.save_camera_position(profile["id"], 1, 250, 340, updated_by="admin")
    assert updated["jog_positions"]["1"] == {"x": 250, "y": 340}
    assert service.get_by_id(profile["id"])["jog_positions"]["1"] == {"x": 250, "y": 340}


def test_save_camera_position_rejects_unknown_profile(tmp_path) -> None:
    service, _cameras, _engine, _plc = make_service(tmp_path)
    with pytest.raises(ConfigurationError):
        service.save_camera_position(999, 1, 0, 0, updated_by="admin")


def test_apply_profile_pushes_saved_jog_positions(tmp_path) -> None:
    service, _cameras, _engine, plc = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    profile = service.save_camera_position(profile["id"], 1, 250, 340, updated_by="admin")

    warnings = service.apply_profile(profile)
    assert warnings == []
    assert plc.positions[1] == (250, 340)


def test_apply_profile_warns_when_position_rejected(tmp_path) -> None:
    service, _cameras, _engine, plc = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    profile = service.save_camera_position(profile["id"], 1, 250, 340, updated_by="admin")
    plc.rejects.add(1)

    warnings = service.apply_profile(profile)
    assert len(warnings) == 1
    assert "camera 1" in warnings[0]
    assert 1 not in plc.positions

"""CameraService: apply_live/save_camera also push brightness to the PLC.

``brightness`` is a 0-255 light-brightness level for an external,
PLC-controlled light source (see camera_base.py), not an in-camera setting —
every apply/save must also write it to that camera's PLC register, and must
never let a PLC problem block the camera settings themselves from applying.
"""

import json

import pytest

from core.plc import PlcManager, RegisterMap, SimulatedPlc
from core.utilities.config_manager import ConfigManager
from services.camera_service import CameraService
from services.plc_service import PlcService

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


class FakeCameraManager:
    """Mimics the one CameraManager method CameraService.apply_live calls."""

    def __init__(self) -> None:
        self.applied: dict[int, object] = {}

    def apply_settings(self, index, settings) -> None:
        self.applied[index] = settings


class FakeCameraConfigsRepo:
    def upsert(self, values: dict) -> None:
        pass

    def delete_by_index(self, index: int) -> None:
        pass


class FakeDatabaseService:
    def __init__(self) -> None:
        self.camera_configs = FakeCameraConfigsRepo()
        self.plc_config = FakeCameraConfigsRepo()  # PlcService._mirror_to_database target


def make_service(tmp_path, *, brightness_register: dict | None = None):
    config_dir = tmp_path / "config"
    (config_dir / "defaults").mkdir(parents=True)
    (config_dir / "camera.json").write_text(json.dumps(CAMERA_DOC))

    plc_doc = {
        "connection": {"protocol": "simulated"},
        "registers": {
            "trigger": 100, "machine_number": 101, "heartbeat": 102,
            "result": 118, "vision_complete": 119,
            "camera_positions": {"1": {"x": 110, "y": 111}},
        },
    }
    if brightness_register is not None:
        plc_doc["registers"]["camera_brightness"] = brightness_register
    (config_dir / "plc.json").write_text(json.dumps(plc_doc))

    config = ConfigManager(config_dir)
    database = FakeDatabaseService()
    rmap = RegisterMap.from_config(plc_doc)
    plc_manager = PlcManager(SimulatedPlc(register_map=rmap), rmap)
    plc_manager.connect()
    plc_service = PlcService(plc_manager, config, database)
    cameras = FakeCameraManager()
    service = CameraService(cameras, config, database, plc_service)
    return service, cameras, plc_manager


def _camera_config(**overrides) -> dict:
    return dict(CAMERA_DOC["cameras"][0], **overrides)


def test_apply_live_pushes_brightness_to_configured_register(tmp_path) -> None:
    service, cameras, plc_manager = make_service(
        tmp_path, brightness_register={"1": 156}
    )
    service.apply_live(_camera_config(brightness=180))

    assert cameras.applied[1].brightness == 180  # camera settings still applied
    assert plc_manager._client.get_register(156) == 180  # and pushed to the PLC


def test_save_camera_pushes_brightness_to_configured_register(tmp_path) -> None:
    service, _cameras, plc_manager = make_service(
        tmp_path, brightness_register={"1": 156}
    )
    service.save_camera(_camera_config(brightness=99))
    assert plc_manager._client.get_register(156) == 99


def test_apply_live_is_unaffected_when_brightness_register_unconfigured(tmp_path) -> None:
    """No camera_brightness block at all — apply_live must still succeed."""
    service, cameras, _plc_manager = make_service(tmp_path, brightness_register=None)
    service.apply_live(_camera_config(brightness=180))
    assert cameras.applied[1].brightness == 180


def test_apply_live_survives_a_plc_write_failure(tmp_path) -> None:
    """A PLC communication failure must never block the camera's own settings."""
    from core.utilities.exceptions import PlcError

    service, cameras, plc_manager = make_service(tmp_path, brightness_register={"1": 156})

    def _boom(*_args, **_kwargs):
        raise PlcError("no PLC")

    plc_manager._client.write_register = _boom

    service.apply_live(_camera_config(brightness=180))  # must not raise
    assert cameras.applied[1].brightness == 180

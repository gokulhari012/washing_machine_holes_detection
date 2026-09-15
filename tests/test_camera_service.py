"""CameraService: apply_live/save_camera also push brightness to the LED
Controller.

``brightness`` is a 0-255 light-brightness level for an external LED light
source (see camera_base.py), not an in-camera setting — every apply/save
must also write it to that camera's configured ``led_channel``, and must
never let an LED communication problem block the camera settings themselves
from applying.
"""

import json

from core.led import LedControllerSettings, LedManager, SimulatedLedClient
from core.utilities.config_manager import ConfigManager
from services.camera_service import CameraService
from services.led_service import LedService

CAMERA_DOC = {
    "cameras": [
        {
            "index": 1, "name": "Cam 1", "driver": "simulated", "connection_id": "",
            "enabled": True, "exposure_us": 10000, "gain_db": 0.0, "gamma": 1.0,
            "brightness": 0, "led_channel": 0, "led_strobe": False,
            "width": 1280, "height": 1024, "trigger_mode": "software",
            "roi": {"x": 0, "y": 0, "width": 0, "height": 0},
        },
    ]
}


class FakeCameraManager:
    """Mimics the CameraManager surface CameraService touches: apply_settings
    (apply_live) and the .cameras dict get_effective_configs()/light_on/
    light_off read (empty here, so effective_config falls back to the
    persisted camera.json entry, same as a camera that failed to construct)."""

    def __init__(self) -> None:
        self.applied: dict[int, object] = {}
        self.cameras: dict[int, object] = {}

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


def make_service(tmp_path):
    config_dir = tmp_path / "config"
    (config_dir / "defaults").mkdir(parents=True)
    (config_dir / "camera.json").write_text(json.dumps(CAMERA_DOC))

    config = ConfigManager(config_dir)
    database = FakeDatabaseService()
    led_client = SimulatedLedClient()
    led_manager = LedManager(led_client, LedControllerSettings())
    led_manager.connect()
    led_service = LedService(led_manager, config)
    cameras = FakeCameraManager()
    service = CameraService(cameras, config, database, led_service)
    return service, cameras, led_client


def _camera_config(**overrides) -> dict:
    return dict(CAMERA_DOC["cameras"][0], **overrides)


def test_apply_live_pushes_brightness_to_configured_channel(tmp_path) -> None:
    service, cameras, led_client = make_service(tmp_path)
    service.apply_live(_camera_config(brightness=180, led_channel=1))

    assert cameras.applied[1].brightness == 180  # camera settings still applied
    assert led_client.sent == ["SA0180#"]  # and pushed to the LED controller


def test_save_camera_pushes_brightness_to_configured_channel(tmp_path) -> None:
    service, _cameras, led_client = make_service(tmp_path)
    service.save_camera(_camera_config(brightness=99, led_channel=2))
    assert led_client.sent == ["SB0099#"]


def test_apply_live_is_unaffected_when_led_channel_unconfigured(tmp_path) -> None:
    """led_channel left at 0 ("not used") — apply_live must still succeed, no I/O."""
    service, cameras, led_client = make_service(tmp_path)
    service.apply_live(_camera_config(brightness=180, led_channel=0))
    assert cameras.applied[1].brightness == 180
    assert led_client.sent == []


def test_apply_live_survives_an_led_communication_failure(tmp_path) -> None:
    """An LED communication failure must never block the camera's own settings."""
    service, cameras, led_client = make_service(tmp_path)
    led_client.disconnect()  # simulate a lost link

    service.apply_live(_camera_config(brightness=180, led_channel=1))  # must not raise
    assert cameras.applied[1].brightness == 180


# --------------------------------------------------------------- strobe mode
def test_save_camera_skips_the_brightness_push_while_strobe_is_on(tmp_path) -> None:
    """Strobe mode's resting state is off - Save must not light the channel."""
    service, _cameras, led_client = make_service(tmp_path)
    service.save_camera(_camera_config(brightness=150, led_channel=3, led_strobe=True))
    assert led_client.sent == []


def test_save_camera_still_pushes_brightness_when_strobe_is_off(tmp_path) -> None:
    service, _cameras, led_client = make_service(tmp_path)
    service.save_camera(_camera_config(brightness=150, led_channel=3, led_strobe=False))
    assert led_client.sent == ["SC0150#"]


def test_light_on_writes_the_configured_brightness_to_the_channel(tmp_path) -> None:
    service, _cameras, led_client = make_service(tmp_path)
    service.save_camera(_camera_config(brightness=150, led_channel=3, led_strobe=True))
    led_client.sent.clear()  # save itself pushed nothing (strobe on); start fresh

    service.light_on(1)
    assert led_client.sent == ["SC0150#"]


def test_light_off_writes_zero_to_the_channel(tmp_path) -> None:
    service, _cameras, led_client = make_service(tmp_path)
    service.save_camera(_camera_config(brightness=150, led_channel=3, led_strobe=True))
    led_client.sent.clear()

    service.light_off(1)
    assert led_client.sent == ["SC0000#"]


def test_light_on_and_off_are_a_noop_when_channel_unconfigured(tmp_path) -> None:
    service, _cameras, led_client = make_service(tmp_path)
    service.save_camera(_camera_config(brightness=150, led_channel=0, led_strobe=True))
    led_client.sent.clear()

    service.light_on(1)
    service.light_off(1)
    assert led_client.sent == []


def test_light_on_survives_an_led_communication_failure(tmp_path) -> None:
    service, _cameras, led_client = make_service(tmp_path)
    service.save_camera(_camera_config(brightness=150, led_channel=3, led_strobe=True))
    led_client.disconnect()  # simulate a lost link

    service.light_on(1)  # must not raise
    service.light_off(1)  # must not raise

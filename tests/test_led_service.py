"""LedService: config persistence/validation and the command facade both
CameraService (per-camera brightness) and the LED Controller page (raw
command tester) call into."""

import json

import pytest

from core.led import LedControllerSettings, LedManager, SimulatedLedClient
from core.utilities.config_manager import ConfigManager
from core.utilities.exceptions import ConfigurationError
from services.led_service import LedService

LED_DOC = {
    "connection": {"driver": "simulated", "port": "", "baud_rate": 19200, "timeout_ms": 1000},
    "max_brightness": 255,
}


def make_service(tmp_path) -> tuple[LedService, SimulatedLedClient, LedManager]:
    config_dir = tmp_path / "config"
    (config_dir / "defaults").mkdir(parents=True)
    (config_dir / "led.json").write_text(json.dumps(LED_DOC))

    config = ConfigManager(config_dir)
    client = SimulatedLedClient()
    manager = LedManager(client, LedControllerSettings())
    service = LedService(manager, config)
    return service, client, manager


def test_get_config_round_trips_the_saved_document(tmp_path) -> None:
    service, _client, _manager = make_service(tmp_path)
    assert service.get_config() == LED_DOC


def test_save_config_persists_to_disk(tmp_path) -> None:
    service, _client, _manager = make_service(tmp_path)
    cfg = service.get_config()
    cfg["connection"]["port"] = "COM7"
    service.save_config(cfg)
    assert service.get_config()["connection"]["port"] == "COM7"


def test_save_config_rejects_out_of_range_max_brightness(tmp_path) -> None:
    service, _client, _manager = make_service(tmp_path)
    cfg = service.get_config()
    cfg["max_brightness"] = 999
    with pytest.raises(ConfigurationError):
        service.save_config(cfg)
    # the bad value must never reach disk
    assert service.get_config()["max_brightness"] == 255


def test_connect_and_disconnect_reach_the_manager(tmp_path) -> None:
    service, client, _manager = make_service(tmp_path)
    service.connect()
    assert client.connected is True
    service.disconnect()
    assert client.connected is False


def test_set_channel_brightness_delegates_to_manager(tmp_path) -> None:
    service, client, _manager = make_service(tmp_path)
    service.connect()
    response = service.set_channel_brightness(2, 128)
    assert response == "!"
    assert client.sent == ["SB0128#"]


def test_send_raw_is_not_clamped(tmp_path) -> None:
    service, client, _manager = make_service(tmp_path)
    service.connect()
    service.send_raw("S100T128T025F000TC#")
    assert client.sent == ["S100T128T025F000TC#"]

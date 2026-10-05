"""LED Controller page: manual light control, and each camera's light
brightness set from the LED page.

Two things that look alike and must stay apart:

- **manual** levels (``led.json``'s ``manual_brightness``) drive only the
  page's ON/OFF buttons. They never touch a camera's ``brightness``, so the
  inspection cycle is unaffected by them;
- **camera** brightness set here is the Camera page's "Light Brightness":
  persisted to camera.json (that key only), live on the running camera for
  the next cycle, pushed to its channel, and announced so the Camera page and
  the applied machine model follow — all *without* the camera rebuild that a
  notifying camera.json save triggers.
"""

from __future__ import annotations

import copy
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from core.camera import CameraManager
from core.led import LedControllerSettings, LedManager, SimulatedLedClient
from core.utilities.config_manager import ConfigManager
from core.utilities.exceptions import ConfigurationError
from models.app_state import AppState
from services.camera_service import CameraService
from services.led_service import DEFAULT_MANUAL_BRIGHTNESS, LedService


def _camera(index: int, **overrides) -> dict:
    entry = {
        "index": index, "name": f"Cam {index}", "driver": "simulated", "connection_id": "",
        "enabled": True, "exposure_us": 10000, "gain_db": 0.0, "gamma": 1.0,
        "brightness": 50, "led_channel": index, "led_strobe": False,
        "width": 640, "height": 480, "trigger_mode": "software",
        "roi": {"x": 0, "y": 0, "width": 0, "height": 0},
    }
    entry.update(overrides)
    return entry


CAMERA_DOC = {"cameras": [_camera(1), _camera(2, led_strobe=True), _camera(3, led_channel=0)]}
LED_DOC = {
    "connection": {"driver": "simulated", "port": "", "baud_rate": 19200, "timeout_ms": 1000},
    "max_brightness": 200,
}


class FakeCameraConfigsRepo:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def upsert(self, values: dict) -> None:
        self.rows.append(values)

    def delete_by_index(self, index: int) -> None:
        pass


class FakeDatabaseService:
    def __init__(self) -> None:
        self.camera_configs = FakeCameraConfigsRepo()


class Rig:
    def __init__(self, tmp_path) -> None:
        config_dir = tmp_path / "config"
        (config_dir / "defaults").mkdir(parents=True)
        (config_dir / "camera.json").write_text(json.dumps(CAMERA_DOC))
        (config_dir / "led.json").write_text(json.dumps(LED_DOC))
        self.config_dir = config_dir
        self.config = ConfigManager(config_dir)
        self.led_client = SimulatedLedClient()
        self.led_manager = LedManager(self.led_client, LedControllerSettings(max_brightness=200))
        self.led_manager.connect()
        self.led = LedService(self.led_manager, self.config)
        self.cameras = CameraManager(copy.deepcopy(CAMERA_DOC["cameras"]))
        self.database = FakeDatabaseService()
        self.camera_service = CameraService(self.cameras, self.config, self.database, self.led)
        # Anything subscribed to these would be the composition root's
        # rebuild/reconnect handlers — neither may fire for these edits.
        self.camera_saves: list[dict] = []
        self.led_saves: list[dict] = []
        self.config.subscribe("camera", self.camera_saves.append)
        self.config.subscribe("led", self.led_saves.append)

    def disk(self, name: str) -> dict:
        return json.loads((self.config_dir / f"{name}.json").read_text())


@pytest.fixture
def rig(tmp_path) -> Rig:
    return Rig(tmp_path)


# ----------------------------------------------------------- manual control
def test_manual_levels_default_when_led_json_has_none(rig) -> None:
    assert rig.led.manual_levels() == {ch: DEFAULT_MANUAL_BRIGHTNESS for ch in (1, 2, 3, 4)}


def test_manual_levels_persist_without_reconnecting_the_led_link(rig) -> None:
    rig.led.save_manual_levels({2: 90})
    assert rig.disk("led")["manual_brightness"] == {"1": 128, "2": 90, "3": 128, "4": 128}
    assert rig.led.manual_levels()[2] == 90
    assert rig.led_saves == []  # no rebuild + reconnect of the serial port
    assert rig.disk("led")["connection"] == LED_DOC["connection"]  # rest untouched


def test_manual_levels_reject_out_of_range(rig) -> None:
    with pytest.raises(ConfigurationError):
        rig.led.save_manual_levels({1: 256})
    with pytest.raises(ConfigurationError):
        rig.led.save_manual_levels({5: 10})
    assert "manual_brightness" not in rig.disk("led")


def test_manual_commands_are_clamped_single_channel_writes(rig) -> None:
    rig.led.manual_on(1, 150)
    rig.led.manual_on(2, 255)  # above max_brightness 200
    rig.led.manual_off(1)
    assert rig.led_client.sent == ["SA0150#", "SB0200#", "SA0000#"]


def test_manual_all_on_uses_each_channels_own_level_and_all_off_zeroes(rig) -> None:
    rig.led.manual_all_on({1: 10, 2: 20, 3: 30, 4: 250})
    rig.led.manual_all_off()
    assert rig.led_client.sent == [
        "SA0010#", "SB0020#", "SC0030#", "SD0200#",
        "SA0000#", "SB0000#", "SC0000#", "SD0000#",
    ]


def test_manual_control_never_changes_a_cameras_brightness(rig) -> None:
    rig.led.manual_all_on({1: 10, 2: 20, 3: 30, 4: 40})
    rig.led.save_manual_levels({1: 10})
    assert [c.settings.brightness for c in rig.cameras.cameras.values()] == [50, 50, 50]
    assert [c["brightness"] for c in rig.disk("camera")["cameras"]] == [50, 50, 50]


# --------------------------------------------------------- camera brightness
def test_set_brightness_is_live_for_the_next_cycle(rig) -> None:
    rig.camera_service.set_brightness({1: 180})
    # InspectionService reads camera.settings.brightness for its strobe.
    assert rig.cameras.get(1).settings.brightness == 180
    assert rig.camera_service.effective_config(1)["brightness"] == 180


def test_set_brightness_persists_only_brightness_without_a_camera_rebuild(rig) -> None:
    rig.camera_service.set_brightness({1: 180})
    on_disk = rig.disk("camera")["cameras"]
    assert on_disk[0] == _camera(1, brightness=180)
    assert on_disk[1:] == CAMERA_DOC["cameras"][1:]
    assert rig.camera_saves == []  # the rebuild/reconnect subscriber never ran
    assert rig.database.camera_configs.rows[-1]["brightness"] == 180


def test_set_brightness_does_not_push_settings_to_the_device(rig, monkeypatch) -> None:
    camera = rig.cameras.get(1)
    monkeypatch.setattr(camera, "_connected", True)
    pushed: list = []
    monkeypatch.setattr(camera, "_apply_to_device", pushed.append)
    rig.camera_service.set_brightness({1: 99})
    assert pushed == []


def test_set_brightness_lights_steady_cameras_only(rig) -> None:
    rig.camera_service.set_brightness({1: 180, 2: 70, 3: 60})
    # 1 is steady on channel 1 -> pushed; 2 is strobe (lit per capture);
    # 3 has no channel.
    assert rig.led_client.sent == ["SA0180#"]


def test_set_brightness_notifies_observers_once_with_every_level(rig) -> None:
    seen: list[dict] = []
    rig.camera_service.subscribe_brightness(seen.append)
    rig.camera_service.set_brightness({1: 10, 2: 20, 3: 30})
    assert seen == [{1: 10, 2: 20, 3: 30}]


@pytest.mark.parametrize("levels", [{9: 10}, {1: 300}, {1: 10, 2: -1}])
def test_set_brightness_is_all_or_nothing_on_bad_input(rig, levels) -> None:
    with pytest.raises(ConfigurationError):
        rig.camera_service.set_brightness(levels)
    assert rig.disk("camera") == CAMERA_DOC
    assert rig.cameras.get(1).settings.brightness == 50
    assert rig.led_client.sent == []


def test_set_brightness_survives_a_disconnected_led_controller(rig) -> None:
    rig.led_manager.disconnect()
    rig.camera_service.set_brightness({1: 111})  # push fails, logged, not raised
    assert rig.cameras.get(1).settings.brightness == 111
    assert rig.disk("camera")["cameras"][0]["brightness"] == 111


# -------------------------------------------------------------------- pages
@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_led_page_camera_set_reaches_the_camera_page(rig, qapp) -> None:
    from ui.camera.camera_page import CameraPage
    from ui.led.led_page import LedPage

    state = AppState()
    # What main.py wires: the service's callback onto the AppState signal.
    rig.camera_service.subscribe_brightness(
        lambda levels: [state.camera_brightness_changed.emit(i, v) for i, v in levels.items()]
    )
    led_page = LedPage(state, rig.led, rig.camera_service)
    camera_page = CameraPage(state, rig.camera_service)
    camera_page._list.setCurrentRow(0)  # camera 1
    assert camera_page._brightness.value() == 50

    led_page._camera_spins[1].setValue(222)
    led_page._on_set_camera_brightness(1)

    assert camera_page._brightness.value() == 222
    assert rig.cameras.get(1).settings.brightness == 222


def test_led_page_set_all_sets_every_camera(rig, qapp, monkeypatch) -> None:
    from ui.led.led_page import LedPage

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    page = LedPage(AppState(), rig.led, rig.camera_service)
    page._all_cameras_spin.setValue(77)
    page._on_set_all_cameras()
    assert [c.settings.brightness for c in rig.cameras.cameras.values()] == [77, 77, 77]


def test_led_page_manual_on_uses_the_manual_level_not_the_camera_level(rig, qapp) -> None:
    from ui.led.led_page import LedPage

    page = LedPage(AppState(), rig.led, rig.camera_service)
    page._manual_spins[1].setValue(33)
    page._on_manual_on(1)
    assert rig.led_client.sent == ["SA0033#"]
    assert rig.cameras.get(1).settings.brightness == 50
    assert rig.led.manual_levels()[1] == 33

    # While on, the channel follows its spin box.
    page._manual_spins[1].setValue(44)
    assert rig.led_client.sent[-1] == "SA0044#"

    page._on_manual_off(1)
    assert rig.led_client.sent[-1] == "SA0000#"
    page._manual_spins[1].setValue(55)  # off: no re-send
    assert rig.led_client.sent[-1] == "SA0000#"


def test_led_page_manual_buttons_follow_the_connection(rig, qapp) -> None:
    from ui.led.led_page import LedPage

    page = LedPage(AppState(), rig.led, rig.camera_service)
    assert all(button.isEnabled() for button in page._manual_buttons)
    page._on_led_state_changed("disconnected")
    assert not any(button.isEnabled() for button in page._manual_buttons)


def test_led_page_shows_channel_wiring(rig, qapp) -> None:
    from ui.led.led_page import LedPage

    page = LedPage(AppState(), rig.led, rig.camera_service)
    assert page._manual_wiring[1].text() == "Cam 1"
    assert page._manual_wiring[2].text() == "Cam 2 (strobe)"
    assert page._manual_wiring[3].text() == "not wired"
    assert page._camera_wiring[3].text() == "no LED channel"


# ------------------------------------------------------------- camera strobe
def test_set_strobe_is_live_and_persists_only_led_strobe(rig) -> None:
    rig.camera_service.set_strobe({1: True})
    # InspectionService reads camera.settings.led_strobe for its strobe.
    assert rig.cameras.get(1).settings.led_strobe is True
    on_disk = rig.disk("camera")["cameras"]
    assert on_disk[0] == _camera(1, led_strobe=True)
    assert on_disk[1:] == CAMERA_DOC["cameras"][1:]
    assert rig.camera_saves == []  # no camera rebuild/reconnect
    assert rig.database.camera_configs.rows[-1]["led_strobe"] is True


def test_set_strobe_moves_each_channel_to_its_new_resting_state(rig) -> None:
    rig.camera_service.set_strobe({1: True, 2: False, 3: True})
    # 1 -> strobe: channel off until a capture; 2 -> steady: held at its
    # brightness; 3 has no channel, so nothing is sent.
    assert rig.led_client.sent == ["SA0000#", "SB0050#"]


def test_set_strobe_notifies_observers_once(rig) -> None:
    seen: list[dict] = []
    rig.camera_service.subscribe_strobe(seen.append)
    rig.camera_service.set_strobe({1: True, 2: False})
    assert seen == [{1: True, 2: False}]


def test_set_strobe_rejects_an_unknown_camera_without_writing(rig) -> None:
    with pytest.raises(ConfigurationError):
        rig.camera_service.set_strobe({1: True, 9: True})
    assert rig.disk("camera") == CAMERA_DOC
    assert rig.cameras.get(1).settings.led_strobe is False
    assert rig.led_client.sent == []


def test_led_page_strobe_checkbox_reaches_the_camera_page(rig, qapp) -> None:
    from ui.camera.camera_page import CameraPage
    from ui.led.led_page import LedPage

    state = AppState()
    rig.camera_service.subscribe_strobe(
        lambda states: [state.camera_strobe_changed.emit(i, v) for i, v in states.items()]
    )
    led_page = LedPage(state, rig.led, rig.camera_service)
    camera_page = CameraPage(state, rig.camera_service)
    camera_page._list.setCurrentRow(0)  # camera 1
    assert led_page._camera_strobes[1].isChecked() is False
    assert led_page._camera_strobes[2].isChecked() is True

    led_page._camera_strobes[1].click()

    assert rig.cameras.get(1).settings.led_strobe is True
    assert camera_page._led_strobe.isChecked() is True
    assert led_page._camera_wiring[1].text() == "Ch 1 · strobe"


@pytest.mark.parametrize("strobe", [True, False])
def test_led_page_strobe_enable_and_disable_all(rig, qapp, monkeypatch, strobe) -> None:
    from ui.led.led_page import LedPage

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    page = LedPage(AppState(), rig.led, rig.camera_service)
    page._on_set_all_strobe(strobe)
    assert [c.settings.led_strobe for c in rig.cameras.cameras.values()] == [strobe] * 3
    assert [c["led_strobe"] for c in rig.disk("camera")["cameras"]] == [strobe] * 3
    assert [box.isChecked() for box in page._camera_strobes.values()] == [strobe] * 3

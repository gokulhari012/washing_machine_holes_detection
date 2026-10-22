"""Camera page "Save Image": writes the previewed frame where the operator
chooses, and is unavailable while Continuous Capture streams (offscreen Qt)."""

from __future__ import annotations

import copy
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import shiboken6
from PySide6.QtWidgets import QApplication, QFileDialog

from core.camera import CameraManager
from core.camera.image_file_camera import read_image, write_image
from core.led import LedControllerSettings, LedManager, SimulatedLedClient
from core.utilities.config_manager import ConfigManager
from models.app_state import AppState
from services.camera_service import CameraService
from services.led_service import LedService

CAMERA_DOC = {
    "cameras": [
        {
            "index": 1, "name": "Cam 1", "driver": "simulated", "connection_id": "",
            "enabled": True, "exposure_us": 10000, "gain_db": 0.0, "gamma": 1.0,
            "brightness": 0, "led_channel": 0, "led_strobe": False,
            "width": 640, "height": 480, "trigger_mode": "software",
            "roi": {"x": 0, "y": 0, "width": 0, "height": 0},
        }
    ]
}


class _FakeRepo:
    def upsert(self, values: dict) -> None:
        pass

    def delete_by_index(self, index: int) -> None:
        pass


class _FakeDatabase:
    camera_configs = _FakeRepo()


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def page(tmp_path, qapp):
    from ui.camera.camera_page import CameraPage

    config_dir = tmp_path / "config"
    (config_dir / "defaults").mkdir(parents=True)
    (config_dir / "camera.json").write_text(json.dumps(CAMERA_DOC))
    config = ConfigManager(config_dir)
    led = LedService(LedManager(SimulatedLedClient(), LedControllerSettings()), config)
    service = CameraService(
        CameraManager(copy.deepcopy(CAMERA_DOC["cameras"])), config, _FakeDatabase(), led
    )
    page = CameraPage(AppState(), service)
    page._save_image_dir = tmp_path / "captures"
    page._list.setCurrentRow(0)
    yield page
    page._continuous_btn.setChecked(False)
    # Delete now, while the QApplication is alive — left to pytest's final gc
    # pass, a parentless widget crashes the interpreter on exit.
    shiboken6.delete(page)


def _frame() -> np.ndarray:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[10:20, 5:30] = (10, 200, 90)
    return frame


def test_disabled_until_something_is_captured(page) -> None:
    assert not page._save_image_btn.isEnabled()
    page._show_full_frame(_frame())
    assert page._save_image_btn.isEnabled()


def test_disabled_while_continuous_capture_runs(page) -> None:
    page._show_full_frame(_frame())
    page._continuous_btn.setChecked(True)
    assert not page._save_image_btn.isEnabled()
    page._show_full_frame(_frame())  # a streamed frame must not re-enable it
    assert not page._save_image_btn.isEnabled()
    page._continuous_btn.setChecked(False)
    assert page._save_image_btn.isEnabled()


def test_saves_the_previewed_frame_to_the_chosen_file(page, tmp_path, monkeypatch) -> None:
    offered: list[str] = []
    target = tmp_path / "out dir" / "picked"  # no suffix -> .png

    def fake_dialog(parent, caption, start, filters):
        offered.append(start)
        target.parent.mkdir(exist_ok=True)
        return str(target), filters

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake_dialog))
    frame = _frame()
    page._show_full_frame(frame)
    page._on_save_image()

    assert offered[0].startswith(str(tmp_path / "captures" / "camera1_"))
    saved = read_image(target.with_suffix(".png"))
    assert np.array_equal(saved, frame)
    assert page._save_image_dir == target.parent  # next dialog opens there


def test_cancelled_dialog_writes_nothing(page, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", ""))
    )
    page._show_full_frame(_frame())
    page._on_save_image()
    assert list((tmp_path / "captures").iterdir()) == []


def test_write_image_handles_non_ascii_paths(tmp_path) -> None:
    path = tmp_path / "kamera_ü.png"
    write_image(path, _frame())
    assert np.array_equal(read_image(path), _frame())

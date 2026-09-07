"""Per-camera frame rate: one camera.json field paces every continuous view.

``fps`` (the Camera page's "Frame Rate") throttles how often the app asks a
camera for a frame — the live preview workers, the Camera page's Continuous
Capture and the Calibration page's Auto Calibrate scan all run at it, and none
of them keeps a fixed cadence of its own. Whether the background preview
threads run at all stays a separate station-wide switch,
``app_config.ui.live_preview_fps``.
"""

import json

import pytest

from core.camera import DEFAULT_VIEW_FPS, CameraSettings, frame_interval_ms
from workers.acquisition_worker import create_acquisition_workers

from tests.test_camera_service import CAMERA_DOC, make_service


# ------------------------------------------------------------- interval maths
@pytest.mark.parametrize(
    "fps, expected",
    [
        (10.0, 100),
        (4.0, 250),
        (30.0, 33),
        (0.5, 2000),  # sub-1 fps is legitimate on a 20 MP GigE camera
    ],
)
def test_frame_interval_ms_converts_rate(fps, expected) -> None:
    assert frame_interval_ms(fps) == expected


@pytest.mark.parametrize("fps", [0.0, -1.0])
def test_frame_interval_ms_guards_a_nonsense_rate(fps) -> None:
    """A legacy/hand-edited 0 must not divide by zero — it reads as the default."""
    assert frame_interval_ms(fps) == frame_interval_ms(DEFAULT_VIEW_FPS)


# ------------------------------------------------------------------- settings
def test_settings_read_fps_from_config() -> None:
    settings = CameraSettings.from_config(dict(CAMERA_DOC["cameras"][0], fps=7.5))
    assert settings.fps == 7.5


def test_settings_default_fps_for_a_config_predating_the_setting() -> None:
    entry = {k: v for k, v in CAMERA_DOC["cameras"][0].items() if k != "fps"}
    assert CameraSettings.from_config(entry).fps == DEFAULT_VIEW_FPS


# ---------------------------------------------------------------- live preview
class _FakeCamera:
    def __init__(self, settings: CameraSettings) -> None:
        self.settings = settings


class _ManagerStub:
    """Just the two members create_acquisition_workers touches."""

    def __init__(self, fps_by_index: dict[int, float]) -> None:
        self.cameras = {
            index: _FakeCamera(
                CameraSettings.from_config(
                    dict(CAMERA_DOC["cameras"][0], index=index, fps=fps)
                )
            )
            for index, fps in fps_by_index.items()
        }

    def get(self, index: int):
        return self.cameras[index]


def test_each_preview_worker_runs_at_its_own_cameras_rate() -> None:
    manager = _ManagerStub({1: 5.0, 2: 2.0})
    workers = create_acquisition_workers(manager, app_state=None, preview_enabled_fps=1.0)

    rates = {w.camera_index: round(1.0 / w._frame_interval_s, 3) for w in workers}
    assert rates == {1: 5.0, 2: 2.0}  # the station-wide value is a switch, not a rate


def test_no_preview_workers_when_the_station_switch_is_off() -> None:
    """ui.live_preview_fps = 0 is this station's setting — no preview threads."""
    manager = _ManagerStub({1: 5.0, 2: 2.0})
    assert create_acquisition_workers(manager, app_state=None, preview_enabled_fps=0.0) == []


# ---------------------------------------------------------------- persistence
def test_camera_fps_reads_back_what_was_saved(tmp_path) -> None:
    """What the Camera page saves is what the page-driven loops pace by."""
    service, _cameras, _plc = make_service(tmp_path)
    service.save_camera(dict(CAMERA_DOC["cameras"][0], fps=6.0))

    assert service.camera_fps(1) == 6.0
    saved = json.loads((tmp_path / "config" / "camera.json").read_text())
    assert saved["cameras"][0]["fps"] == 6.0


def test_camera_fps_defaults_for_an_entry_without_the_setting(tmp_path) -> None:
    service, _cameras, _plc = make_service(tmp_path)  # CAMERA_DOC carries no fps
    assert service.camera_fps(1) == DEFAULT_VIEW_FPS
    assert service.camera_fps(99) == DEFAULT_VIEW_FPS  # unknown camera

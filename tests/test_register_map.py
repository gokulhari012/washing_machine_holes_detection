"""RegisterMap parsing and position codec."""

import pytest

from core.plc import RegisterMap
from core.utilities.exceptions import ConfigurationError


def make_config() -> dict:
    return {
        "registers": {
            "trigger": 100,
            "machine_number": 101,
            "heartbeat": 102,
            "result": 118,
            "vision_complete": 119,
            "camera_positions": {
                "1": {"x": 110, "y": 111},
                "2": {"x": 112, "y": 113},
                "3": {"x": 114, "y": 115},
                "4": {"x": 116, "y": 117},
            },
        },
        "scaling": {"position_scale": 10, "position_offset": 10000},
    }


def test_from_config() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.trigger == 100
    assert rmap.camera_positions[3] == (114, 115)
    assert rmap.position_scale == 10


def test_codec_round_trip() -> None:
    rmap = RegisterMap.from_config(make_config())
    for mm in (0.0, 123.4, -5.0, 999.9):
        assert rmap.decode_position(rmap.encode_position(mm)) == pytest.approx(mm)


def test_codec_clamps_to_uint16() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.encode_position(99999.0) == 65535
    assert rmap.encode_position(-99999.0) == 0
    assert RegisterMap.NO_HOLE_RAW == 0


def test_invalid_config_raises() -> None:
    broken = make_config()
    del broken["registers"]["trigger"]
    with pytest.raises(ConfigurationError):
        RegisterMap.from_config(broken)


def test_model_select_defaults_to_none() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.model_select is None


def test_model_select_parsed_when_present() -> None:
    config = make_config()
    config["registers"]["model_select"] = 103
    rmap = RegisterMap.from_config(config)
    assert rmap.model_select == 103


def test_camera_jog_defaults_to_empty() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.camera_jog == {}
    assert rmap.camera_jog_home == {}
    assert rmap.jog_step == 10


def test_camera_jog_parsed_when_present() -> None:
    config = make_config()
    config["camera_jog"] = {
        "step": 25,
        "registers": {
            "1": {"x": 120, "y": 121, "home_x": 5, "home_y": 6},
            "2": {"x": 122, "y": 123},  # home_x/home_y default to 0
        },
    }
    rmap = RegisterMap.from_config(config)
    assert rmap.jog_step == 25
    assert rmap.camera_jog[1] == (120, 121)
    assert rmap.camera_jog[2] == (122, 123)
    assert rmap.camera_jog_home[1] == (5, 6)
    assert rmap.camera_jog_home[2] == (0, 0)


def test_camera_results_defaults_to_empty() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.camera_results == {}


def test_camera_results_parsed_when_present() -> None:
    config = make_config()
    config["registers"]["camera_results"] = {"1": 128, "2": 129}
    rmap = RegisterMap.from_config(config)
    assert rmap.camera_results == {1: 128, 2: 129}

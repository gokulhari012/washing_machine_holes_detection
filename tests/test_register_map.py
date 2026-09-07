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
        "scaling": {"position_scale": 10},
    }


def test_from_config() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.trigger == 100
    assert rmap.camera_positions[3] == (114, 115)
    assert rmap.position_scale == 10


def test_codec_round_trip() -> None:
    rmap = RegisterMap.from_config(make_config())
    for mm in (0.0, 123.4, -5.0, 999.9):
        # Home has to leave room for the offset in both directions; with
        # home 0 a negative offset clamps at 0, which is the point of having
        # the servo datum in the first place.
        for home in (30000, 40000):
            assert rmap.decode_position(
                rmap.encode_position(mm, home), home
            ) == pytest.approx(mm)


def test_negative_offset_clamps_without_a_servo_home() -> None:
    """No home means no room below zero — the register cannot go negative."""
    rmap = RegisterMap.from_config(make_config())
    assert rmap.encode_position(-5.0) == 0


def test_encode_is_servo_home_plus_scaled_offset() -> None:
    """The worked example from the spec: home 6000, +2.0 mm, scale 100 -> 6200."""
    config = make_config()
    config["scaling"]["position_scale"] = 100
    rmap = RegisterMap.from_config(config)
    assert rmap.encode_position(2.0, 6000) == 6200
    # A negative offset lands below home — that is how the unsigned register
    # carries a negative measurement.
    assert rmap.encode_position(-2.0, 6000) == 5800
    assert rmap.encode_position(0.0, 6000) == 6000


def test_encode_without_servo_home_is_plain_scaled_mm() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.encode_position(12.3) == 123
    assert RegisterMap.NO_HOLE_RAW == 0


def test_codec_clamps_to_uint16() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.encode_position(99999.0, 6000) == 65535
    assert rmap.encode_position(-99999.0, 6000) == 0


def test_servo_home_positions_default_to_empty() -> None:
    assert RegisterMap.from_config(make_config()).servo_home_positions == {}


def test_servo_home_positions_parsed_when_present() -> None:
    config = make_config()
    config["registers"]["servo_home_positions"] = {
        "1": {"x": 144, "y": 145},
        "2": {"x": 146, "y": 147},
    }
    rmap = RegisterMap.from_config(config)
    assert rmap.servo_home_positions == {1: (144, 145), 2: (146, 147)}


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
    assert rmap.camera_jog_z == {}
    assert rmap.camera_jog_busy == {}
    assert rmap.jog_step == 10


def test_camera_jog_parsed_when_present() -> None:
    config = make_config()
    config["camera_jog"] = {
        "step": 25,
        "registers": {
            "1": {"x": 120, "y": 121, "z": 122, "busy": 200},
            "2": {"x": 123, "y": 124},  # no Z axis or busy coil wired up for this camera
        },
    }
    rmap = RegisterMap.from_config(config)
    assert rmap.jog_step == 25
    assert rmap.camera_jog[1] == (120, 121)
    assert rmap.camera_jog[2] == (123, 124)
    assert rmap.camera_jog_z[1] == 122
    assert 2 not in rmap.camera_jog_z
    assert rmap.camera_jog_busy[1] == 200
    assert 2 not in rmap.camera_jog_busy


def test_camera_results_defaults_to_empty() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.camera_results == {}


def test_camera_results_parsed_when_present() -> None:
    config = make_config()
    config["registers"]["camera_results"] = {"1": 128, "2": 129}
    rmap = RegisterMap.from_config(config)
    assert rmap.camera_results == {1: 128, 2: 129}


def test_camera_brightness_defaults_to_empty() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.camera_brightness == {}


def test_camera_brightness_parsed_when_present() -> None:
    config = make_config()
    config["registers"]["camera_brightness"] = {"1": 156, "2": 157}
    rmap = RegisterMap.from_config(config)
    assert rmap.camera_brightness == {1: 156, 2: 157}

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
            # 32-bit: each axis is a base address (low word) + base+1 (high
            # word), so X and Y are configured 2 apart, not 1 — see
            # RegisterMap.split_dword/join_dword.
            "camera_positions": {
                "1": {"x": 200, "y": 202},
                "2": {"x": 204, "y": 206},
                "3": {"x": 208, "y": 210},
                "4": {"x": 212, "y": 214},
            },
        },
        "scaling": {"position_scale": 10},
    }


def test_from_config() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.trigger == 100
    assert rmap.camera_positions[3] == (208, 210)
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


def test_codec_clamps_to_uint32() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.encode_position(999_999_999.0, 6000) == 4294967295
    assert rmap.encode_position(-999_999_999.0, 6000) == 0


def test_a_uint16_sized_value_no_longer_clamps() -> None:
    """The whole point of the conversion: values that used to clamp at 65535
    now fit, since a position register is 32-bit."""
    rmap = RegisterMap.from_config(make_config())
    assert rmap.encode_position(99999.0, 6000) == 999990 + 6000


def test_split_and_join_dword_round_trip() -> None:
    for raw in (0, 1, 65535, 65536, 66536, 4294967295):
        low, high = RegisterMap.split_dword(raw)
        assert 0 <= low <= 0xFFFF
        assert 0 <= high <= 0xFFFF
        assert RegisterMap.join_dword(low, high) == raw


def test_split_dword_is_low_word_first() -> None:
    """66536 = 65536 (1 << 16) + 1000 -> high word 1, low word 1000 — the
    base address gets the low word, base+1 the high word."""
    low, high = RegisterMap.split_dword(66536)
    assert (low, high) == (1000, 1)


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


def test_camera_results_defaults_to_empty() -> None:
    rmap = RegisterMap.from_config(make_config())
    assert rmap.camera_results == {}


def test_camera_results_parsed_when_present() -> None:
    config = make_config()
    config["registers"]["camera_results"] = {"1": 128, "2": 129}
    rmap = RegisterMap.from_config(config)
    assert rmap.camera_results == {1: 128, 2: 129}



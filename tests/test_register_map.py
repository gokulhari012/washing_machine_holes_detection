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

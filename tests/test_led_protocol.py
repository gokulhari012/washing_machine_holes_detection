"""core.led.protocol: command builders and the max_brightness safety ceiling.

Mirrors the self-check the module ships in its ``__main__`` block, plus the
validation edges that block doesn't cover.
"""

import pytest

from core.led import protocol


def test_build_channel_command_examples() -> None:
    assert protocol.build_channel_command(1, 200) == "SA0200#"
    assert protocol.build_channel_command(2, 100) == "SB0100#"
    assert protocol.build_channel_command(3, 200) == "SC0200#"
    assert protocol.build_channel_command(4, 255) == "SD0255#"


def test_build_channel_command_rejects_out_of_range_channel() -> None:
    with pytest.raises(ValueError):
        protocol.build_channel_command(5, 100)
    with pytest.raises(ValueError):
        protocol.build_channel_command(0, 100)


def test_format_brightness_is_always_three_digits() -> None:
    assert protocol.format_brightness(0) == "000"
    assert protocol.format_brightness(5) == "005"
    assert protocol.format_brightness(25) == "025"
    assert protocol.format_brightness(255) == "255"


def test_format_brightness_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        protocol.format_brightness(256)
    with pytest.raises(ValueError):
        protocol.format_brightness(-1)


def test_build_multichannel_command_documented_example() -> None:
    command = protocol.build_multichannel_command(
        [(100, True), (128, True), (25, False), (0, True)]
    )
    assert command == "S100T128T025F000TC#"


def test_build_multichannel_command_requires_exactly_four_states() -> None:
    with pytest.raises(ValueError):
        protocol.build_multichannel_command([(100, True)] * 3)


def test_preset_brightness_documented_percentages() -> None:
    assert protocol.preset_brightness(0) == 0
    assert protocol.preset_brightness(25) == 64
    assert protocol.preset_brightness(50) == 128
    assert protocol.preset_brightness(75) == 191
    assert protocol.preset_brightness(100) == 255


def test_preset_brightness_rejects_unsupported_percentage() -> None:
    with pytest.raises(ValueError):
        protocol.preset_brightness(10)


def test_is_ack() -> None:
    assert protocol.is_ack("!") is True
    assert protocol.is_ack("garbage!") is True
    assert protocol.is_ack("") is False
    assert protocol.is_ack(None) is False


def test_clamp_brightness() -> None:
    assert protocol.clamp_brightness(300, max_brightness=200) == 200
    assert protocol.clamp_brightness(-5, max_brightness=200) == 0
    assert protocol.clamp_brightness(150, max_brightness=200) == 150


def test_settings_build_command_enforces_max_brightness() -> None:
    settings = protocol.LedControllerSettings(port="COM5", max_brightness=200)
    assert settings.build_command(1, 255) == "SA0200#"  # clamped down to the ceiling
    assert settings.build_command(1, 100) == "SA0100#"  # under the ceiling, unchanged


def test_settings_rejects_out_of_range_max_brightness() -> None:
    with pytest.raises(ValueError):
        protocol.LedControllerSettings(max_brightness=256)


def test_settings_round_trips_through_dict() -> None:
    settings = protocol.LedControllerSettings(port="COM5", max_brightness=200)
    assert protocol.LedControllerSettings.from_dict(settings.to_dict()) == settings


def test_settings_from_dict_falls_back_to_defaults_for_missing_keys() -> None:
    settings = protocol.LedControllerSettings.from_dict({})
    assert settings == protocol.LedControllerSettings()

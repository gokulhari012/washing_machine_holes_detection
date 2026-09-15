"""
Protocol module for the KCS/Shunshangxin KDC-24V60W-4T 4-channel LED Light
Source Controller.

This file has zero third-party dependencies (stdlib only: dataclasses,
typing) so it can be dropped as-is into another application - it does not
assume pyserial, Qt, or any particular UI/connection framework. It was
ported unchanged from the standalone ``led_controller_tester`` tool into
this application's ``core.led`` package for exactly that reason.

Only commands explicitly documented in the controller's user manual are
implemented here:

1. Individual channel brightness command:
       S <channel-letter> 0 <XXX> #
   where channel-letter is A/B/C/D (channels 1-4) and XXX is a
   three-digit brightness value from 000 to 255.

       build_channel_command(1, 200) -> "SA0200#"
       build_channel_command(2, 100) -> "SB0100#"
       build_channel_command(3, 200) -> "SC0200#"
       build_channel_command(4, 255) -> "SD0255#"

2. Multi-channel T/F command, documented example:
       S100T128T025F000TC#
   meaning, in channel order A,B,C,D:
       channel A: ON,  brightness 100
       channel B: ON,  brightness 128
       channel C: OFF, brightness 025
       channel D: ON,  brightness 000

Do not add any undocumented command to this module.

--------------------------------------------------------------------------
Integration in this application:

``core.led.led_client_base.LedClientBase`` is the abstract adapter
(mirroring ``core.plc.plc_client_base.PlcClientBase``); concrete adapters
(``SerialLedClient`` over RS232, ``SimulatedLedClient`` for offline
development and tests) implement it. ``core.led.led_manager.LedManager``
wraps a client with connection-state tracking and enforces
``LedControllerSettings.max_brightness`` at the single choke point every
command goes through — ``services.led_service.LedService`` is the UI-facing
facade, and ``ui.led.led_page.LedPage`` is the Communication-tab-style page
(connection settings, four channel panels, raw command tester, log).

On every brightness change the page calls:

    led_service.set_channel_brightness(channel, brightness)

which reaches ``LedControllerSettings.build_command()`` internally, so the
configured ceiling is enforced no matter what range the UI slider/spinbox
itself allows.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence, Tuple

MIN_CHANNEL = 1
MAX_CHANNEL = 4
MIN_BRIGHTNESS = 0
MAX_BRIGHTNESS = 255

# Channel 1 = A, Channel 2 = B, Channel 3 = C, Channel 4 = D
CHANNEL_LETTERS = {1: "A", 2: "B", 3: "C", 4: "D"}

# Documented percentage presets (section 14 of the spec).
PRESET_PERCENTAGES = {0: 0, 25: 64, 50: 128, 75: 191, 100: 255}

# Ack byte returned by the controller for a successfully received command.
ACK = "!"

# Documented RS232 defaults.
DEFAULT_BAUD_RATE = 19200
DEFAULT_TIMEOUT_MS = 1000
DEFAULT_MAX_BRIGHTNESS = MAX_BRIGHTNESS


def format_brightness(value: int) -> str:
    """Format a brightness value as an exact three-digit decimal string.

    0   -> "000"
    5   -> "005"
    25  -> "025"
    100 -> "100"
    255 -> "255"
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"brightness must be an int, got {type(value).__name__}")
    if not (MIN_BRIGHTNESS <= value <= MAX_BRIGHTNESS):
        raise ValueError(
            f"brightness must be between {MIN_BRIGHTNESS} and {MAX_BRIGHTNESS}, got {value}"
        )
    return f"{value:03d}"


def build_channel_command(channel: int, brightness: int) -> str:
    """Build the documented single-channel brightness command.

    build_channel_command(1, 200) -> "SA0200#"
    build_channel_command(2, 100) -> "SB0100#"
    build_channel_command(3, 200) -> "SC0200#"
    build_channel_command(4, 255) -> "SD0255#"
    """
    if channel not in CHANNEL_LETTERS:
        raise ValueError(f"channel must be {MIN_CHANNEL}-{MAX_CHANNEL}, got {channel}")
    letter = CHANNEL_LETTERS[channel]
    return f"S{letter}0{format_brightness(brightness)}#"


def build_multichannel_command(states: Sequence[Tuple[int, bool]]) -> str:
    """Build the documented multi-channel T/F command.

    ``states`` must contain exactly 4 (brightness, on) pairs, in channel
    order A, B, C, D (channels 1-4).

    Documented example:
        build_multichannel_command([(100, True), (128, True), (25, False), (0, True)])
        -> "S100T128T025F000TC#"
    """
    states = list(states)
    if len(states) != MAX_CHANNEL:
        raise ValueError(
            f"states must contain exactly {MAX_CHANNEL} (brightness, on) entries"
        )
    parts = []
    for brightness, on in states:
        parts.append(f"{format_brightness(brightness)}{'T' if on else 'F'}")
    return "S" + "".join(parts) + "C#"


def preset_brightness(percentage: int) -> int:
    """Map a documented percentage preset (0/25/50/75/100) to a brightness value."""
    if percentage not in PRESET_PERCENTAGES:
        raise ValueError(f"unsupported preset percentage: {percentage}")
    return PRESET_PERCENTAGES[percentage]


def is_ack(response: str) -> bool:
    """Return True if the raw response text represents the documented ack ("!")."""
    return response is not None and ACK in response


def clamp_brightness(value: int, max_brightness: int = MAX_BRIGHTNESS) -> int:
    """Clamp a requested brightness into [0, max_brightness].

    ``max_brightness`` is an operator-configured safety ceiling (see
    LedControllerSettings), separate from the protocol's hard 0-255 range.
    Values below 0 or above ``max_brightness`` are clamped rather than
    rejected, since a UI brightness slider/spinbox should never be able to
    raise a hardware error just by being dragged too far.
    """
    if not (MIN_BRIGHTNESS <= max_brightness <= MAX_BRIGHTNESS):
        raise ValueError(
            f"max_brightness must be between {MIN_BRIGHTNESS} and {MAX_BRIGHTNESS}, got {max_brightness}"
        )
    if value < MIN_BRIGHTNESS:
        return MIN_BRIGHTNESS
    if value > max_brightness:
        return max_brightness
    return value


@dataclass
class LedControllerSettings:
    """All settings needed to talk to the KDC-24V60W-4T controller.

    Backs the LED Controller page's connection section. Data bits/stop
    bits/parity/flow control are not included: the manual fixes them at
    8/1/None/None, so they are not meaningful settings to expose - only
    port, baud rate, response timeout, and the max brightness safety
    ceiling are.
    """

    port: str = ""
    baud_rate: int = DEFAULT_BAUD_RATE
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    max_brightness: int = DEFAULT_MAX_BRIGHTNESS

    def __post_init__(self) -> None:
        if not (MIN_BRIGHTNESS <= self.max_brightness <= MAX_BRIGHTNESS):
            raise ValueError(
                f"max_brightness must be between {MIN_BRIGHTNESS} and {MAX_BRIGHTNESS}, "
                f"got {self.max_brightness}"
            )
        if self.timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {self.timeout_ms}")
        if self.baud_rate <= 0:
            raise ValueError(f"baud_rate must be positive, got {self.baud_rate}")

    def clamp(self, brightness: int) -> int:
        """Clamp a requested brightness to this settings object's max_brightness."""
        return clamp_brightness(brightness, self.max_brightness)

    def build_command(self, channel: int, brightness: int) -> str:
        """Build the channel command with max_brightness enforced.

        This is the single call the LED Controller page's brightness
        controls go through - it guarantees the configured safety ceiling
        can never be exceeded, regardless of the range the slider or
        spinbox itself allows.
        """
        return build_channel_command(channel, self.clamp(brightness))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for project/config storage (e.g. the shared .pvp format)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LedControllerSettings":
        """Rebuild from a dict produced by to_dict(); missing keys fall back
        to the documented defaults."""
        return cls(
            port=data.get("port", ""),
            baud_rate=data.get("baud_rate", DEFAULT_BAUD_RATE),
            timeout_ms=data.get("timeout_ms", DEFAULT_TIMEOUT_MS),
            max_brightness=data.get("max_brightness", DEFAULT_MAX_BRIGHTNESS),
        )


if __name__ == "__main__":
    # Sanity check for the examples required by the spec.
    assert build_channel_command(1, 200) == "SA0200#"
    assert build_channel_command(2, 100) == "SB0100#"
    assert build_channel_command(3, 200) == "SC0200#"
    assert build_channel_command(4, 255) == "SD0255#"
    assert build_multichannel_command([(100, True), (128, True), (25, False), (0, True)]) == "S100T128T025F000TC#"

    # max_brightness clamping
    assert clamp_brightness(300, max_brightness=200) == 200
    assert clamp_brightness(-5, max_brightness=200) == 0
    assert clamp_brightness(150, max_brightness=200) == 150

    settings = LedControllerSettings(port="COM5", max_brightness=200)
    assert settings.build_command(1, 255) == "SA0200#"  # clamped down to the ceiling
    assert settings.build_command(1, 100) == "SA0100#"  # under the ceiling, unchanged
    assert LedControllerSettings.from_dict(settings.to_dict()) == settings

    print("protocol.py self-check OK")

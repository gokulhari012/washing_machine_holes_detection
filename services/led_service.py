"""LED Controller facade: configuration persistence, connect/disconnect,
per-channel brightness, and the raw command tester.

Owns no thread and no polling loop - the RS232 link to the KDC-24V60W-4T
auto-connects at application startup and again after every configuration
save (see ``Application.start`` / ``Application._on_led_config_saved`` in
``main.py``); every command is a one-shot request/response the caller
triggers directly, blocking briefly the same way ``PlcService``'s manual
register write and ``PlcPage``'s Test Connection already do (see
``core.led.led_manager``).

Two callers: ``CameraService._push_brightness`` (every camera apply/save
pushes that camera's brightness to its configured LED channel) and
``ui.led.led_page.LedPage`` (connection settings, manual light control, and
the raw command tester for troubleshooting the hardware directly).

**Manual light control is separate from the cameras' brightness.** The
LED page's per-channel ON/OFF and ALL ON/ALL OFF buttons light a channel at
its own *manual* level (``led.json``'s ``manual_brightness``, one per
channel), which nothing else reads: the inspection cycle keeps lighting each
camera at that camera's ``brightness`` (camera.json). A manual command is a
one-off write, so the next cycle's strobe - or a camera's steady-brightness
push - simply overwrites whatever the operator left a channel at.
"""

from __future__ import annotations

from core.led import LedManager, build_led_settings, create_led_client
from core.led.protocol import MAX_BRIGHTNESS, MAX_CHANNEL, MIN_BRIGHTNESS, MIN_CHANNEL
from core.logging import get_logger
from core.utilities import ConfigManager
from core.utilities.enums import ConnectionState, LogSource
from core.utilities.exceptions import ConfigurationError

logger = get_logger(LogSource.LED)

CHANNELS = tuple(range(MIN_CHANNEL, MAX_CHANNEL + 1))
#: Manual level for a channel led.json has no ``manual_brightness`` entry for.
DEFAULT_MANUAL_BRIGHTNESS = 128


class LedService:
    """Everything the LED Controller page needs; owns no thread."""

    def __init__(self, led_manager: LedManager, config_manager: ConfigManager) -> None:
        self._manager = led_manager
        self._config = config_manager

    # ---------------------------------------------------------------- state
    @property
    def state(self) -> ConnectionState:
        return self._manager.state

    @property
    def last_error(self) -> str:
        return self._manager.last_error

    @property
    def max_brightness(self) -> int:
        return self._manager.settings.max_brightness

    def connect(self) -> None:
        """Raises LedError on failure (state becomes ERROR)."""
        self._manager.connect()

    def disconnect(self) -> None:
        self._manager.disconnect()

    @staticmethod
    def list_ports() -> list[str]:
        """Available COM port device names for the connection picker."""
        from core.led.serial_led_client import list_serial_ports

        return list_serial_ports()

    # ---------------------------------------------------------------- config
    def get_config(self) -> dict:
        return self._config.load("led")

    def save_config(self, led_config: dict) -> None:
        """Validate, then persist to led.json.

        The runtime client/settings rebuild is wired in the composition root
        via ``ConfigManager.subscribe("led", ...)``, mirroring the PLC page's
        immediate-effect save.

        Raises:
            ConfigurationError: an out-of-range setting.
        """
        build_led_settings(led_config)  # validate before persisting
        self._config.save("led", led_config)

    def manual_levels(self) -> dict[int, int]:
        """The manual-control level of every channel, ``{1: 128, ...}``.

        Read from ``led.json``'s ``manual_brightness``; a missing or
        malformed entry falls back to :data:`DEFAULT_MANUAL_BRIGHTNESS`
        rather than failing the page.
        """
        stored = self.get_config().get("manual_brightness", {})
        levels: dict[int, int] = {}
        for channel in CHANNELS:
            try:
                value = int(stored.get(str(channel), DEFAULT_MANUAL_BRIGHTNESS))
            except (TypeError, ValueError, AttributeError):
                value = DEFAULT_MANUAL_BRIGHTNESS
            levels[channel] = min(max(value, MIN_BRIGHTNESS), MAX_BRIGHTNESS)
        return levels

    def save_manual_levels(self, levels: dict[int, int]) -> None:
        """Remember the manual-control levels in ``led.json``.

        Saved without notifying the ``led`` subscribers: the composition
        root's one rebuilds the client and reconnects the serial port, which
        a remembered slider position has no reason to do.

        Raises:
            ConfigurationError: a level outside 0-255, or the file could not
                be written.
        """
        merged = self.manual_levels()
        for channel, value in levels.items():
            channel, value = int(channel), int(value)
            if channel not in CHANNELS:
                raise ConfigurationError(f"LED channel must be 1-4, got {channel}")
            if not MIN_BRIGHTNESS <= value <= MAX_BRIGHTNESS:
                raise ConfigurationError(
                    f"LED channel {channel}: brightness must be 0-255, got {value}"
                )
            merged[channel] = value
        cfg = self.get_config()
        cfg["manual_brightness"] = {str(channel): merged[channel] for channel in CHANNELS}
        self._config.save("led", cfg, notify=False)

    # -------------------------------------------------------------- commands
    def set_channel_brightness(self, channel: int, brightness: int) -> str:
        """Raises LedError; caller must already be connected."""
        return self._manager.send_channel(channel, brightness)

    def channel_command(self, channel: int, brightness: int) -> str:
        """The exact command :meth:`set_channel_brightness` would send, with
        ``max_brightness`` applied - for the page's communication log."""
        return self._manager.settings.build_command(channel, brightness)

    def manual_on(self, channel: int, brightness: int) -> None:
        """Manual control: light *channel* at *brightness* (clamped to
        ``max_brightness``). Does not touch any camera's brightness.

        Raises:
            LedError: the link is down or broke.
        """
        self._manager.send_channel(channel, brightness)

    def manual_off(self, channel: int) -> None:
        """Manual control: switch *channel* off (brightness 0).

        Raises:
            LedError: the link is down or broke.
        """
        self._manager.send_channel(channel, 0)

    def manual_all_on(self, levels: dict[int, int]) -> None:
        """Manual control: light every channel, each at its own level.

        One single-channel command per channel, in order - the same command
        the per-channel buttons send, so every value goes through the
        ``max_brightness`` ceiling. Stops at the first failure, like
        ``LedManager.send_all_channels``.

        Raises:
            LedError: the link is down or broke.
        """
        for channel in CHANNELS:
            self._manager.send_channel(channel, int(levels.get(channel, 0)))

    def manual_all_off(self) -> None:
        """Manual control: switch every channel off.

        Raises:
            LedError: the link is down or broke.
        """
        self._manager.send_all_channels(0)

    def send_raw(self, command: str, *, append_terminator: bool = False) -> str:
        """Manual/raw command tester - sent exactly as typed, not clamped."""
        return self._manager.send_raw(command, append_terminator=append_terminator)

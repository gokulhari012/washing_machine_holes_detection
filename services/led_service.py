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
``ui.led.led_page.LedPage`` (connection settings + the raw command tester
for troubleshooting the hardware directly).
"""

from __future__ import annotations

from core.led import LedManager, build_led_settings, create_led_client
from core.logging import get_logger
from core.utilities import ConfigManager
from core.utilities.enums import ConnectionState, LogSource

logger = get_logger(LogSource.LED)


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

    # -------------------------------------------------------------- commands
    def set_channel_brightness(self, channel: int, brightness: int) -> str:
        """Raises LedError; caller must already be connected."""
        return self._manager.send_channel(channel, brightness)

    def send_raw(self, command: str, *, append_terminator: bool = False) -> str:
        """Manual/raw command tester - sent exactly as typed, not clamped."""
        return self._manager.send_raw(command, append_terminator=append_terminator)

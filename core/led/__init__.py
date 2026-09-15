"""LED Controller (KCS/Shunshangxin KDC-24V60W-4T) communication: protocol,
abstract client, adapters, connection manager."""

from __future__ import annotations

from core.led.led_client_base import LedClientBase
from core.led.led_manager import LedManager
from core.led.protocol import (
    DEFAULT_BAUD_RATE,
    DEFAULT_MAX_BRIGHTNESS,
    DEFAULT_TIMEOUT_MS,
    LedControllerSettings,
)
from core.led.simulated_led_client import SimulatedLedClient

__all__ = [
    "LedClientBase",
    "LedManager",
    "LedControllerSettings",
    "SimulatedLedClient",
    "build_led_settings",
    "create_led_client",
]


def build_led_settings(led_config: dict) -> LedControllerSettings:
    """Parse ``led.json``'s document into the dataclass that enforces the
    ``max_brightness`` ceiling for every channel command (see
    ``LedControllerSettings.build_command``).

    Raises:
        ConfigurationError: an out-of-range value (see
            ``LedControllerSettings.__post_init__``).
    """
    from core.utilities.exceptions import ConfigurationError

    connection = led_config.get("connection", {})
    try:
        return LedControllerSettings(
            port=str(connection.get("port", "")),
            baud_rate=int(connection.get("baud_rate", DEFAULT_BAUD_RATE)),
            timeout_ms=int(connection.get("timeout_ms", DEFAULT_TIMEOUT_MS)),
            max_brightness=int(led_config.get("max_brightness", DEFAULT_MAX_BRIGHTNESS)),
        )
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc


def create_led_client(led_config: dict) -> LedClientBase:
    """Factory: build the client named by ``connection.driver``.

    ``serial`` imports pyserial lazily so the simulator (and tests) run
    without a physical adapter attached; ``simulated`` needs no hardware or
    third-party package at all.
    """
    connection = led_config.get("connection", {})
    driver = str(connection.get("driver", "simulated")).lower()

    if driver == "simulated":
        return SimulatedLedClient()
    if driver == "serial":
        from core.led.serial_led_client import SerialLedClient

        return SerialLedClient(
            port=str(connection.get("port", "")),
            baud_rate=int(connection.get("baud_rate", DEFAULT_BAUD_RATE)),
            timeout_s=int(connection.get("timeout_ms", DEFAULT_TIMEOUT_MS)) / 1000.0,
        )

    from core.utilities.exceptions import ConfigurationError

    raise ConfigurationError(f"Unknown LED controller driver: {driver!r}")

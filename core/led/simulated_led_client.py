"""In-memory LED controller simulator.

Purposes:
- run the full application, and the LED Controller page, with no physical
  KDC-24V60W-4T attached - the factory function selects it when
  ``connection.driver == "simulated"``, mirroring ``core.plc.SimulatedPlc``;
- deterministic unit tests for ``LedManager`` and ``LedService``.

Every command received while connected is acknowledged with the documented
"!" response - the manual defines no rejection/NACK response, so nothing
here fabricates one. Sent commands are recorded on :attr:`sent` for tests to
assert against.
"""

from __future__ import annotations

from core.led.led_client_base import LedClientBase
from core.led.protocol import ACK
from core.logging import get_logger
from core.utilities.enums import LogSource
from core.utilities.exceptions import LedConnectionError

logger = get_logger(LogSource.LED)


class SimulatedLedClient(LedClientBase):
    """Thread-unsafe fake LED controller - fine here, since every call site
    in this application already serialises access through ``LedManager``."""

    def __init__(self) -> None:
        self._connected = False
        self.sent: list[str] = []

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True
        logger.info("Connected to simulated LED controller")

    def disconnect(self) -> None:
        self._connected = False

    def send(self, command: str, *, append_terminator: bool = False) -> str:
        if not self._connected:
            raise LedConnectionError("Not connected to LED controller")
        self.sent.append(command)
        return ACK

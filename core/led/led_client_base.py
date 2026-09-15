"""Abstract LED controller client interface (adapter pattern).

Concrete adapters (RS232 today, a simulator for offline development and
tests) implement this interface; everything above it - ``LedManager``,
``LedService``, the UI - depends only on this contract. Mirrors
``core.plc.plc_client_base.PlcClientBase``.

Error contract (all from ``core.utilities.exceptions``):
- ``LedConnectionError`` - port could not be opened / connection lost
- ``LedTimeoutError``    - controller did not answer within the configured timeout
- ``LedWriteError``      - the command could not be written
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LedClientBase(ABC):
    """Contract every LED controller transport adapter must fulfil."""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """True while the transport link is usable."""

    @abstractmethod
    def connect(self) -> None:
        """Open the connection.

        Raises:
            LedConnectionError: the controller/port is unreachable.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection. Must be safe to call repeatedly."""

    @abstractmethod
    def send(self, command: str, *, append_terminator: bool = False) -> str:
        """Write *command* exactly as given and return whatever the
        controller sends back within the configured timeout.

        The manual only documents ``"!"`` as the success acknowledgement, so
        the raw bytes received are returned as-is (never fabricated) -
        callers use :func:`core.led.protocol.is_ack` to check for it.
        ``append_terminator`` adds a trailing CR/LF; the manual's own
        end-of-line/termination character is ``#``, embedded in *command*
        itself, so this defaults to off.

        Raises:
            LedConnectionError: not connected.
            LedWriteError: the command could not be written.
            LedTimeoutError: no bytes were received before the timeout elapsed.
        """

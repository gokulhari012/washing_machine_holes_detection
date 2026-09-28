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
    def send(
        self,
        command: str,
        *,
        append_terminator: bool = False,
        expect_response: bool = True,
    ) -> str:
        """Write *command* exactly as given and return whatever the
        controller sends back within the configured timeout.

        The manual only documents ``"!"`` as the success acknowledgement, so
        the raw bytes received are returned as-is (never fabricated) -
        callers use :func:`core.led.protocol.is_ack` to check for it.
        ``append_terminator`` adds a trailing CR/LF; the manual's own
        end-of-line/termination character is ``#``, embedded in *command*
        itself, so this defaults to off.

        ``expect_response=False`` makes this **fire-and-forget**: the command
        is still written synchronously - so a strobe really is lit by the
        time this returns and the capture that follows sees it - but the
        adapter does not then sit waiting for the acknowledgement, and
        returns ``""`` instead. That is what every one-way command in this
        application uses (brightness pushes, strobe on/off), because none of
        them reads the reply and a silent controller must never stall a
        capture, an inspection cycle, or the GUI thread. Only the LED
        Controller page's raw command tester - where the operator is asking
        the hardware a question and wants the answer - waits for a response.

        Raises:
            LedConnectionError: not connected.
            LedWriteError: the command could not be written.
            LedTimeoutError: nothing was received before the timeout elapsed.
                Never raised when ``expect_response`` is False.
        """

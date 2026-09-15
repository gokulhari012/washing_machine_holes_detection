"""Connection lifecycle and command dispatch for the LED controller.

``LedManager`` wraps a :class:`LedClientBase` with:

- a small state machine (DISCONNECTED / CONNECTING / CONNECTED / ERROR) with
  Qt-free observer callbacks (the UI bridges them to signals), the same
  pattern as ``core.plc.PlcManager``;
- the ``max_brightness`` safety ceiling (``LedControllerSettings``) enforced
  at a single choke point every channel command goes through, regardless of
  what range a UI slider/spinbox allows;
- :meth:`rebuild` to swap in a new client/settings in place after a
  configuration save, so port/baud/timeout/ceiling changes take effect live.

Unlike the PLC link, this connection is opened and closed on operator
request (the LED Controller page's Connect/Disconnect buttons) rather than
auto-reconnected by a poll loop: there is no continuous polling need, since
every command is a one-shot brightness write the UI triggers directly. A
communication failure still drops the link into ERROR state and disconnects
the underlying client, exactly like ``PlcManager``, so a half-open serial
port is never left behind - the operator reconnects explicitly afterwards.
"""

from __future__ import annotations

import threading
from typing import Callable, Sequence, Tuple

from core.led.led_client_base import LedClientBase
from core.led.protocol import LedControllerSettings, build_multichannel_command
from core.logging import get_logger
from core.utilities.enums import ConnectionState, LogSource
from core.utilities.exceptions import LedError

logger = get_logger(LogSource.LED)

StateCallback = Callable[[ConnectionState], None]


class LedManager:
    """Owns the LED controller connection state and command dispatch."""

    def __init__(self, client: LedClientBase, settings: LedControllerSettings) -> None:
        self._client = client
        self._settings = settings
        self._state = ConnectionState.DISCONNECTED
        self._lock = threading.Lock()
        self._callbacks: list[StateCallback] = []
        self.last_error: str = ""

    # ----------------------------------------------------------------- state
    @property
    def state(self) -> ConnectionState:
        with self._lock:
            return self._state

    @property
    def settings(self) -> LedControllerSettings:
        return self._settings

    def subscribe_state(self, callback: StateCallback) -> None:
        """Register a callback fired on every state *change* (any thread)."""
        with self._lock:
            self._callbacks.append(callback)

    def _set_state(self, new_state: ConnectionState) -> None:
        with self._lock:
            if new_state is self._state:
                return
            self._state = new_state
            callbacks = list(self._callbacks)
        logger.info("LED controller state -> %s", new_state.value)
        for callback in callbacks:
            try:
                callback(new_state)
            except Exception:  # observers must never break the caller
                logger.exception("LED controller state callback raised")

    # ------------------------------------------------------------ connection
    def connect(self) -> None:
        """Blocking connect attempt.

        Raises:
            LedError: on failure (state becomes ERROR).
        """
        self._set_state(ConnectionState.CONNECTING)
        try:
            self._client.connect()
        except LedError as exc:
            self.last_error = str(exc)
            self._set_state(ConnectionState.ERROR)
            raise
        self._set_state(ConnectionState.CONNECTED)

    def disconnect(self) -> None:
        try:
            self._client.disconnect()
        finally:
            self._set_state(ConnectionState.DISCONNECTED)

    def rebuild(self, client: LedClientBase, settings: LedControllerSettings) -> None:
        """Swap in a new client and settings without replacing this
        ``LedManager`` instance, so every holder of it (``LedService``) keeps
        working unchanged - the same trick ``PlcManager.rebuild`` uses for a
        live PLC configuration save.

        Disconnects the outgoing client (best-effort) and resets connection
        state to DISCONNECTED; the operator reconnects explicitly (there is
        no poll loop to do it automatically for this link).
        """
        try:
            self._client.disconnect()
        except Exception:  # a dying old client must never block the rebuild
            logger.exception("Error disconnecting outgoing LED client during rebuild")
        self._client = client
        self._settings = settings
        self._set_state(ConnectionState.DISCONNECTED)

    # ------------------------------------------------------------- commands
    def send_channel(self, channel: int, brightness: int) -> str:
        """Build and send the documented single-channel command with
        ``max_brightness`` enforced.

        Raises:
            LedError: communication failure (state becomes ERROR).
        """
        return self._send(self._settings.build_command(channel, brightness))

    def send_all_channels(self, brightness: int) -> dict[int, str]:
        """Send the same clamped brightness to channels 1-4 sequentially -
        the manual documents no single "all channels" command - returning
        each channel's response keyed by channel number.

        Stops at the first failure so a lost link mid-sequence surfaces
        immediately rather than silently skipping the remaining channels.

        Raises:
            LedError: communication failure on any channel (state becomes ERROR).
        """
        return {channel: self.send_channel(channel, brightness) for channel in (1, 2, 3, 4)}

    def send_multichannel(self, states: Sequence[Tuple[int, bool]]) -> str:
        """Documented multi-channel T/F command; *states* is exactly 4
        ``(brightness, on)`` pairs in channel order A-D. Not clamped to
        ``max_brightness`` - like the raw command tester, this format exists
        to test the hardware directly (see ``core.led.protocol``)."""
        return self._send(build_multichannel_command(states))

    def send_raw(self, command: str, *, append_terminator: bool = False) -> str:
        """Manual/raw command tester: sent exactly as typed, with no
        clamping or validation."""
        return self._send(command, append_terminator=append_terminator)

    def _send(self, command: str, *, append_terminator: bool = False) -> str:
        try:
            response = self._client.send(command, append_terminator=append_terminator)
        except LedError as exc:
            self.last_error = str(exc)
            logger.error("LED controller communication error: %s", exc)
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._set_state(ConnectionState.ERROR)
            raise
        logger.info("LED command: %s -> %s", command, response)
        return response

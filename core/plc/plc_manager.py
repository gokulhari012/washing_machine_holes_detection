"""Connection lifecycle and high-level register operations for the PLC.

``PlcManager`` wraps a :class:`PlcClientBase` with:

- a small state machine (DISCONNECTED / CONNECTING / CONNECTED / ERROR) with
  Qt-free observer callbacks (the UI bridges them to signals),
- non-blocking auto-reconnect with an escalating backoff schedule, designed
  to be driven by the PLC poll thread calling :meth:`ensure_connected` each
  tick,
- typed operations for the inspection workflow (trigger, machine number,
  heartbeat, position/result writes) and raw access for the manual register
  viewer on the PLC page.

Thread ownership: all methods are intended to run on the single PLC worker
thread; UI actions must be marshalled onto it by the worker.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from core.logging import get_logger
from core.plc.plc_client_base import PlcClientBase
from core.plc.register_map import RegisterMap
from core.utilities.enums import ConnectionState, LogSource, PlcResultCode
from core.utilities.exceptions import PlcError

logger = get_logger(LogSource.PLC)

StateCallback = Callable[[ConnectionState], None]

# camera index -> (x_mm, y_mm), or None when that camera found no hole
PositionMap = dict[int, tuple[float, float] | None]

DEFAULT_BACKOFF_MS = (1000, 2000, 5000, 10000)


class PlcManager:
    """Owns the PLC connection state and translates workflow intents to registers."""

    def __init__(
        self,
        client: PlcClientBase,
        register_map: RegisterMap,
        reconnect_backoff_ms: list[int] | None = None,
    ) -> None:
        self._client = client
        self._map = register_map
        self._backoff_ms = list(reconnect_backoff_ms or DEFAULT_BACKOFF_MS)
        self._state = ConnectionState.DISCONNECTED
        self._state_lock = threading.Lock()
        self._callbacks: list[StateCallback] = []
        self._heartbeat_value = 0
        self._backoff_index = 0
        self._next_attempt_monotonic = 0.0
        self.last_error: str = ""

    # ----------------------------------------------------------------- state
    @property
    def state(self) -> ConnectionState:
        with self._state_lock:
            return self._state

    @property
    def register_map(self) -> RegisterMap:
        return self._map

    def subscribe_state(self, callback: StateCallback) -> None:
        """Register a callback fired on every state *change* (any thread)."""
        with self._state_lock:
            self._callbacks.append(callback)

    def _set_state(self, new_state: ConnectionState) -> None:
        with self._state_lock:
            if new_state is self._state:
                return
            self._state = new_state
            callbacks = list(self._callbacks)
        logger.info("PLC state -> %s", new_state.value)
        for callback in callbacks:
            try:
                callback(new_state)
            except Exception:  # observers must never break the PLC loop
                logger.exception("PLC state callback raised")

    # ------------------------------------------------------------ connection
    def connect(self) -> None:
        """Blocking connect attempt.

        Raises:
            PlcConnectionError: on failure (state becomes ERROR).
        """
        self._set_state(ConnectionState.CONNECTING)
        try:
            self._client.connect()
        except PlcError as exc:
            self.last_error = str(exc)
            self._set_state(ConnectionState.ERROR)
            raise
        self._backoff_index = 0
        self._next_attempt_monotonic = 0.0
        self._set_state(ConnectionState.CONNECTED)

    def disconnect(self) -> None:
        try:
            self._client.disconnect()
        finally:
            self._set_state(ConnectionState.DISCONNECTED)

    def ensure_connected(self) -> bool:
        """Reconnect helper for the poll loop: cheap when healthy, backs off when not.

        Returns True when the link is usable. Never raises.
        """
        if self._client.connected:
            return True

        now = time.monotonic()
        if now < self._next_attempt_monotonic:
            return False

        try:
            self.connect()
            logger.info("PLC reconnected")
            return True
        except PlcError:
            delay_ms = self._backoff_ms[min(self._backoff_index, len(self._backoff_ms) - 1)]
            self._backoff_index += 1
            self._next_attempt_monotonic = now + delay_ms / 1000.0
            logger.warning(
                "PLC reconnect failed (%s); next attempt in %d ms",
                self.last_error,
                delay_ms,
            )
            return False

    # ------------------------------------------------------ workflow reads
    def read_trigger(self) -> int:
        """Current trigger register value (edge detection is the worker's job)."""
        return self._read(self._map.trigger, 1)[0]

    def read_machine_number(self) -> int:
        return self._read(self._map.machine_number, 1)[0]

    # ----------------------------------------------------- workflow writes
    def toggle_heartbeat(self) -> None:
        """Flip the heartbeat register (0↔1) so the PLC can watchdog the PC."""
        self._heartbeat_value ^= 1
        self._write(self._map.heartbeat, [self._heartbeat_value])

    def write_inspection_output(self, positions: PositionMap, result: PlcResultCode) -> None:
        """Publish one complete inspection to the PLC.

        Writes every camera's X/Y (no-hole sentinel where *positions* holds
        ``None``), then the result code, then raises vision_complete — order
        matters: the PLC may read results the moment vision_complete goes high.
        """
        for camera_index, (x_address, y_address) in sorted(self._map.camera_positions.items()):
            position = positions.get(camera_index)
            if position is None:
                x_raw = y_raw = RegisterMap.NO_HOLE_RAW
            else:
                x_raw = self._map.encode_position(position[0])
                y_raw = self._map.encode_position(position[1])

            if y_address == x_address + 1:  # contiguous pair -> one transaction
                self._write(x_address, [x_raw, y_raw])
            else:
                self._write(x_address, [x_raw])
                self._write(y_address, [y_raw])

        self._write(self._map.result, [int(result)])
        self._write(self._map.vision_complete, [1])
        logger.info("Inspection output written to PLC (result=%s)", result.name)

    # ------------------------------------------------- manual register access
    def read_raw(self, address: int, count: int = 1) -> list[int]:
        """Manual/live register viewer read (PLC Configuration page)."""
        return self._read(address, count)

    def write_raw(self, address: int, value: int) -> None:
        """Manual register write (PLC Configuration page, admin only)."""
        self._write(address, [int(value)])
        logger.info("Manual register write: [%d] = %d", address, value)

    # -------------------------------------------------------------- internal
    def _read(self, address: int, count: int) -> list[int]:
        try:
            return self._client.read_registers(address, count)
        except PlcError as exc:
            self._handle_comm_error(exc)
            raise

    def _write(self, address: int, values: list[int]) -> None:
        try:
            if len(values) == 1:
                self._client.write_register(address, values[0])
            else:
                self._client.write_registers(address, values)
        except PlcError as exc:
            self._handle_comm_error(exc)
            raise

    def _handle_comm_error(self, exc: PlcError) -> None:
        """Drop into ERROR state so ensure_connected() drives recovery."""
        self.last_error = str(exc)
        logger.error("PLC communication error: %s", exc)
        try:
            self._client.disconnect()
        except Exception:
            pass
        self._next_attempt_monotonic = 0.0  # allow immediate first retry
        self._set_state(ConnectionState.ERROR)

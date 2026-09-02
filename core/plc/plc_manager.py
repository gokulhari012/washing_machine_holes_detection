"""Connection lifecycle and high-level register operations for the PLC.

``PlcManager`` wraps a :class:`PlcClientBase` with:

- a small state machine (DISCONNECTED / CONNECTING / CONNECTED / ERROR) with
  Qt-free observer callbacks (the UI bridges them to signals),
- non-blocking auto-reconnect with an escalating backoff schedule, designed
  to be driven by the PLC poll thread calling :meth:`ensure_connected` each
  tick,
- typed operations for the inspection workflow (trigger, machine number,
  heartbeat, position/result writes), camera jog/home for physical alignment,
  and raw access for the manual register viewer on the PLC page.

Thread ownership: connection-lifecycle methods (:meth:`connect`,
:meth:`ensure_connected`, :meth:`disconnect`) are meant to run on the single
PLC worker thread only. Individual register reads/writes are safe to call
from the UI thread too — the underlying client (``ModbusTcpPlcClient``,
``SimulatedPlc``) locks every transaction — which is what the PLC page's
manual write and the camera jog/home buttons do; just note that a *sequence*
of several register writes (like :meth:`write_inspection_output`) is not
atomic across threads, so multi-register workflows stay on the worker thread.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from core.logging import get_logger
from core.plc.plc_client_base import PlcClientBase
from core.plc.register_map import UINT16_MAX, RegisterMap
from core.utilities.enums import ConnectionState, LogSource, PlcResultCode
from core.utilities.exceptions import ConfigurationError, PlcError

logger = get_logger(LogSource.PLC)

StateCallback = Callable[[ConnectionState], None]

# camera index -> (x_mm, y_mm), or None when that camera found no hole
PositionMap = dict[int, tuple[float, float] | None]

DEFAULT_BACKOFF_MS = (1000, 2000, 5000, 10000)

# jog direction -> (dx_sign, dy_sign); the single place that decides which
# way "up"/"down"/"left"/"right" moves the registers. If a real rig turns
# out mirrored, flip the sign of "step" in plc.json rather than here.
_JOG_DIRECTIONS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


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

    def read_camera_trigger(self, camera_index: int) -> int | None:
        """Current per-camera trigger value, or ``None`` when that camera has
        no trigger register configured (no I/O in that case)."""
        address = self._map.camera_triggers.get(camera_index)
        if address is None:
            return None
        return self._read(address, 1)[0]

    def camera_trigger_configured(self, camera_index: int) -> bool:
        return camera_index in self._map.camera_triggers

    def read_model_select(self) -> int | None:
        """Current machine-model code, or ``None`` when the register is not
        configured (no I/O in that case — the feature is simply inert)."""
        if self._map.model_select is None:
            return None
        return self._read(self._map.model_select, 1)[0]

    # ----------------------------------------------------- workflow writes
    def toggle_heartbeat(self) -> None:
        """Flip the heartbeat register (0↔1) so the PLC can watchdog the PC."""
        self._heartbeat_value ^= 1
        self._write(self._map.heartbeat, [self._heartbeat_value])

    def write_inspection_output(
        self,
        positions: PositionMap,
        camera_results: dict[int, PlcResultCode],
        result: PlcResultCode,
    ) -> None:
        """Publish one complete inspection to the PLC.

        Writes every camera's X/Y (no-hole sentinel where *positions* holds
        ``None``), then each camera's own GOOD/NG/ERROR verdict (``ERROR``
        where *camera_results* holds nothing for that camera — not inspected
        this cycle), then the overall result code, then raises
        vision_complete — order matters: the PLC may read results the moment
        vision_complete goes high.
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

        for camera_index, result_address in sorted(self._map.camera_results.items()):
            camera_result = camera_results.get(camera_index, PlcResultCode.ERROR)
            self._write(result_address, [int(camera_result)])

        self._write(self._map.result, [int(result)])
        self._write(self._map.vision_complete, [1])
        logger.info("Inspection output written to PLC (result=%s)", result.name)

    def write_camera_inspection_output(
        self,
        camera_index: int,
        position: tuple[float, float] | None,
        result: PlcResultCode,
    ) -> None:
        """Publish a *single* camera's inspection to the PLC.

        The per-camera counterpart of :meth:`write_inspection_output`: writes
        only this camera's X/Y (no-hole sentinel when *position* is ``None``)
        and its own result register, then raises its own vision_complete —
        the other cameras' registers and the overall result/vision_complete
        registers are left untouched, because the other cameras were not
        inspected this cycle and their last values still stand.

        Registers this camera has not been given are skipped silently, the
        same way :meth:`write_inspection_output` skips absent entries.
        """
        addresses = self._map.camera_positions.get(camera_index)
        if addresses is not None:
            x_address, y_address = addresses
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

        result_address = self._map.camera_results.get(camera_index)
        if result_address is not None:
            self._write(result_address, [int(result)])

        complete_address = self._map.camera_vision_complete.get(camera_index)
        if complete_address is not None:
            self._write(complete_address, [1])
        logger.info(
            "Camera %d inspection output written to PLC (result=%s)",
            camera_index,
            result.name,
        )

    # --------------------------------------------------- camera jog / home
    def jog_camera(self, camera_index: int, direction: str) -> tuple[int, int]:
        """Nudge camera *camera_index*'s physical-position registers one
        step in *direction* ('up'/'down'/'left'/'right'). Read-modify-write,
        clamped to the register's uint16 range; only writes a register whose
        value actually changed.

        Returns:
            The new (x, y) register values.

        Raises:
            ConfigurationError: no jog registers configured for this camera,
                or an unknown direction.
            PlcError: communication failure.
        """
        addresses = self._jog_addresses(camera_index)
        if direction not in _JOG_DIRECTIONS:
            raise ConfigurationError(f"Unknown jog direction: {direction!r}")
        dx_sign, dy_sign = _JOG_DIRECTIONS[direction]
        step = self._map.jog_step
        return self._apply_jog(addresses, dx_sign * step, dy_sign * step)

    def home_camera(self, camera_index: int) -> tuple[int, int]:
        """Write camera *camera_index*'s configured home X/Y values outright.

        Returns:
            The (home_x, home_y) values written.

        Raises:
            ConfigurationError: no jog registers configured for this camera.
        """
        x_addr, y_addr = self._jog_addresses(camera_index)
        home_x, home_y = self._map.camera_jog_home.get(camera_index, (0, 0))
        self._write(x_addr, [home_x])
        self._write(y_addr, [home_y])
        logger.info("Camera %d homed -> (%d, %d)", camera_index, home_x, home_y)
        return home_x, home_y

    def read_camera_jog_position(self, camera_index: int) -> tuple[int, int]:
        """Current physical jog position (x, y) for *camera_index* — the
        value to snapshot as a machine model's default position.

        Raises:
            ConfigurationError: no jog registers configured for this camera.
            PlcError: communication failure.
        """
        x_addr, y_addr = self._jog_addresses(camera_index)
        x = self._read(x_addr, 1)[0]
        y = self._read(y_addr, 1)[0]
        return x, y

    def set_camera_jog_position(self, camera_index: int, x: int, y: int) -> tuple[int, int]:
        """Write camera *camera_index*'s jog X/Y registers directly to
        *(x, y)* — used to restore a machine model's saved default position.

        Raises:
            ConfigurationError: no jog registers configured for this camera.
            PlcError: communication failure.
        """
        x_addr, y_addr = self._jog_addresses(camera_index)
        x = max(0, min(UINT16_MAX, int(x)))
        y = max(0, min(UINT16_MAX, int(y)))
        self._write(x_addr, [x])
        self._write(y_addr, [y])
        logger.info("Camera %d position set -> (%d, %d)", camera_index, x, y)
        return x, y

    def jog_configured(self, camera_index: int) -> bool:
        return camera_index in self._map.camera_jog

    def _jog_addresses(self, camera_index: int) -> tuple[int, int]:
        addresses = self._map.camera_jog.get(camera_index)
        if addresses is None:
            raise ConfigurationError(f"No jog registers configured for camera {camera_index}")
        return addresses

    def _apply_jog(self, addresses: tuple[int, int], dx: int, dy: int) -> tuple[int, int]:
        x_addr, y_addr = addresses
        x = self._read(x_addr, 1)[0]
        y = self._read(y_addr, 1)[0]
        new_x = max(0, min(UINT16_MAX, x + dx))
        new_y = max(0, min(UINT16_MAX, y + dy))
        if new_x != x:
            self._write(x_addr, [new_x])
        if new_y != y:
            self._write(y_addr, [new_y])
        logger.info("Camera jog -> (%d, %d)", new_x, new_y)
        return new_x, new_y

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

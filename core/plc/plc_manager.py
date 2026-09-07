"""Connection lifecycle and high-level register operations for the PLC.

``PlcManager`` wraps a :class:`PlcClientBase` with:

- a small state machine (DISCONNECTED / CONNECTING / CONNECTED / ERROR) with
  Qt-free observer callbacks (the UI bridges them to signals),
- non-blocking auto-reconnect with an escalating backoff schedule, designed
  to be driven by the PLC poll thread calling :meth:`ensure_connected` each
  tick,
- typed operations for the inspection workflow (trigger, machine number,
  heartbeat, position/result writes), camera jog/home for physical alignment,
  and raw access for the manual register viewer on the PLC page,
- :meth:`rebuild` to swap in a new client/register map in place after a PLC
  configuration save, so the connection and register addresses take effect
  live instead of requiring an application restart.

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
from core.utilities.exceptions import CameraBusyError, ConfigurationError, PlcError

logger = get_logger(LogSource.PLC)

StateCallback = Callable[[ConnectionState], None]

# camera index -> (x_mm, y_mm), or None when that camera found no hole
PositionMap = dict[int, tuple[float, float] | None]

DEFAULT_BACKOFF_MS = (1000, 2000, 5000, 10000)

# jog direction -> (axis, sign); the single place that decides which way
# each button moves its register. If a real rig turns out mirrored on an
# axis, flip that axis's sign here rather than touching plc.json.
_JOG_AXES: dict[str, tuple[str, int]] = {
    "x+": ("x", 1),
    "x-": ("x", -1),
    "y+": ("y", 1),
    "y-": ("y", -1),
    "z+": ("z", 1),
    "z-": ("z", -1),
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

    def rebuild(
        self,
        client: PlcClientBase,
        register_map: RegisterMap,
        reconnect_backoff_ms: list[int] | None = None,
    ) -> None:
        """Swap in a new client and register map without replacing this
        ``PlcManager`` instance, so every holder of it (PlcService,
        InspectionService, the poll worker) keeps working unchanged — the
        same trick ``CameraManager.rebuild()`` uses for a live camera-config
        save.

        The caller must ensure nothing is using the outgoing client
        concurrently — in this application that means stopping the PLC poll
        worker first (see ``Application._on_plc_config_saved``). A write
        from an in-flight inspection cycle that loses that race simply sees
        the old, now-disconnected client raise a communication error, which
        the existing fault handling already turns into an ERROR result for
        that one cycle, not a crash.

        Disconnects the outgoing client (best-effort) and resets connection
        state to DISCONNECTED; ``ensure_connected()`` re-establishes the
        link on the next poll tick, exactly like at startup.
        """
        try:
            self._client.disconnect()
        except Exception:  # a dying old client must never block the rebuild
            logger.exception("Error disconnecting outgoing PLC client during rebuild")
        self._client = client
        self._map = register_map
        self._backoff_ms = list(reconnect_backoff_ms or DEFAULT_BACKOFF_MS)
        self._backoff_index = 0
        self._next_attempt_monotonic = 0.0
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

    def clear_trigger(self) -> None:
        """Acknowledge the global trigger by writing 0 back to it.

        Called by the poll loop the moment the edge is detected, so a 0 here
        means *received*. This is deliberately unlike
        :meth:`clear_camera_trigger`, which is released only once that
        camera's cycle has finished.
        """
        self._write(self._map.trigger, [0])

    def clear_camera_trigger(self, camera_index: int) -> bool:
        """Release one per-camera trigger by writing 0 back to it.

        Called from :meth:`write_camera_inspection_output` once that camera's
        inspection is finished and its results are on the wire — *not* when
        the poll loop first saw the edge. A camera trigger still reading 1
        therefore means that camera is mid-inspection. Each camera is released
        on its own, independently of the global trigger and of every other
        camera.

        Returns False (no I/O) when the camera has no trigger register
        configured, matching :meth:`read_camera_trigger`.
        """
        address = self._map.camera_triggers.get(camera_index)
        if address is None:
            return False
        self._write(address, [0])
        return True

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

    def write_model_select(self, code: int) -> bool:
        """Publish *code* to the machine-model-select register.

        Used when a profile is applied from the PC (PLC-triggered or the
        Machine Models page's "Apply Now") so the PLC's own register agrees
        with whichever model is actually active, instead of the PC's choice
        being silently overwritten the next time the poll worker's
        change-detection cache resets and re-reads whatever the PLC still
        holds there (see :mod:`workers.plc_poll_worker`).

        Returns False (no I/O) when the register is not configured — the
        feature is simply inert, same as :meth:`read_model_select`.
        """
        if self._map.model_select is None:
            return False
        self._write(self._map.model_select, [int(code)])
        return True

    def read_servo_home(self, camera_index: int) -> tuple[int, int]:
        """Current servo home position for a camera's X and Y axes.

        The PLC owns these values; a hole position is written relative to
        them (see :mod:`core.plc.register_map`). Returns ``(0, 0)`` when the
        camera has no servo-home registers configured, which makes the
        encoding fall back to plain ``mm * position_scale``.
        """
        addresses = self._map.servo_home_positions.get(camera_index)
        if addresses is None:
            return 0, 0
        x_address, y_address = addresses
        if y_address == x_address + 1:  # contiguous pair -> one transaction
            values = self._read(x_address, 2)
            return values[0], values[1]
        return self._read(x_address, 1)[0], self._read(y_address, 1)[0]

    def _write_position(
        self, camera_index: int, position: tuple[float, float] | None
    ) -> None:
        """Write one camera's X/Y, biased by its servo home position.

        Reads the servo home registers first, so the value the PLC receives is
        an absolute servo target: ``home + mm * position_scale``. ``position``
        of ``None`` is the no-hole case and writes
        :attr:`RegisterMap.NO_HOLE_RAW` to both registers instead — no servo
        read is needed, and none is done. Cameras with no position addresses
        configured are skipped silently.
        """
        addresses = self._map.camera_positions.get(camera_index)
        if addresses is None:
            return
        x_address, y_address = addresses
        if position is None:
            x_raw = y_raw = RegisterMap.NO_HOLE_RAW
        else:
            x_home, y_home = self.read_servo_home(camera_index)
            x_raw = self._map.encode_position(position[0], x_home)
            y_raw = self._map.encode_position(position[1], y_home)

        if y_address == x_address + 1:  # contiguous pair -> one transaction
            self._write(x_address, [x_raw, y_raw])
        else:
            self._write(x_address, [x_raw])
            self._write(y_address, [y_raw])

    def write_inspection_output(
        self,
        positions: PositionMap,
        camera_results: dict[int, PlcResultCode],
        result: PlcResultCode,
    ) -> None:
        """Publish one complete inspection to the PLC.

        Writes every camera's X/Y as a servo target (no-hole sentinel where
        *positions* holds ``None``), then each camera's own GOOD/NG/ERROR
        verdict (``ERROR`` where *camera_results* holds nothing for that
        camera — not inspected this cycle), then the overall result code,
        then raises vision_complete — order matters: the PLC may read results
        the moment vision_complete goes high.
        """
        for camera_index in sorted(self._map.camera_positions):
            self._write_position(camera_index, positions.get(camera_index))

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
        and its own result register, then releases this camera's trigger and
        raises its own vision_complete — the other cameras' registers and the
        overall result/vision_complete registers are left untouched, because
        the other cameras were not inspected this cycle and their last values
        still stand.

        Write order: position → result → **trigger back to 0** →
        vision_complete. The trigger is released here, at the end of the
        cycle, rather than when the poll loop first saw it, so a camera
        trigger sitting high means "this camera is still being inspected".
        It is cleared just *before* vision_complete so that by the moment the
        PLC is told the results are ready, the trigger it raised is already
        released — vision_complete stays strictly last, because the PLC may
        read everything the instant it goes high.

        Registers this camera has not been given are skipped silently, the
        same way :meth:`write_inspection_output` skips absent entries.
        """
        self._write_position(camera_index, position)

        result_address = self._map.camera_results.get(camera_index)
        if result_address is not None:
            self._write(result_address, [int(result)])

        self.clear_camera_trigger(camera_index)

        complete_address = self._map.camera_vision_complete.get(camera_index)
        if complete_address is not None:
            self._write(complete_address, [1])
        logger.info(
            "Camera %d inspection output written to PLC (result=%s)",
            camera_index,
            result.name,
        )

    def write_camera_status(self, camera_index: int, available: bool) -> bool:
        """Publish whether camera *camera_index* is usable right now.

        Writes :attr:`RegisterMap.CAMERA_AVAILABLE` / ``CAMERA_UNAVAILABLE``.
        Returns False (no I/O) when that camera has no status register
        configured, so the feature is simply inert until it is wired up.

        Raises:
            PlcError: communication failure.
        """
        address = self._map.camera_status.get(camera_index)
        if address is None:
            return False
        value = (
            RegisterMap.CAMERA_AVAILABLE if available else RegisterMap.CAMERA_UNAVAILABLE
        )
        self._write(address, [value])
        return True

    def camera_status_configured(self, camera_index: int) -> bool:
        return camera_index in self._map.camera_status

    def write_camera_brightness(self, camera_index: int, level: int) -> bool:
        """Publish camera *camera_index*'s light-brightness level (0-255) to
        its register, clamped to that range — this drives an external,
        PLC-controlled light source, not the camera's own image pipeline.

        Called whenever that camera's settings are applied or saved (see
        ``CameraService``), so the light tracks the configured level the same
        way :meth:`write_camera_status` tracks connectivity. Returns False
        (no I/O) when that camera has no brightness register configured, so
        the feature is simply inert until one is wired up.

        Raises:
            PlcError: communication failure.
        """
        address = self._map.camera_brightness.get(camera_index)
        if address is None:
            return False
        self._write(address, [max(0, min(255, int(level)))])
        return True

    def camera_brightness_configured(self, camera_index: int) -> bool:
        return camera_index in self._map.camera_brightness

    # --------------------------------------------------- camera jog / home
    def jog_camera(self, camera_index: int, direction: str) -> int:
        """Nudge one axis of camera *camera_index*'s physical-position
        registers one step in *direction* ('x+'/'x-'/'y+'/'y-'/'z+'/'z-').
        Read-modify-write on that single register, clamped to the uint16
        range, and only written back if the value actually changed. Raises
        the busy coil (if configured) once the write lands — see
        :meth:`_signal_move_started`.

        Returns:
            That axis's new register value.

        Raises:
            CameraBusyError: this camera's busy coil already reads 1 — a
                previous move hasn't finished yet.
            ConfigurationError: unknown direction, or that axis has no jog
                register configured for this camera (X/Y are configured as
                a pair; Z is independent and optional).
            PlcError: communication failure.
        """
        if direction not in _JOG_AXES:
            raise ConfigurationError(f"Unknown jog direction: {direction!r}")
        self._check_not_busy(camera_index)
        axis, sign = _JOG_AXES[direction]
        address = self._axis_address(camera_index, axis)
        result = self._apply_single_jog(address, sign * self._map.jog_step)
        self._signal_move_started(camera_index)
        return result

    def home_camera(self, camera_index: int) -> tuple[int, int, int]:
        """Move camera *camera_index* to home: X and Y set to zero, and Z
        too if this camera has a Z jog register configured. Home is not a
        stored value — the mount's true home is electrical/mechanical zero.

        Returns:
            ``(0, 0, 0)`` — the third value is 0 regardless of whether a Z
            register is actually configured for this camera.

        Raises:
            CameraBusyError: this camera's busy coil already reads 1.
            ConfigurationError: no X/Y jog registers configured for this
                camera.
            PlcError: communication failure.
        """
        self._check_not_busy(camera_index)
        x_addr, y_addr = self._jog_addresses(camera_index)
        self._write(x_addr, [0])
        self._write(y_addr, [0])
        z_addr = self._map.camera_jog_z.get(camera_index)
        if z_addr is not None:
            self._write(z_addr, [0])
        self._signal_move_started(camera_index)
        logger.info("Camera %d homed -> (0, 0, 0)", camera_index)
        return 0, 0, 0

    def read_camera_jog_position(self, camera_index: int) -> tuple[int, int, int]:
        """Current physical jog position (x, y, z) for *camera_index* — the
        value to snapshot as a machine model's image capture position. z is
        reported as 0 when this camera has no Z jog register configured.

        Raises:
            ConfigurationError: no X/Y jog registers configured for this
                camera.
            PlcError: communication failure.
        """
        x_addr, y_addr = self._jog_addresses(camera_index)
        x = self._read(x_addr, 1)[0]
        y = self._read(y_addr, 1)[0]
        z_addr = self._map.camera_jog_z.get(camera_index)
        z = self._read(z_addr, 1)[0] if z_addr is not None else 0
        return x, y, z

    def set_camera_jog_position(
        self, camera_index: int, x: int, y: int, z: int = 0
    ) -> tuple[int, int, int]:
        """Write camera *camera_index*'s jog X/Y(/Z) registers directly to
        *(x, y, z)* — used to restore a machine model's saved image capture
        position (manually via "Go to Default", or automatically whenever a
        machine model is applied). *z* is silently dropped (returned as 0)
        when this camera has no Z jog register configured, the same way an
        absent X/Y camera is skipped elsewhere in this class.

        Raises:
            CameraBusyError: this camera's busy coil already reads 1.
            ConfigurationError: no X/Y jog registers configured for this
                camera.
            PlcError: communication failure.
        """
        self._check_not_busy(camera_index)
        x_addr, y_addr = self._jog_addresses(camera_index)
        x = max(0, min(UINT16_MAX, int(x)))
        y = max(0, min(UINT16_MAX, int(y)))
        self._write(x_addr, [x])
        self._write(y_addr, [y])
        z_addr = self._map.camera_jog_z.get(camera_index)
        if z_addr is not None:
            z = max(0, min(UINT16_MAX, int(z)))
            self._write(z_addr, [z])
        else:
            z = 0
        self._signal_move_started(camera_index)
        logger.info("Camera %d position set -> (%d, %d, %d)", camera_index, x, y, z)
        return x, y, z

    def jog_configured(self, camera_index: int) -> bool:
        return camera_index in self._map.camera_jog

    def jog_z_configured(self, camera_index: int) -> bool:
        return camera_index in self._map.camera_jog_z

    def jog_busy_configured(self, camera_index: int) -> bool:
        return camera_index in self._map.camera_jog_busy

    def read_camera_jog_busy(self, camera_index: int) -> bool:
        """Current value of camera *camera_index*'s busy/moving coil.

        Returns False (no I/O, never busy) when the camera has no busy coil
        configured — the interlock is simply inert until one is wired up,
        the same way every other optional per-camera register degrades.

        Raises:
            PlcError: communication failure.
        """
        address = self._map.camera_jog_busy.get(camera_index)
        if address is None:
            return False
        return self._read_coil(address)

    def _check_not_busy(self, camera_index: int) -> None:
        """Raise CameraBusyError if this camera is still executing a
        previous move. Called before every jog/home/position write so a
        second command can't be issued on top of one already in flight."""
        if self.read_camera_jog_busy(camera_index):
            raise CameraBusyError(
                f"Camera {camera_index} is still moving — wait for the current move to finish"
            )

    def _signal_move_started(self, camera_index: int) -> None:
        """Raise the busy coil (if configured) so the PLC knows a new
        position was just written and it's time to start moving; the PLC
        clears it back to 0 once the physical move completes. No-op (no I/O)
        when this camera has no busy coil configured."""
        address = self._map.camera_jog_busy.get(camera_index)
        if address is not None:
            self._write_coil(address, True)

    def _jog_addresses(self, camera_index: int) -> tuple[int, int]:
        addresses = self._map.camera_jog.get(camera_index)
        if addresses is None:
            raise ConfigurationError(f"No jog registers configured for camera {camera_index}")
        return addresses

    def _axis_address(self, camera_index: int, axis: str) -> int:
        if axis == "z":
            address = self._map.camera_jog_z.get(camera_index)
            if address is None:
                raise ConfigurationError(
                    f"No Z jog register configured for camera {camera_index}"
                )
            return address
        x_addr, y_addr = self._jog_addresses(camera_index)
        return x_addr if axis == "x" else y_addr

    def _apply_single_jog(self, address: int, delta: int) -> int:
        value = self._read(address, 1)[0]
        new_value = max(0, min(UINT16_MAX, value + delta))
        if new_value != value:
            self._write(address, [new_value])
        logger.info("Camera jog register %d -> %d", address, new_value)
        return new_value

    # ------------------------------------------------- manual register access
    def read_raw(self, address: int, count: int = 1) -> list[int]:
        """Manual/live register viewer read (PLC Configuration page)."""
        return self._read(address, count)

    def write_raw(self, address: int, value: int) -> None:
        """Manual register write (PLC Configuration page, admin only)."""
        self._write(address, [int(value)])
        logger.info("Manual register write: [%d] = %d", address, value)

    def read_raw_coils(self, address: int, count: int = 1) -> list[bool]:
        """Live register table read for a coil row (PLC Configuration page)."""
        return self._read_coils(address, count)

    def write_raw_coil(self, address: int, value: bool) -> None:
        """Manual coil write (PLC Configuration page, admin only)."""
        self._write_coil(address, bool(value))
        logger.info("Manual coil write: [%d] = %s", address, bool(value))

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

    def _read_coil(self, address: int) -> bool:
        return self._read_coils(address, 1)[0]

    def _read_coils(self, address: int, count: int) -> list[bool]:
        try:
            return self._client.read_coils(address, count)
        except PlcError as exc:
            self._handle_comm_error(exc)
            raise

    def _write_coil(self, address: int, value: bool) -> None:
        try:
            self._client.write_coil(address, value)
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

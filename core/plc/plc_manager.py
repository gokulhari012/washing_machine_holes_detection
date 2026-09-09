"""Connection lifecycle and high-level register operations for the PLC.

``PlcManager`` wraps a :class:`PlcClientBase` with:

- a small state machine (DISCONNECTED / CONNECTING / CONNECTED / ERROR) with
  Qt-free observer callbacks (the UI bridges them to signals),
- non-blocking auto-reconnect with an escalating backoff schedule, designed
  to be driven by the PLC poll thread calling :meth:`ensure_connected` each
  tick,
- typed operations for the inspection workflow (trigger, machine number,
  heartbeat, position/result writes) and raw access for the manual register
  viewer on the PLC page,
- :meth:`pause`/:meth:`resume` to suspend every outgoing register/coil write
  except the heartbeat — see their docstrings,
- :meth:`rebuild` to swap in a new client/register map in place after a PLC
  configuration save, so the connection and register addresses take effect
  live instead of requiring an application restart.

Thread ownership: connection-lifecycle methods (:meth:`connect`,
:meth:`ensure_connected`, :meth:`disconnect`) are meant to run on the single
PLC worker thread only. Individual register reads/writes are safe to call
from the UI thread too — the underlying client (``ModbusTcpPlcClient``,
``SimulatedPlc``) locks every transaction — which is what the PLC page's
manual write does; just note that a *sequence*
of several register writes (like :meth:`write_inspection_output`) is not
atomic across threads, so multi-register workflows stay on the worker thread.
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
PausedCallback = Callable[[bool], None]

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
        self._paused = False
        self._paused_callbacks: list[PausedCallback] = []
        self._pause_write_logged = False

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

    # ------------------------------------------------------------- pause
    @property
    def paused(self) -> bool:
        return self._paused

    def subscribe_paused(self, callback: PausedCallback) -> None:
        """Register a callback fired whenever :meth:`pause`/:meth:`resume`
        actually changes state (any thread) — mirrors :meth:`subscribe_state`."""
        with self._state_lock:
            self._paused_callbacks.append(callback)

    def pause(self) -> None:
        """Suspend every outgoing register/coil write except the heartbeat.

        Reads, and everything upstream of a write (trigger edge detection,
        the inspection pipeline, database persistence, the dashboard), keep
        running unchanged — only the write itself is silently skipped, at
        :meth:`_write`/:meth:`_write_coil`, the single choke point every
        write call site already goes through. No ``PlcError`` is raised and
        no alarm fires: a deliberate operator pause is not a communication
        fault.

        In practice this halts the *line*, not just this station's chatter:
        the global trigger's acknowledgement (:meth:`clear_trigger`) and
        every inspection-output write are skipped too, so a paused PLC never
        sees ``vision_complete`` go high and dead-waits on whatever it last
        raised — exactly the effect "pause communication" is for. The
        heartbeat keeps toggling regardless (see :meth:`toggle_heartbeat`),
        so the PLC's watchdog does not trip and drop the link entirely while
        paused.

        Idempotent; a second call while already paused is a no-op (no
        duplicate log line, no duplicate callback).
        """
        if self._paused:
            return
        self._paused = True
        self._pause_write_logged = False
        logger.info("PLC communication paused (writes suspended, heartbeat continues)")
        self._notify_paused(True)

    def resume(self) -> None:
        """Undo :meth:`pause`: writes flow again from the next call site."""
        if not self._paused:
            return
        self._paused = False
        logger.info("PLC communication resumed")
        self._notify_paused(False)

    def _notify_paused(self, paused: bool) -> None:
        with self._state_lock:
            callbacks = list(self._paused_callbacks)
        for callback in callbacks:
            try:
                callback(paused)
            except Exception:  # observers must never break the PLC loop
                logger.exception("PLC paused callback raised")

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

    def read_serial_number(self) -> int | None:
        """Serial number of the machine on the station, or ``None`` when the
        register is not configured (no I/O in that case — the caller then
        falls back to the machine number)."""
        if self._map.serial_number is None:
            return None
        return self._read(self._map.serial_number, 1)[0]

    # ----------------------------------------------------- workflow writes
    def toggle_heartbeat(self) -> None:
        """Flip the heartbeat register (0↔1) so the PLC can watchdog the PC.

        The one write exempt from :meth:`pause` — see its docstring for why.
        """
        self._heartbeat_value ^= 1
        self._write(self._map.heartbeat, [self._heartbeat_value], bypass_pause=True)

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

    def read_gantry_status(self, camera_index: int) -> bool:
        """Whether camera *camera_index*'s gantry is active this cycle.

        The PLC owns this value; the PC reads it live at the start of every
        cycle (global or single-camera) and inspects only the cameras it
        reports active — see :mod:`core.plc.register_map`. Only
        :attr:`RegisterMap.GANTRY_ACTIVE` counts as active, so a garbled value
        skips the camera rather than inspecting a part its gantry is not
        presenting.

        Returns True (no I/O) when the camera has no gantry-status register
        configured: the gate is then simply inert and that camera is always
        inspected, exactly as before this register existed.

        Raises:
            PlcError: communication failure.
        """
        address = self._map.gantry_status.get(camera_index)
        if address is None:
            return True
        return self._read(address, 1)[0] == RegisterMap.GANTRY_ACTIVE

    def gantry_status_configured(self, camera_index: int) -> bool:
        return camera_index in self._map.gantry_status

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
        skipped: "set[int] | frozenset[int] | None" = None,
    ) -> None:
        """Publish one complete inspection to the PLC.

        Writes every camera's X/Y as a servo target (no-hole sentinel where
        *positions* holds ``None``), then each camera's own GOOD/NG/ERROR
        verdict (``ERROR`` where *camera_results* holds nothing for that
        camera — not inspected this cycle), then the overall result code,
        then raises vision_complete — order matters: the PLC may read results
        the moment vision_complete goes high.

        *skipped* names the cameras whose gantry the PLC reported inactive
        (see :meth:`read_gantry_status`). Their position **and** result
        registers are left completely untouched, so whatever the last cycle
        that really inspected them wrote still stands. That is deliberately
        unlike a camera merely missing from *camera_results*, which gets the
        ERROR code: not-inspected-because-the-PLC-said-so is not a fault, and
        writing the no-hole sentinel or ERROR over a stale-but-valid position
        would tell the PLC something the vision system never measured.
        """
        skipped = frozenset(skipped or ())
        for camera_index in sorted(self._map.camera_positions):
            if camera_index in skipped:
                continue
            self._write_position(camera_index, positions.get(camera_index))

        for camera_index, result_address in sorted(self._map.camera_results.items()):
            if camera_index in skipped:
                continue
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

    def write_camera_skipped_output(self, camera_index: int) -> None:
        """Answer a per-camera trigger raised for a camera whose gantry is
        inactive, **without** claiming any measurement.

        Releases that camera's trigger and raises its vision_complete in the
        same order :meth:`write_camera_inspection_output` uses, so the PLC's
        handshake completes and it never dead-waits on a trigger it should
        not have raised — but no position and no result register is written,
        because nothing was captured or judged. The camera's last real values
        therefore still stand, matching how a skipped camera is treated in
        :meth:`write_inspection_output`.
        """
        self.clear_camera_trigger(camera_index)

        complete_address = self._map.camera_vision_complete.get(camera_index)
        if complete_address is not None:
            self._write(complete_address, [1])
        logger.info(
            "Camera %d skipped (gantry inactive) — handshake answered, no result written",
            camera_index,
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

    def _write(self, address: int, values: list[int], *, bypass_pause: bool = False) -> None:
        if self._paused and not bypass_pause:
            self._log_paused_write_skipped(address)
            return
        try:
            if len(values) == 1:
                self._client.write_register(address, values[0])
            else:
                self._client.write_registers(address, values)
        except PlcError as exc:
            self._handle_comm_error(exc)
            raise

    def _read_coils(self, address: int, count: int) -> list[bool]:
        try:
            return self._client.read_coils(address, count)
        except PlcError as exc:
            self._handle_comm_error(exc)
            raise

    def _write_coil(self, address: int, value: bool) -> None:
        if self._paused:
            self._log_paused_write_skipped(address)
            return
        try:
            self._client.write_coil(address, value)
        except PlcError as exc:
            self._handle_comm_error(exc)
            raise

    def _log_paused_write_skipped(self, address: int) -> None:
        """One line per pause session, not per skipped write — a paused
        station can otherwise sit on the poll loop for minutes, and every
        tick's trigger/heartbeat-adjacent writes would each try to log."""
        if self._pause_write_logged:
            return
        self._pause_write_logged = True
        logger.info("PLC write to [%d] skipped (communication paused)", address)

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

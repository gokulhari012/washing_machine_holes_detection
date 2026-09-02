"""PLC polling thread: trigger edge detection, heartbeat, reconnect driving.

Runs a tight loop (default 50 ms) that:

1. keeps the link alive via ``PlcManager.ensure_connected`` (non-blocking
   backoff when the PLC is away),
2. reads the trigger register and emits :attr:`trigger_detected` exactly once
   per **rising edge** — after a (re)connect the first value read is taken as
   baseline, never as an edge, so a trigger frozen high cannot re-fire,
3. reads each configured per-camera trigger register and emits
   :attr:`camera_trigger_detected` on its **rising edge**, to inspect that one
   camera only (same baseline-after-reconnect rule as the global trigger),
4. toggles the heartbeat register (default every 500 ms) so the PLC can
   watchdog the PC,
5. publishes each camera's availability to its status register whenever it
   changes (and re-publishes all of them after a reconnect),
6. reads the machine-model-select register (default every 1000 ms, skipped
   entirely when unconfigured) and emits :attr:`machine_model_changed` on
   every value seen that differs from the last — *including* the first read
   after a (re)connect, unlike the trigger: we want whichever model is
   already selected to load right away, not wait for the PLC to toggle it.

The inspection itself runs on the InspectionWorker — this loop must never be
blocked for longer than one poll interval, or the heartbeat would jitter.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from PySide6.QtCore import QThread, Signal

from core.logging import get_logger
from core.plc import PlcManager
from core.utilities.enums import LogSource
from core.utilities.exceptions import PlcError

logger = get_logger(LogSource.PLC)


class PlcPollWorker(QThread):
    """Owns the poll cadence; all PLC state flows out via PlcManager callbacks."""

    trigger_detected = Signal(int)  # machine number
    camera_trigger_detected = Signal(int, int)  # camera index, machine number
    machine_model_changed = Signal(int)  # new model_select value

    def __init__(
        self,
        plc_manager: PlcManager,
        poll_interval_ms: int = 50,
        heartbeat_interval_ms: int = 500,
        model_poll_interval_ms: int = 1000,
        camera_status_provider: "Callable[[], dict[int, bool]] | None" = None,
    ) -> None:
        super().__init__()
        self.setObjectName("PlcPollWorker")
        self._manager = plc_manager
        self._camera_status_provider = camera_status_provider
        self._poll_interval_s = max(0.01, poll_interval_ms / 1000.0)
        self._heartbeat_interval_s = max(0.1, heartbeat_interval_ms / 1000.0)
        self._model_poll_interval_s = max(0.1, model_poll_interval_ms / 1000.0)
        self._stop_event = threading.Event()
        self._last_trigger: int | None = None  # None = re-baseline required
        self._last_model: int | None = None  # None = not yet read this session
        # per-camera trigger edge state, same baseline rule as _last_trigger
        self._last_camera_triggers: dict[int, int | None] = {
            index: None for index in plc_manager.register_map.camera_triggers
        }
        # last availability value actually written per camera; cleared on any
        # link loss so the PLC is refreshed after a reconnect or power-cycle
        self._last_camera_status: dict[int, bool] = {}

    # ------------------------------------------------------------------ api
    def stop(self, timeout_ms: int = 3000) -> None:
        """Request shutdown and join the thread."""
        self._stop_event.set()
        if not self.wait(timeout_ms):
            logger.error("PLC poll worker did not stop within %d ms", timeout_ms)

    # ----------------------------------------------------------------- loop
    def run(self) -> None:  # executes in the worker thread
        logger.info(
            "PLC poll worker started (poll %.0f ms, heartbeat %.0f ms)",
            self._poll_interval_s * 1000,
            self._heartbeat_interval_s * 1000,
        )
        last_heartbeat = 0.0
        last_model_poll = 0.0

        while not self._stop_event.is_set():
            tick_started = time.monotonic()

            if self._manager.ensure_connected():
                try:
                    value = self._manager.read_trigger()
                    if self._last_trigger is None:
                        # first read after (re)connect: baseline, not an edge
                        self._last_trigger = value
                    elif value == 1 and self._last_trigger == 0:
                        machine_number = self._manager.read_machine_number()
                        logger.info(
                            "Trigger edge detected (machine %d)", machine_number
                        )
                        self.trigger_detected.emit(machine_number)
                    self._last_trigger = value

                    self._poll_camera_triggers()
                    self._publish_camera_status()

                    if tick_started - last_heartbeat >= self._heartbeat_interval_s:
                        self._manager.toggle_heartbeat()
                        last_heartbeat = tick_started

                    if tick_started - last_model_poll >= self._model_poll_interval_s:
                        last_model_poll = tick_started
                        model = self._manager.read_model_select()
                        if model is not None and model != self._last_model:
                            logger.info("Machine model select changed -> %d", model)
                            self._last_model = model
                            self.machine_model_changed.emit(model)
                except PlcError:
                    # manager already logged, moved to ERROR state and armed
                    # its backoff; require a fresh baseline after recovery
                    self._last_trigger = None
                    self._last_model = None
                    self._rebaseline_camera_triggers()
                    self._last_camera_status.clear()
            else:
                self._last_trigger = None
                self._last_model = None
                self._rebaseline_camera_triggers()
                self._last_camera_status.clear()

            elapsed = time.monotonic() - tick_started
            remaining = self._poll_interval_s - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

        logger.info("PLC poll worker stopped")

    # ------------------------------------------------------ camera triggers
    def _poll_camera_triggers(self) -> None:
        """Rising-edge detection on each configured per-camera trigger.

        Same baseline rule as the global trigger: the first value seen after a
        (re)connect is adopted, never fired, so a camera trigger left high
        cannot re-fire on reconnect. Runs on every poll tick, so a per-camera
        trigger is picked up as fast as the global one.
        """
        for camera_index in sorted(self._last_camera_triggers):
            value = self._manager.read_camera_trigger(camera_index)
            if value is None:  # register not configured
                continue
            previous = self._last_camera_triggers[camera_index]
            if previous is None:
                self._last_camera_triggers[camera_index] = value
                continue
            if value == 1 and previous == 0:
                machine_number = self._manager.read_machine_number()
                logger.info(
                    "Camera %d trigger edge detected (machine %d)",
                    camera_index,
                    machine_number,
                )
                self.camera_trigger_detected.emit(camera_index, machine_number)
            self._last_camera_triggers[camera_index] = value

    def _rebaseline_camera_triggers(self) -> None:
        for camera_index in self._last_camera_triggers:
            self._last_camera_triggers[camera_index] = None

    # -------------------------------------------------------- camera status
    def _publish_camera_status(self) -> None:
        """Write each camera's availability, but only when it changed.

        Camera state changes originate on whichever thread hit them (startup
        connect, an acquisition grab, an inspection capture); pushing them to
        the PLC from here instead keeps all PLC I/O on this thread and means a
        change that happens while the link is down is not lost — the cache is
        cleared on any link loss, so the next healthy tick re-publishes every
        camera to a PLC that may have been power-cycled.
        """
        if self._camera_status_provider is None:
            return
        for camera_index, available in sorted(self._camera_status_provider().items()):
            if self._last_camera_status.get(camera_index) == available:
                continue
            if self._manager.write_camera_status(camera_index, available):
                logger.info(
                    "Camera %d reported to PLC as %s",
                    camera_index,
                    "available" if available else "unavailable",
                )
            self._last_camera_status[camera_index] = available

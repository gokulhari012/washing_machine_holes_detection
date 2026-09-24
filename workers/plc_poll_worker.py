"""PLC polling thread: trigger edge detection, heartbeat, reconnect driving.

Runs a tight loop (default 50 ms) that:

1. keeps the link alive via ``PlcManager.ensure_connected`` (non-blocking
   backoff when the PLC is away),
2. reads the trigger register and emits :attr:`trigger_detected` exactly once
   per **rising edge** — after a (re)connect the first value read is taken as
   baseline, never as an edge, so a trigger frozen high cannot re-fire,
   and writes 0 back to the register to acknowledge it,
3. reads each configured per-camera trigger register and emits
   :attr:`camera_trigger_detected` on its **rising edge**, to inspect that one
   camera only (same baseline-after-reconnect rule as the global trigger);
   unlike the global trigger it is *not* cleared here — see below,
4. toggles the heartbeat register (default every 500 ms) so the PLC can
   watchdog the PC,
5. publishes each camera's availability to its status register whenever it
   changes (and re-publishes all of them after a reconnect),
6. reads the machine-model-select register (default every 1000 ms, skipped
   entirely when unconfigured) and emits :attr:`machine_model_changed` on
   every value seen that differs from the last — *including* the first read
   after a (re)connect, unlike the trigger: we want whichever model is
   already selected to load right away, not wait for the PLC to toggle it.

Trigger acknowledgement
-----------------------
The PC clears every trigger register it acts on, so the PLC only has to raise
one, never lower it. A PLC program that still clears its own triggers stays
correct — it is simply writing 0 over a 0. *When* the 0 is written differs
between the two kinds of trigger, on purpose:

* the **global trigger** is cleared here, on the tick that detects the edge
  and before the inspection has run, so a 0 means *received*. Completion is
  the separate signal on vision_complete, which is what the PLC waits on.
  The edge state is baselined on the value actually *read* (1), never on the
  0 written back, so a PLC that keeps driving the trigger high until it sees
  vision_complete cannot re-fire the cycle every tick: a 0 has to be observed
  on the wire before the next 1 counts as an edge. It is then cleared **again
  after the cycle**, twice — see below.
* a **per-camera trigger** is cleared at the *end* of that camera's cycle, by
  ``PlcManager.write_camera_inspection_output`` on the inspection thread, just
  before that camera's vision_complete goes high. A camera trigger still
  reading 1 therefore means that camera is mid-inspection.

Each register is handled independently: releasing camera 2's trigger touches
neither camera 3's nor the global one.

Clearing the global trigger again after the cycle
-------------------------------------------------
The edge-detection clear above is written *before* the cycle runs, so a PLC
that drives the trigger high until it sees vision_complete simply overwrites
it and the register sits at 1 for the whole cycle. :meth:`notify_cycle_finished`
(wired to ``InspectionWorker.inspection_finished`` in the composition root, for
full cycles only) therefore arms **two** more clears of that same register:

* one on the next poll tick, i.e. as soon as the cycle is finished, and
* one ``trigger_clear_delay_ms`` later — 1 s by default — for the PLC that is
  still holding the line high at the moment the first one lands, or that
  re-asserts it in between.

Both go through the same guard, and that guard is the part worth keeping: a
clear is written **only when the trigger is still high from the cycle that
just ran** (read 1 this tick *and* baselined at 1, so not a fresh edge). A
trigger already at 0 is left alone rather than re-zeroed, and a **fresh rising
edge is never swallowed** — the edge branch above runs first on every tick and
owns that 1, so a new trigger raised during the delay window starts its cycle
in the normal way and the pending clear declines to touch it. A new cycle
simply re-arms both clears when it finishes.

Neither clear survives a link loss: a pending clear is dropped when the link
drops, because writing 0 to a trigger this worker never acted on would fake a
handshake that never happened — the same reason a trigger frozen high across a
reconnect is not acknowledged.

Set ``trigger_clear_delay_ms`` to 0 to keep only the end-of-cycle clear.

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
        trigger_clear_delay_ms: int = 1000,
        camera_status_provider: "Callable[[], dict[int, bool]] | None" = None,
    ) -> None:
        super().__init__()
        self.setObjectName("PlcPollWorker")
        self._manager = plc_manager
        self._camera_status_provider = camera_status_provider
        self._poll_interval_s = max(0.01, poll_interval_ms / 1000.0)
        self._heartbeat_interval_s = max(0.1, heartbeat_interval_ms / 1000.0)
        self._model_poll_interval_s = max(0.1, model_poll_interval_ms / 1000.0)
        # Delay between a finished cycle and the second, time-based clear of
        # the global trigger. 0 disables that one; the end-of-cycle clear on
        # the next tick always happens.
        self._trigger_clear_delay_s = max(0.0, trigger_clear_delay_ms / 1000.0)
        # Post-cycle clears, armed from the inspection thread by
        # notify_cycle_finished and serviced on this thread. Guarded because
        # they are written and read from different threads.
        self._clear_lock = threading.Lock()
        self._clear_trigger_now = False
        self._clear_trigger_at: float | None = None
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
    def notify_cycle_finished(self, cycle: object = None) -> None:
        """Arm the two post-cycle clears of the global trigger.

        Called from whichever thread finished the cycle (the inspection
        thread, via ``InspectionWorker.inspection_finished``), so it only
        records the intent — the register is written on the poll thread like
        every other PLC I/O this worker owns. Re-arming while a delayed clear
        is still pending simply restarts the delay from this cycle, which is
        what a back-to-back pair of cycles should do.

        *cycle* is accepted and ignored so the method can be connected
        straight to a signal that carries the finished cycle.
        """
        with self._clear_lock:
            self._clear_trigger_now = True
            self._clear_trigger_at = (
                time.monotonic() + self._trigger_clear_delay_s
                if self._trigger_clear_delay_s > 0
                else None
            )

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
                    previous = self._last_trigger
                    if self._last_trigger is None:
                        # first read after (re)connect: baseline, not an edge
                        self._last_trigger = value
                    elif value == 1 and self._last_trigger == 0:
                        machine_number = self._manager.read_machine_number()
                        logger.info(
                            "Trigger edge detected (machine %d)", machine_number
                        )
                        self.trigger_detected.emit(machine_number)
                        # Acknowledge immediately: the PLC only has to raise
                        # the trigger, never lower it. Baseline on the 1 we
                        # actually read, *not* on the 0 we just wrote: a PLC
                        # that drives the trigger high until it sees
                        # vision_complete overwrites that 0, and assuming it
                        # stuck makes every following tick look like a fresh
                        # rising edge — one PLC trigger then runs the cycle
                        # over and over. Requiring a 0 to be *observed* before
                        # the next edge costs at most one tick when our clear
                        # does stick, and the PLC waits on vision_complete
                        # before raising the next trigger anyway.
                        self._manager.clear_trigger()
                    self._last_trigger = value

                    # After the edge branch, so a fresh rising edge is always
                    # claimed by the cycle it belongs to before any post-cycle
                    # clear gets to look at the register.
                    self._service_trigger_clears(value, previous, tick_started)

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
                    self._discard_pending_trigger_clears()
            else:
                self._last_trigger = None
                self._last_model = None
                self._rebaseline_camera_triggers()
                self._last_camera_status.clear()
                self._discard_pending_trigger_clears()

            elapsed = time.monotonic() - tick_started
            remaining = self._poll_interval_s - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

        logger.info("PLC poll worker stopped")

    # -------------------------------------------------- post-cycle clearing
    def _service_trigger_clears(
        self, value: int, previous: int | None, now: float
    ) -> None:
        """Write the post-cycle 0 to the global trigger, if one is due.

        Two independent one-shots are armed by :meth:`notify_cycle_finished`:
        an immediate one, serviced on the first tick after the cycle, and a
        delayed one ``trigger_clear_delay_ms`` after it. Both are serviced
        here, on the poll thread, with the trigger value this tick already
        read — no extra round trip.

        *value* is what the register reads now and *previous* the baseline it
        was compared against. A clear is written only for ``value == 1`` with
        ``previous == 1``: the trigger is still high, and it is the same 1
        this worker already acted on rather than a fresh edge (which the edge
        branch above has just claimed and cleared itself). Anything else needs
        no write — a trigger already at 0 is left alone rather than
        re-zeroed.
        """
        with self._clear_lock:
            immediate = self._clear_trigger_now
            deadline = self._clear_trigger_at
            delayed = deadline is not None and now >= deadline
            if not immediate and not delayed:
                return
            self._clear_trigger_now = False
            if delayed:
                self._clear_trigger_at = None

        if value == 1 and previous == 1:
            self._manager.clear_trigger()
            logger.info(
                "Global trigger still high after the cycle — cleared (%s)",
                "delayed" if delayed and not immediate else "end of cycle",
            )

    def _discard_pending_trigger_clears(self) -> None:
        """Forget any armed post-cycle clear on link loss.

        Writing 0 to the trigger after a reconnect would acknowledge a
        handshake this worker never completed — the same reason a trigger
        found frozen high at (re)connect is baselined rather than cleared.
        """
        with self._clear_lock:
            self._clear_trigger_now = False
            self._clear_trigger_at = None

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
                # No acknowledgement here: a camera trigger is released at the
                # *end* of its cycle, by write_camera_inspection_output on the
                # inspection thread. Baselining on the 1 we just read is what
                # makes that work — the release then reads as a falling edge,
                # and the PLC's next 1 as a fresh rising one.
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

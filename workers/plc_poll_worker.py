"""PLC polling thread: trigger edge detection, heartbeat, reconnect driving.

Runs a tight loop (default 50 ms) that:

1. keeps the link alive via ``PlcManager.ensure_connected`` (non-blocking
   backoff when the PLC is away),
2. reads the trigger register and emits :attr:`trigger_detected` exactly once
   per **rising edge** — after a (re)connect the first value read is taken as
   baseline, never as an edge, so a trigger frozen high cannot re-fire,
3. toggles the heartbeat register (default every 500 ms) so the PLC can
   watchdog the PC.

The inspection itself runs on the InspectionWorker — this loop must never be
blocked for longer than one poll interval, or the heartbeat would jitter.
"""

from __future__ import annotations

import threading
import time

from PySide6.QtCore import QThread, Signal

from core.logging import get_logger
from core.plc import PlcManager
from core.utilities.enums import LogSource
from core.utilities.exceptions import PlcError

logger = get_logger(LogSource.PLC)


class PlcPollWorker(QThread):
    """Owns the poll cadence; all PLC state flows out via PlcManager callbacks."""

    trigger_detected = Signal(int)  # machine number

    def __init__(
        self,
        plc_manager: PlcManager,
        poll_interval_ms: int = 50,
        heartbeat_interval_ms: int = 500,
    ) -> None:
        super().__init__()
        self.setObjectName("PlcPollWorker")
        self._manager = plc_manager
        self._poll_interval_s = max(0.01, poll_interval_ms / 1000.0)
        self._heartbeat_interval_s = max(0.1, heartbeat_interval_ms / 1000.0)
        self._stop_event = threading.Event()
        self._last_trigger: int | None = None  # None = re-baseline required

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

                    if tick_started - last_heartbeat >= self._heartbeat_interval_s:
                        self._manager.toggle_heartbeat()
                        last_heartbeat = tick_started
                except PlcError:
                    # manager already logged, moved to ERROR state and armed
                    # its backoff; require a fresh baseline after recovery
                    self._last_trigger = None
            else:
                self._last_trigger = None

            elapsed = time.monotonic() - tick_started
            remaining = self._poll_interval_s - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

        logger.info("PLC poll worker stopped")

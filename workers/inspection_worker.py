"""Inspection execution thread.

Hosts :class:`InspectionService` on a dedicated QThread using the
worker-object pattern: the poll worker's ``trigger_detected`` signal is
connected (queued) to :meth:`on_trigger`, so cycles execute here and the PLC
poll loop keeps its 50 ms cadence (heartbeat never stalls behind a capture).

PLC output writes issued during the cycle interleave safely with polling via
the PLC client's internal lock.

``trigger_requested`` lets the UI start a manual cycle ("Simulate Trigger")
from the main thread — emitting the signal marshals the call onto this
thread automatically.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal, Slot

from core.logging import get_logger
from core.utilities.enums import LogSource
from services.inspection_service import InspectionService

logger = get_logger(LogSource.VISION)


class InspectionWorker(QObject):
    """Queued, serialised execution of inspection cycles."""

    #: emit to run a cycle from any thread (UI "Simulate Trigger" button)
    trigger_requested = Signal(int)
    #: emit to inspect one camera from any thread (dashboard per-camera button);
    #: arguments are (camera_index, machine_number)
    camera_trigger_requested = Signal(int, int)
    #: InspectionCycleData, after the cycle fully completed
    inspection_finished = Signal(object)

    def __init__(self, inspection_service: InspectionService) -> None:
        super().__init__()
        self._service = inspection_service
        self._busy = False  # touched only on the worker thread (queued slots)
        self._thread = QThread()
        self._thread.setObjectName("InspectionWorker")
        self.moveToThread(self._thread)
        self.trigger_requested.connect(self.on_trigger)
        self.camera_trigger_requested.connect(self.on_camera_trigger)

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._thread.start()
        logger.info("Inspection worker started")

    def stop(self, timeout_ms: int = 10000) -> None:
        """Let a running cycle finish, then stop the event loop and join."""
        self._thread.quit()
        if not self._thread.wait(timeout_ms):
            logger.error("Inspection worker did not stop within %d ms", timeout_ms)

    # ----------------------------------------------------------------- slot
    @Slot(int)
    def on_trigger(self, machine_number: int) -> None:
        """Run one cycle. Re-entry cannot normally happen (the PLC handshake
        serialises triggers); if it does, the extra trigger is dropped loudly."""
        if self._busy:
            logger.warning(
                "Trigger for machine %d ignored — inspection already running "
                "(check PLC handshake configuration)",
                machine_number,
            )
            return
        self._busy = True
        try:
            cycle = self._service.run_inspection(machine_number)
            self.inspection_finished.emit(cycle)
        finally:
            self._busy = False

    @Slot(int, int)
    def on_camera_trigger(self, camera_index: int, machine_number: int) -> None:
        """Inspect a single camera. Shares the busy flag with :meth:`on_trigger`
        so a per-camera trigger and a full cycle can never overlap on the same
        cameras — whichever arrives second is dropped loudly."""
        if self._busy:
            logger.warning(
                "Camera %d trigger ignored — inspection already running",
                camera_index,
            )
            return
        self._busy = True
        try:
            cycle = self._service.run_camera_inspection(camera_index, machine_number)
            self.inspection_finished.emit(cycle)
        finally:
            self._busy = False

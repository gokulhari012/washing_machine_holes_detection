"""Observable application state — the single hub between workers and UI.

``AppState`` is a QObject whose signals are the *only* channel through which
background threads reach the UI. Emitting a Qt signal is thread-safe from any
Python thread; receivers connected from the UI thread get queued delivery.

It also keeps a lock-protected snapshot (counters, connection states, last
trigger) so newly-opened pages can render current values immediately instead
of waiting for the next event.
"""

from __future__ import annotations

import threading
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from core.utilities.enums import ConnectionState, InspectionResult
from models.dto import InspectionCycleData, LogEvent


class AppState(QObject):
    """View-model hub: signals for events, thread-safe snapshot for state."""

    # connection / hardware
    plc_state_changed = Signal(str)            # ConnectionState value
    camera_state_changed = Signal(int, str)    # camera index, ConnectionState value
    active_machine_model_changed = Signal(str, int)  # profile name, PLC code

    # inspection flow
    trigger_received = Signal(int)             # machine number
    camera_captured = Signal(int, object)      # camera index, freshly grabbed frame
    camera_inspected = Signal(int, object)     # camera index, CameraInspectionData
    inspection_completed = Signal(object)      # InspectionCycleData
    counters_changed = Signal(int, int, int)   # total, good, ng

    # live view / logs / alarms
    preview_frame = Signal(int, object)        # camera index, np.ndarray (BGR)
    log_event = Signal(object)                 # LogEvent
    alarm_raised = Signal(str, str)            # severity ("warning"|"error"), message
    alarm_cleared = Signal()
    status_message = Signal(str)               # transient status-bar text

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._plc_state = ConnectionState.DISCONNECTED
        self._camera_states: dict[int, ConnectionState] = {}
        self._total = 0
        self._good = 0
        self._ng = 0
        self._last_trigger_at: datetime | None = None
        self._last_machine_number: int | None = None
        self._active_machine_model: str = ""
        self._active_machine_model_code: int | None = None

    # ------------------------------------------------------------- updaters
    def update_plc_state(self, state: ConnectionState) -> None:
        with self._lock:
            self._plc_state = state
        self.plc_state_changed.emit(state.value)

    def update_camera_state(self, camera_index: int, state: ConnectionState) -> None:
        with self._lock:
            self._camera_states[camera_index] = state
        self.camera_state_changed.emit(camera_index, state.value)

    def set_active_machine_model(self, name: str, plc_code: int) -> None:
        with self._lock:
            self._active_machine_model = name
            self._active_machine_model_code = plc_code
        self.active_machine_model_changed.emit(name, plc_code)

    def notify_trigger(self, machine_number: int) -> None:
        with self._lock:
            self._last_trigger_at = datetime.now()
            self._last_machine_number = machine_number
        self.trigger_received.emit(machine_number)

    def publish_camera_capture(self, camera_index: int, frame) -> None:
        """One camera just took its picture (sequential capture progress)."""
        self.camera_captured.emit(camera_index, frame)

    def publish_camera_result(self, camera_index: int, data) -> None:
        """One camera has been judged, before the whole cycle is finished."""
        self.camera_inspected.emit(camera_index, data)

    def set_counters(self, total: int, good: int, ng: int) -> None:
        """Initialise counters from the database at startup / day rollover."""
        with self._lock:
            self._total, self._good, self._ng = total, good, ng
        self.counters_changed.emit(total, good, ng)

    def publish_inspection(self, cycle: InspectionCycleData) -> None:
        with self._lock:
            self._total += 1
            if cycle.overall_result is InspectionResult.GOOD:
                self._good += 1
            else:
                self._ng += 1
            total, good, ng = self._total, self._good, self._ng
        self.inspection_completed.emit(cycle)
        self.counters_changed.emit(total, good, ng)

    def publish_preview(self, camera_index: int, frame) -> None:
        self.preview_frame.emit(camera_index, frame)

    def post_log(self, event: LogEvent) -> None:
        self.log_event.emit(event)

    def raise_alarm(self, severity: str, message: str) -> None:
        self.alarm_raised.emit(severity, message)

    def clear_alarm(self) -> None:
        self.alarm_cleared.emit()

    def post_status(self, message: str) -> None:
        self.status_message.emit(message)

    # ------------------------------------------------------------- snapshot
    @property
    def plc_state(self) -> ConnectionState:
        with self._lock:
            return self._plc_state

    @property
    def camera_states(self) -> dict[int, ConnectionState]:
        with self._lock:
            return dict(self._camera_states)

    @property
    def counters(self) -> tuple[int, int, int]:
        with self._lock:
            return self._total, self._good, self._ng

    @property
    def last_trigger_at(self) -> datetime | None:
        with self._lock:
            return self._last_trigger_at

    @property
    def last_machine_number(self) -> int | None:
        with self._lock:
            return self._last_machine_number

    @property
    def active_machine_model(self) -> tuple[str, int | None]:
        with self._lock:
            return self._active_machine_model, self._active_machine_model_code

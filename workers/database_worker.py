"""Asynchronous, batched persistence of log records.

Wired to the logging fan-out handler in the composition root:

    log_manager.fanout.add_callback(database_worker.enqueue_log_record)

Every INFO+ record from any thread becomes a :class:`LogEvent` that is
(a) forwarded live to the Logs page via ``AppState.post_log`` and
(b) queued here and flushed to SQLite in batches, so the hot inspection path
never waits on a log INSERT.

Feedback-loop guard: if a flush itself fails, the error is logged with
``extra={"no_db": True}`` which this worker refuses to enqueue — a down
database cannot generate an ever-growing queue of its own failure messages.

(Inspection rows are persisted synchronously on the inspection worker — WAL
makes that a millisecond operation and keeps result ordering strict.)
"""

from __future__ import annotations

import logging
import queue
import time

from PySide6.QtCore import QThread

from core.logging import get_logger
from core.utilities.enums import LogSource
from core.utilities.exceptions import DatabaseError
from models.app_state import AppState
from models.dto import LogEvent
from services.database_service import DatabaseService

logger = get_logger(LogSource.DATABASE)

_STOP = object()  # queue sentinel


class DatabaseWorker(QThread):
    """Drains a thread-safe queue of LogEvents into the ``logs`` table."""

    def __init__(
        self,
        database_service: DatabaseService,
        app_state: AppState,
        flush_interval_s: float = 1.0,
        batch_limit: int = 200,
    ) -> None:
        super().__init__()
        self.setObjectName("DatabaseWorker")
        self._database = database_service
        self._app_state = app_state
        self._flush_interval_s = flush_interval_s
        self._batch_limit = batch_limit
        self._queue: queue.Queue = queue.Queue()

    # ------------------------------------------------------------ producers
    def enqueue_log_record(self, record: logging.LogRecord) -> None:
        """Fan-out callback; called from ANY thread, must stay cheap."""
        if record.levelno < logging.INFO or getattr(record, "no_db", False):
            return
        event = LogEvent.from_record(record)
        self._app_state.post_log(event)  # live Logs page (queued signal)
        self._queue.put(event)

    # ------------------------------------------------------------ lifecycle
    def stop(self, timeout_ms: int = 5000) -> None:
        """Flush remaining records and join."""
        self._queue.put(_STOP)
        if not self.wait(timeout_ms):
            logger.error("Database worker did not stop within %d ms", timeout_ms)

    # ----------------------------------------------------------------- loop
    def run(self) -> None:
        logger.info("Database worker started")
        buffer: list[LogEvent] = []
        last_flush = time.monotonic()
        while True:
            try:
                item = self._queue.get(timeout=self._flush_interval_s)
            except queue.Empty:
                item = None

            if item is _STOP:
                self._flush(buffer)
                break
            if item is not None:
                buffer.append(item)

            # flush on size OR elapsed time — a steady trickle of events must
            # not defer persistence indefinitely
            now = time.monotonic()
            if buffer and (
                len(buffer) >= self._batch_limit
                or now - last_flush >= self._flush_interval_s
            ):
                self._flush(buffer)
                last_flush = now

        logger.info("Database worker stopped", extra={"no_db": True})

    def _flush(self, buffer: list[LogEvent]) -> None:
        if not buffer:
            return
        try:
            self._database.save_log_events(buffer)
        except DatabaseError as exc:
            logger.error(
                "Log persistence failed, %d records dropped: %s",
                len(buffer),
                exc,
                extra={"no_db": True},  # never re-enqueue our own failure
            )
        buffer.clear()

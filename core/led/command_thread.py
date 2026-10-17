"""Ordered LED command execution on one daemon thread.

LED controller commands must reach the port in the order they were issued -
a strobe's light-off must never overtake its light-on - yet no caller may be
held on the port for longer than it chooses to wait. ``CommandThread`` runs
submitted callables one at a time, in submission order, on a single daemon
thread and hands each caller a ``Future`` to wait on with whatever deadline
suits it. A caller that gives up leaves its command to finish in the
background, still in order.

Daemon, so a call wedged inside a serial driver never holds up process exit;
and the thread exits after :data:`IDLE_EXIT_S` with nothing to do, so a client
replaced by an LED configuration save does not leave a parked thread behind.

Used twice: inside ``SerialLedClient`` for the port itself (no serial call
ever runs on a caller's thread), and by ``InspectionService`` around every
strobe command, so an inspection cycle stays bounded whatever ``LedManager``
or client it was built with.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from typing import Callable

#: Seconds the thread lingers with an empty queue before exiting.
IDLE_EXIT_S = 5.0


class CommandThread:
    """One daemon thread executing submitted callables strictly in order."""

    def __init__(self, name: str, *, max_backlog: int) -> None:
        self._name = name
        self._max_backlog = max_backlog
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._pending = 0
        self._started_at: float | None = None
        self._thread: threading.Thread | None = None

    @property
    def pending(self) -> int:
        """Commands queued or running."""
        with self._lock:
            return self._pending

    def running_for(self) -> float:
        """Seconds the command now executing has been running; 0.0 when idle."""
        with self._lock:
            if self._started_at is None:
                return 0.0
            return time.monotonic() - self._started_at

    def submit(self, fn: Callable[[], object], *, force: bool = False) -> Future | None:
        """Queue *fn* behind everything already submitted.

        Returns ``None`` - nothing queued - when ``max_backlog`` commands are
        already waiting, unless *force* is set: closing or opening the port
        must always get in line, however far behind a stuck command it is.
        """
        future: Future = Future()
        with self._lock:
            if self._pending >= self._max_backlog and not force:
                return None
            self._pending += 1
            # Queued under the lock, so the idle exit in _run cannot miss it.
            self._queue.put((fn, future))
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name=self._name, daemon=True
                )
                self._thread.start()
        return future

    def _run(self) -> None:
        while True:
            try:
                fn, future = self._queue.get(timeout=IDLE_EXIT_S)
            except queue.Empty:
                with self._lock:
                    if self._queue.empty():
                        self._thread = None
                        return
                continue
            with self._lock:
                self._started_at = time.monotonic()
            try:
                future.set_result(fn())
            except BaseException as exc:  # handed to whoever waits on it
                future.set_exception(exc)
            finally:
                with self._lock:
                    self._started_at = None
                    self._pending -= 1

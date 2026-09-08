"""Shift resolution: the single answer to "which shift is it".

Everything that needs a shift name -- the inspection pipeline stamping a
cycle, the dashboard tile, the status bar, the Database Viewer's filter --
asks this service rather than reading ``app_config.json`` itself, so there is
exactly one place where "automatic, by the clock" versus "manual, as chosen on
the Settings page" is decided.

Deliberately Qt-free, like every other service: it owns no timer and starts no
thread. The composition root drives :meth:`poll` from a ``QTimer`` on the GUI
thread and publishes the result to ``AppState``; the pipeline calls
:meth:`current_name` directly on the inspection thread. Both are safe because
the parsed schedule is cached under a lock and ``ConfigManager`` is itself
thread-safe.

**Why polling rather than a timer armed at the next boundary:** an industrial
PC's clock gets corrected (NTP, a manual fix at commissioning, a DST step). A
single-shot timer armed hours ahead would then fire at the wrong moment or
strand the station on a stale shift until restart. Re-reading the clock every
:data:`POLL_INTERVAL_MS` is self-correcting -- after any clock jump the next
tick lands on the right shift -- and costs a dictionary lookup a minute.

A malformed ``shifts`` block must not take the line down: the schedule falls
back to the shipped default rota and the fault is logged once, the same way a
bad camera entry degrades rather than aborts startup.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Callable

from core.logging import get_logger
from core.utilities import ConfigManager
from core.utilities.enums import LogSource
from core.utilities.exceptions import ConfigurationError
from core.utilities.shift_schedule import Shift, ShiftSchedule

logger = get_logger(LogSource.SYSTEM)

#: How often the composition root re-resolves the clock. 30 s bounds how long
#: the displayed shift can lag a handover, which is far finer than any shift
#: boundary needs and still nothing next to a 50 ms PLC tick.
POLL_INTERVAL_MS = 30_000

ShiftCallback = Callable[[str], None]


class ShiftService:
    """Reads the rota from config; answers the current shift name."""

    def __init__(self, config_manager: ConfigManager) -> None:
        self._config = config_manager
        self._lock = threading.RLock()
        self._schedule: ShiftSchedule | None = None
        self._manual: str = ""
        self._current: str = ""
        self._observers: list[ShiftCallback] = []
        self._warned_invalid = False
        # A save of app_config from anywhere (the Settings page is the only
        # writer today) drops the cache, so the very next resolution already
        # uses the new rota -- no restart, and no explicit call from the page.
        self._config.subscribe("app_config", self._on_config_saved)

    # ---------------------------------------------------------------- config
    def schedule(self) -> ShiftSchedule:
        """The parsed rota, cached until app_config is saved again."""
        with self._lock:
            if self._schedule is None:
                self._schedule, self._manual = self._read()
            return self._schedule

    def manual_name(self) -> str:
        """The shift chosen by hand on the Settings page (``application.shift``)."""
        with self._lock:
            if self._schedule is None:
                self._schedule, self._manual = self._read()
            return self._manual

    def _read(self) -> tuple[ShiftSchedule, str]:
        """Parse app_config, degrading to the shipped rota on a bad block."""
        try:
            cfg = self._config.load("app_config")
        except ConfigurationError as exc:
            self._warn_once("Shift schedule unreadable (%s); using the default rota", exc)
            return ShiftSchedule.defaults(), ""
        manual = str(cfg.get("application", {}).get("shift", ""))
        try:
            schedule = ShiftSchedule.from_config(cfg.get("shifts"))
        except ConfigurationError as exc:
            self._warn_once("Shift schedule invalid (%s); using the default rota", exc)
            return ShiftSchedule.defaults(), manual
        self._warned_invalid = False
        return schedule, manual

    def _warn_once(self, message: str, *args: Any) -> None:
        """Log a bad rota once per fault, not once per poll tick."""
        if not self._warned_invalid:
            self._warned_invalid = True
            logger.warning(message, *args)

    def _on_config_saved(self, _cfg: dict[str, Any]) -> None:
        with self._lock:
            self._schedule = None
        # Re-resolve immediately so a Save that moves a boundary across "now"
        # is visible without waiting out the poll interval.
        self.poll()

    # --------------------------------------------------------------- queries
    def current_shift(self, moment: datetime | None = None) -> Shift | None:
        """The :class:`Shift` covering *moment* (default: now), automatic only."""
        schedule = self.schedule()
        if not schedule.automatic:
            return None
        return schedule.shift_at(moment or datetime.now())

    def current_name(self, moment: datetime | None = None) -> str:
        """The shift name to stamp on a cycle / show on screen.

        Automatic mode resolves the clock and, where the rota leaves a gap,
        falls back to the manually chosen name rather than stamping an empty
        string -- an uncovered hour still produced parts, and a blank shift
        column is worse for reporting than a slightly wrong one.
        """
        schedule = self.schedule()
        manual = self.manual_name()
        if not schedule.automatic:
            return manual
        shift = schedule.shift_at(moment or datetime.now())
        return shift.name if shift is not None else manual

    def next_change_after(self, moment: datetime | None = None) -> datetime | None:
        """When the rota next changes, or ``None`` in manual mode."""
        schedule = self.schedule()
        if not schedule.automatic:
            return None
        return schedule.next_change_after(moment or datetime.now())

    # -------------------------------------------------------------- observers
    def subscribe(self, callback: ShiftCallback) -> None:
        """Call *callback* with the new name whenever the shift changes."""
        with self._lock:
            self._observers.append(callback)

    def poll(self) -> str | None:
        """Re-resolve the clock; return the new name if it changed, else ``None``.

        Driven by the composition root's timer. Observers are notified outside
        the lock, and a raising observer never breaks the caller -- the same
        fan-out contract every other observer site in this codebase keeps.
        """
        name = self.current_name()
        with self._lock:
            if name == self._current:
                return None
            previous, self._current = self._current, name
            observers = list(self._observers)
        logger.info("Shift changed: %s -> %s", previous or "(none)", name or "(none)")
        for callback in observers:
            try:
                callback(name)
            except Exception:
                logger.exception("Shift observer raised")
        return name

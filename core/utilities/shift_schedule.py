"""Production shift rota: named windows of the day, resolved by the clock.

The station runs a standard three-shift rota (morning / evening / night).
Every inspection is stamped with the shift it was produced in so the line's
output can be reported per shift; before this module that stamp was a *manual*
``application.shift`` letter on the Settings page, which meant it was only
correct for as long as somebody remembered to change it at handover.

A :class:`ShiftSchedule` is the parsed, validated form of ``app_config.json``'s
``shifts`` block. It is a pure value object: no Qt, no I/O and no clock of its
own -- the caller passes the moment in. That keeps it trivially testable and
lets the *same* object answer both "which shift is it right now" (the live
dashboard) and "which shift was 03:14 last Tuesday in" (reporting).

Three conventions worth knowing:

* **A shift may wrap midnight.** ``22:00 -> 06:00`` is the night shift, and a
  window is half-open -- ``start`` inclusive, ``end`` exclusive -- so two
  back-to-back shifts sharing a boundary never both claim the same instant.
* **First match wins.** Overlapping windows are not rejected, because a plant
  is entitled to a deliberate overlap; :meth:`ShiftSchedule.shift_at` returns
  the first shift in configured order that contains the moment.
* **A gap returns ``None``**, which the caller degrades however it likes
  (``ShiftService`` falls back to the manually selected shift name).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Iterable

from core.utilities.exceptions import ConfigurationError

#: The rota only ever cares about time of day, but ``shift_at`` takes a
#: datetime (it is the same call the live pipeline makes). Analysis helpers
#: pair a clock time with this arbitrary date to reuse it unchanged.
_ANY_DATE = date(2000, 1, 1)

#: Shipped rota -- the ordinary 8-hour three-shift day. ``config/defaults/``
#: mirrors this, and it is also what a config carrying no ``shifts`` block
#: resolves to, so an app_config.json written before this feature keeps
#: loading without being rewritten.
DEFAULT_SHIFT_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("morning", "Morning", "06:00", "14:00"),
    ("evening", "Evening", "14:00", "22:00"),
    ("night", "Night", "22:00", "06:00"),
)


def parse_clock(value: Any, *, field: str) -> time:
    """Parse ``"HH:MM"`` (or ``"HH:MM:SS"``) into a :class:`~datetime.time`.

    Raises:
        ConfigurationError: not a string, or not a 24-hour clock time.
    """
    if isinstance(value, time):
        return value
    if not isinstance(value, str):
        raise ConfigurationError(f"Shift {field} must be a 'HH:MM' string, got {value!r}")
    text = value.strip()
    for pattern in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(text, pattern).time()
        except ValueError:
            continue
    raise ConfigurationError(f"Shift {field} is not a valid 24-hour time: {value!r}")


def format_clock(value: time) -> str:
    """Render a time back to the ``"HH:MM"`` form stored in JSON."""
    return value.strftime("%H:%M")


@dataclass(frozen=True)
class Shift:
    """One named window of the day.

    ``id`` is the stable key (``morning``/``evening``/``night``) that the
    config file and the Settings page address a row by; ``name`` is the
    operator-facing label that gets stamped onto every inspection, so it --
    not the id -- is what appears in the database, the exports and on the
    dashboard. Renaming a shift therefore changes what new records say and
    leaves historical ones alone, which is the behaviour a rename should have.
    """

    id: str
    name: str
    start: time
    end: time

    @property
    def wraps_midnight(self) -> bool:
        return self.start > self.end

    def contains(self, moment: time) -> bool:
        """Is *moment* inside this shift? ``start`` inclusive, ``end`` exclusive."""
        if self.wraps_midnight:
            return moment >= self.start or moment < self.end
        return self.start <= moment < self.end

    @property
    def duration(self) -> timedelta:
        """Length of the window, counting a midnight wrap as going forward."""
        start = timedelta(
            hours=self.start.hour, minutes=self.start.minute, seconds=self.start.second
        )
        end = timedelta(hours=self.end.hour, minutes=self.end.minute, seconds=self.end.second)
        span = end - start
        return span + timedelta(days=1) if span <= timedelta(0) else span

    @property
    def window_text(self) -> str:
        """``"06:00-14:00"`` -- for captions and tooltips."""
        return f"{format_clock(self.start)}-{format_clock(self.end)}"

    @classmethod
    def from_config(cls, cfg: Any, *, position: int) -> "Shift":
        """Build one shift from its JSON entry.

        Raises:
            ConfigurationError: the entry is not an object, has a blank name,
                or covers a zero-length window.
        """
        if not isinstance(cfg, dict):
            raise ConfigurationError(f"Shift #{position} must be an object, got {cfg!r}")
        identifier = str(cfg.get("id", "") or f"shift{position}").strip()
        name = str(cfg.get("name", "") or "").strip()
        if not name:
            raise ConfigurationError(f"Shift '{identifier}' has no name")
        start = parse_clock(cfg.get("start"), field=f"'{identifier}' start")
        end = parse_clock(cfg.get("end"), field=f"'{identifier}' end")
        if start == end:
            # Ambiguous: it could equally mean "never" or "all day". Rejecting
            # it is better than silently picking one -- the operator is told on
            # Save instead of wondering why the shift never changes.
            raise ConfigurationError(
                f"Shift '{identifier}' starts and ends at {format_clock(start)} - "
                f"a shift must cover a non-zero span of the day"
            )
        return cls(id=identifier, name=name, start=start, end=end)

    def to_config(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "start": format_clock(self.start),
            "end": format_clock(self.end),
        }


@dataclass(frozen=True)
class OverlapWindow:
    """A stretch of the day claimed by more than one shift.

    ``shifts`` is in configured order, so ``shifts[0]`` is the one that
    actually wins those minutes and the rest are shadowed there.
    """

    shifts: tuple[Shift, ...]
    start: time
    end: time

    @property
    def winner(self) -> Shift:
        return self.shifts[0]

    @property
    def shadowed(self) -> tuple[Shift, ...]:
        return self.shifts[1:]

    @property
    def window_text(self) -> str:
        return f"{format_clock(self.start)}-{format_clock(self.end)}"


@dataclass(frozen=True)
class ShiftSchedule:
    """The whole rota plus the automatic/manual switch.

    ``automatic`` is what the Settings page's checkbox writes: with it on, the
    stamped shift follows the clock through :meth:`shift_at`; with it off the
    schedule is still parsed and displayed, but the caller keeps using the
    manually chosen shift name.
    """

    shifts: tuple[Shift, ...] = ()
    automatic: bool = True

    # -------------------------------------------------------------- parsing
    @classmethod
    def defaults(cls) -> "ShiftSchedule":
        return cls(
            shifts=tuple(
                Shift(
                    id=key,
                    name=label,
                    start=parse_clock(start, field=key),
                    end=parse_clock(end, field=key),
                )
                for key, label, start, end in DEFAULT_SHIFT_SPECS
            ),
            automatic=True,
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "ShiftSchedule":
        """Parse ``app_config.json``'s ``shifts`` block.

        A missing or empty block yields :meth:`defaults` -- that is what makes
        an app_config.json written before this feature keep loading.

        Raises:
            ConfigurationError: an entry is malformed (see :meth:`Shift.from_config`).
        """
        if not isinstance(cfg, dict):
            return cls.defaults()
        entries = cfg.get("schedule")
        if not isinstance(entries, list) or not entries:
            return cls.defaults()
        shifts = tuple(
            Shift.from_config(entry, position=position)
            for position, entry in enumerate(entries, start=1)
        )
        return cls(shifts=shifts, automatic=bool(cfg.get("automatic", True)))

    @classmethod
    def from_shifts(cls, shifts: Iterable[Shift], *, automatic: bool = True) -> "ShiftSchedule":
        return cls(shifts=tuple(shifts), automatic=automatic)

    def to_config(self) -> dict[str, Any]:
        return {
            "automatic": self.automatic,
            "schedule": [shift.to_config() for shift in self.shifts],
        }

    # -------------------------------------------------------------- queries
    def get(self, shift_id: str) -> Shift | None:
        for shift in self.shifts:
            if shift.id == shift_id:
                return shift
        return None

    @property
    def names(self) -> list[str]:
        """Operator-facing labels in configured order -- the manual dropdown."""
        return [shift.name for shift in self.shifts]

    def shift_at(self, moment: datetime) -> Shift | None:
        """The shift covering *moment*, or ``None`` if the rota has a gap there."""
        clock = moment.time()
        for shift in self.shifts:
            if shift.contains(clock):
                return shift
        return None

    def name_at(self, moment: datetime) -> str:
        """:meth:`shift_at`'s name, or ``""`` for an uncovered moment."""
        shift = self.shift_at(moment)
        return shift.name if shift is not None else ""

    def next_change_after(self, moment: datetime) -> datetime | None:
        """When the rota next changes shift, strictly after *moment*.

        Every shift *start* is a boundary; the soonest one after *moment*
        wins, searching today then tomorrow so a night shift's 06:00 handover
        resolves correctly just before midnight. ``None`` when no shifts are
        configured. Used for display only -- the live shift is re-resolved by
        polling the clock, not by scheduling a timer at this instant, so that
        a system clock correction cannot strand the application on a stale
        shift.
        """
        if not self.shifts:
            return None
        candidates = [
            datetime.combine(moment.date() + timedelta(days=day), shift.start)
            for day in (0, 1)
            for shift in self.shifts
        ]
        future = [candidate for candidate in candidates if candidate > moment]
        return min(future) if future else None

    def coverage_gaps(self) -> list[tuple[time, time]]:
        """Windows of the day no shift claims -- what the Settings page warns on.

        Computed by sweeping the 1440 minutes of a day rather than by interval
        arithmetic: shifts may overlap *and* may wrap midnight, and a
        minute-wise sweep handles both without a pile of special cases. It is
        cheap enough at this cadence -- this runs on Save and on page load,
        never per inspection cycle.
        """
        gaps: list[tuple[time, time]] = []
        run_start: time | None = None
        for minute in range(24 * 60):
            clock = time(hour=minute // 60, minute=minute % 60)
            covered = any(shift.contains(clock) for shift in self.shifts)
            if not covered and run_start is None:
                run_start = clock
            elif covered and run_start is not None:
                gaps.append((run_start, clock))
                run_start = None
        if run_start is not None:
            # An uncovered run that reaches the end of the day closes at midnight.
            gaps.append((run_start, time(0, 0)))
        return gaps

    def overlaps(self) -> list["OverlapWindow"]:
        """Windows two or more shifts both claim, and which one actually wins.

        An overlap is legal (a plant may run a deliberate handover overlap) and
        is resolved by configured order, so this is not an error -- it exists
        so the Settings page can *say* that an hour is double-claimed rather
        than letting it pass unremarked. That silence was the real trap: a gap
        announces itself, an overlap does not.

        Same minute sweep as :meth:`coverage_gaps`, and with the same
        convention -- a run touching the end of the day closes at midnight
        rather than being merged with one starting at 00:00, so the windows
        reported are always within a single day.
        """
        windows: list[OverlapWindow] = []
        run_start: time | None = None
        run_owners: tuple[Shift, ...] = ()
        for minute in range(24 * 60):
            clock = time(hour=minute // 60, minute=minute % 60)
            owners = tuple(shift for shift in self.shifts if shift.contains(clock))
            contested = owners if len(owners) > 1 else ()
            if contested != run_owners:
                if run_owners and run_start is not None:
                    windows.append(OverlapWindow(run_owners, run_start, clock))
                run_start = clock if contested else None
                run_owners = contested
        if run_owners and run_start is not None:
            windows.append(OverlapWindow(run_owners, run_start, time(0, 0)))
        return windows

    def unreachable(self) -> list[Shift]:
        """Shifts that never win a single minute of the day.

        A shift fully masked by one listed before it is still configured, still
        shown on the Settings page and still looks reasonable -- but it can
        never be stamped on an inspection. That is the failure mode of setting
        a start later than an end by mistake: the shift becomes a 22-hour
        window that swallows the two after it, and nothing else in the rota
        looks wrong. Reported separately from :meth:`overlaps` because the
        operator needs to be told the shift is *dead*, not merely contested.
        """
        winners = {
            shift.id
            for minute in range(24 * 60)
            for shift in (self.shift_at(datetime.combine(_ANY_DATE, time(minute // 60, minute % 60))),)
            if shift is not None
        }
        return [shift for shift in self.shifts if shift.id not in winners]

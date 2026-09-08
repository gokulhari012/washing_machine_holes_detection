"""ShiftSchedule + ShiftService: the rota and how the application reads it.

Every test pins an explicit ``datetime`` rather than using ``datetime.now()``
-- the whole point of keeping the schedule clock-free is that the moment is an
argument, and a test that read the wall clock would pass or fail depending on
what time the suite ran.

``ShiftService`` is exercised against a stub ConfigManager with the same
surface the real one exposes to it (``load`` and ``subscribe``), so no file is
touched and a save can be simulated by mutating the document and firing the
subscriber the way ``ConfigManager.save`` does.
"""

from __future__ import annotations

import copy
from datetime import datetime, time

import pytest

from core.utilities.exceptions import ConfigurationError
from core.utilities.shift_schedule import Shift, ShiftSchedule, format_clock, parse_clock
from services.shift_service import ShiftService


def at(hour: int, minute: int = 0) -> datetime:
    """A fixed calendar day -- only the time of day matters to the rota."""
    return datetime(2026, 3, 17, hour, minute)


# --------------------------------------------------------------------- parsing
def test_parse_clock_accepts_hh_mm_and_hh_mm_ss() -> None:
    assert parse_clock("06:00", field="start") == time(6, 0)
    assert parse_clock("22:30:45", field="start") == time(22, 30, 45)


@pytest.mark.parametrize("value", ["", "6", "25:00", "06:99", "morning", None, 600])
def test_parse_clock_rejects_junk(value) -> None:
    with pytest.raises(ConfigurationError):
        parse_clock(value, field="start")


def test_format_clock_round_trips() -> None:
    assert format_clock(parse_clock("06:05", field="start")) == "06:05"


# ------------------------------------------------------------------- windows
def test_default_rota_covers_every_hour_of_the_day() -> None:
    schedule = ShiftSchedule.defaults()
    assert schedule.names == ["Morning", "Evening", "Night"]
    assert schedule.coverage_gaps() == []


@pytest.mark.parametrize(
    "hour, expected",
    [
        (0, "Night"), (5, "Night"), (5, "Night"),
        (6, "Morning"), (10, "Morning"), (13, "Morning"),
        (14, "Evening"), (18, "Evening"), (21, "Evening"),
        (22, "Night"), (23, "Night"),
    ],
)
def test_default_rota_resolves_each_hour(hour: int, expected: str) -> None:
    assert ShiftSchedule.defaults().name_at(at(hour)) == expected


def test_boundary_belongs_to_the_starting_shift_not_the_ending_one() -> None:
    """Half-open windows: 14:00 is the first minute of Evening, and the last
    minute of Morning is 13:59 -- no instant is claimed twice."""
    schedule = ShiftSchedule.defaults()
    assert schedule.name_at(at(13, 59)) == "Morning"
    assert schedule.name_at(at(14, 0)) == "Evening"
    assert schedule.name_at(at(21, 59)) == "Evening"
    assert schedule.name_at(at(22, 0)) == "Night"


def test_night_shift_wraps_midnight_in_both_directions() -> None:
    night = ShiftSchedule.defaults().get("night")
    assert night is not None and night.wraps_midnight
    assert night.contains(time(23, 59))
    assert night.contains(time(0, 0))
    assert night.contains(time(5, 59))
    assert not night.contains(time(6, 0))


def test_shift_duration_counts_a_midnight_wrap_forwards() -> None:
    schedule = ShiftSchedule.defaults()
    assert all(
        shift.duration.total_seconds() == 8 * 3600 for shift in schedule.shifts
    ), [str(shift.duration) for shift in schedule.shifts]


# ------------------------------------------------------------ next_change_after
def test_next_change_is_the_soonest_upcoming_start() -> None:
    schedule = ShiftSchedule.defaults()
    assert schedule.next_change_after(at(9, 30)) == datetime(2026, 3, 17, 14, 0)
    assert schedule.next_change_after(at(15, 0)) == datetime(2026, 3, 17, 22, 0)


def test_next_change_rolls_over_midnight() -> None:
    """Late in the night shift the next boundary is tomorrow's 06:00."""
    schedule = ShiftSchedule.defaults()
    assert schedule.next_change_after(at(23, 30)) == datetime(2026, 3, 18, 6, 0)


def test_next_change_is_strictly_after_the_moment() -> None:
    """Standing exactly on a boundary must advance, not return that boundary
    again -- otherwise a poll landing on 14:00:00 would report no progress."""
    schedule = ShiftSchedule.defaults()
    assert schedule.next_change_after(at(14, 0)) == datetime(2026, 3, 17, 22, 0)


def test_next_change_of_an_empty_rota_is_none() -> None:
    assert ShiftSchedule().next_change_after(at(9)) is None


# -------------------------------------------------------------------- config
def test_missing_block_falls_back_to_the_shipped_rota() -> None:
    """An app_config.json written before this feature must keep loading."""
    for absent in (None, {}, {"schedule": []}, {"schedule": "nonsense"}, "nope"):
        assert ShiftSchedule.from_config(absent).names == ["Morning", "Evening", "Night"]


def test_round_trips_through_to_config() -> None:
    original = ShiftSchedule.defaults()
    assert ShiftSchedule.from_config(original.to_config()) == original


def test_automatic_flag_is_carried_through_config() -> None:
    doc = ShiftSchedule.defaults().to_config()
    doc["automatic"] = False
    assert ShiftSchedule.from_config(doc).automatic is False


def test_zero_length_shift_is_rejected() -> None:
    doc = ShiftSchedule.defaults().to_config()
    doc["schedule"][0]["end"] = doc["schedule"][0]["start"]
    with pytest.raises(ConfigurationError, match="non-zero span"):
        ShiftSchedule.from_config(doc)


def test_blank_name_is_rejected() -> None:
    doc = ShiftSchedule.defaults().to_config()
    doc["schedule"][1]["name"] = "   "
    with pytest.raises(ConfigurationError, match="no name"):
        ShiftSchedule.from_config(doc)


def test_renaming_a_shift_keeps_its_id() -> None:
    doc = ShiftSchedule.defaults().to_config()
    doc["schedule"][0]["name"] = "Early"
    schedule = ShiftSchedule.from_config(doc)
    assert schedule.get("morning") is not None
    assert schedule.name_at(at(7)) == "Early"


# ----------------------------------------------------------- gaps + overlaps
def test_a_gap_reports_no_shift_and_is_listed() -> None:
    schedule = ShiftSchedule.from_shifts(
        [
            Shift("day", "Day", time(8, 0), time(16, 0)),
            Shift("late", "Late", time(16, 0), time(23, 0)),
        ]
    )
    assert schedule.shift_at(at(3)) is None
    assert schedule.name_at(at(3)) == ""
    assert schedule.coverage_gaps() == [(time(0, 0), time(8, 0)), (time(23, 0), time(0, 0))]


def test_overlapping_shifts_resolve_to_the_first_configured() -> None:
    """A deliberate overlap is allowed; configured order breaks the tie."""
    schedule = ShiftSchedule.from_shifts(
        [
            Shift("a", "A", time(6, 0), time(15, 0)),
            Shift("b", "B", time(14, 0), time(22, 0)),
        ]
    )
    assert schedule.name_at(at(14, 30)) == "A"
    # ...and the shift that would have owned it alone still owns the rest.
    assert schedule.name_at(at(16, 0)) == "B"


def test_overlap_leaves_only_the_genuinely_uncovered_window() -> None:
    schedule = ShiftSchedule.from_shifts(
        [
            Shift("a", "A", time(6, 0), time(15, 0)),
            Shift("b", "B", time(14, 0), time(22, 0)),
        ]
    )
    assert schedule.coverage_gaps() == [(time(0, 0), time(6, 0)), (time(22, 0), time(0, 0))]


# --------------------------------------------------------------- ShiftService
class StubConfig:
    """The slice of ConfigManager that ShiftService actually uses."""

    def __init__(self, document: dict) -> None:
        self.document = document
        self._subscribers: list = []

    def load(self, name: str) -> dict:
        assert name == "app_config"
        return copy.deepcopy(self.document)

    def subscribe(self, name: str, callback) -> None:
        assert name == "app_config"
        self._subscribers.append(callback)

    def save(self, document: dict) -> None:
        """Mimic ConfigManager.save's notification, which is what invalidates
        the service's cached rota."""
        self.document = document
        for callback in list(self._subscribers):
            callback(copy.deepcopy(document))


def make_service(*, automatic: bool = True, manual: str = "Fallback") -> tuple[ShiftService, StubConfig]:
    schedule = ShiftSchedule.defaults().to_config()
    schedule["automatic"] = automatic
    config = StubConfig({"application": {"shift": manual}, "shifts": schedule})
    return ShiftService(config), config


def test_service_resolves_the_clock_in_automatic_mode() -> None:
    service, _ = make_service()
    assert service.current_name(at(7)) == "Morning"
    assert service.current_name(at(23)) == "Night"


def test_service_uses_the_manual_name_when_automatic_is_off() -> None:
    service, _ = make_service(automatic=False, manual="Evening")
    assert service.current_name(at(7)) == "Evening"
    assert service.current_shift(at(7)) is None
    assert service.next_change_after(at(7)) is None


def test_service_falls_back_to_the_manual_name_inside_a_gap() -> None:
    """An uncovered hour still produced parts; a blank shift column is worse
    for reporting than the manually selected one."""
    schedule = ShiftSchedule.from_shifts(
        [Shift("day", "Day", time(8, 0), time(16, 0))]
    ).to_config()
    config = StubConfig({"application": {"shift": "Unscheduled"}, "shifts": schedule})
    service = ShiftService(config)
    assert service.current_name(at(10)) == "Day"
    assert service.current_name(at(3)) == "Unscheduled"


def test_service_degrades_to_the_default_rota_on_a_malformed_block() -> None:
    """A bad config must not take the line down."""
    config = StubConfig(
        {"application": {"shift": "M"}, "shifts": {"schedule": [{"id": "x", "name": "X"}]}}
    )
    service = ShiftService(config)
    assert service.schedule().names == ["Morning", "Evening", "Night"]
    assert service.current_name(at(7)) == "Morning"


def test_saving_the_config_reloads_the_rota() -> None:
    service, config = make_service()
    assert service.current_name(at(7)) == "Morning"

    document = copy.deepcopy(config.document)
    document["shifts"]["schedule"][0]["name"] = "Early"
    config.save(document)

    assert service.current_name(at(7)) == "Early"


def test_poll_reports_only_actual_changes_and_notifies_observers() -> None:
    service, config = make_service()
    seen: list[str] = []
    service.subscribe(seen.append)

    first = service.poll()  # from "" to whatever the rota says now
    assert first is not None and seen == [first]
    assert service.poll() is None  # unchanged -> no second notification
    assert seen == [first]

    # Force a change by renaming the shift that is current right now.
    document = copy.deepcopy(config.document)
    for entry in document["shifts"]["schedule"]:
        entry["name"] = f"Renamed {entry['id']}"
    config.save(document)  # the save itself polls, via the subscription

    assert len(seen) == 2
    assert seen[1].startswith("Renamed ")
    assert service.poll() is None


def test_a_raising_observer_does_not_break_the_poll() -> None:
    """Same fan-out contract every observer site in this codebase keeps."""
    service, _ = make_service()
    calls: list[str] = []

    def boom(_name: str) -> None:
        raise RuntimeError("observer fault")

    service.subscribe(boom)
    service.subscribe(calls.append)
    assert service.poll() is not None
    assert len(calls) == 1

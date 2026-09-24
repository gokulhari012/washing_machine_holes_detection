"""The global trigger is re-zeroed after every full cycle, twice.

The edge-detection clear is written *before* the cycle runs, so a PLC that
drives the trigger high until it sees vision_complete simply overwrites it and
register 100 sits at 1 for the whole cycle. ``notify_cycle_finished`` arms two
more clears of it: one on the next poll tick (end of cycle) and one
``trigger_clear_delay_ms`` later, 1 s by default.

What these tests are really guarding is the condition on those writes — a
clear must never swallow a *fresh* trigger, must not fake a handshake after a
link loss, and must not write to a register that is already 0.

Loop-driving conventions (plain thread, DirectConnection) match
``tests/test_plc_poll_worker.py``.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest
from PySide6.QtCore import Qt

from core.plc import PlcManager, RegisterMap, SimulatedPlc
from workers.plc_poll_worker import PlcPollWorker

from tests.test_register_map import make_config

# Short enough to keep the tests quick, long enough to tell the two clears
# apart at a 10 ms poll interval.
DELAY_MS = 200


@pytest.fixture()
def stack() -> tuple[SimulatedPlc, PlcManager, RegisterMap]:
    config = make_config()
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    return client, manager, rmap


@contextmanager
def running(worker: PlcPollWorker):
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        worker._stop_event.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def hold_trigger_high(client: SimulatedPlc, rmap: RegisterMap) -> list[float]:
    """Make the PLC win every fight over the trigger register, as one driving
    it high until vision_complete does, and record when the PC tried.

    Returns the list of monotonic timestamps of each attempted clear, which is
    what the writes-suppressed case lets us assert on at all.
    """
    attempts: list[float] = []
    original = client.write_register

    def write(address: int, value: int) -> None:
        if address == rmap.trigger and value == 0:
            attempts.append(time.monotonic())
            return  # PLC re-asserts it immediately
        original(address, value)

    client.write_register = write
    return attempts


def test_a_trigger_held_high_through_the_cycle_is_cleared_again_afterwards(stack) -> None:
    """The point of the feature: the cycle's own clear was overwritten, so a
    fresh one is written once the cycle is done."""
    client, manager, rmap = stack
    worker = PlcPollWorker(manager, poll_interval_ms=10, trigger_clear_delay_ms=0)
    attempts = hold_trigger_high(client, rmap)

    with running(worker):
        assert wait_until(lambda: worker._last_trigger == 0)
        client.set_register(rmap.trigger, 1)
        assert wait_until(lambda: len(attempts) == 1)  # the edge-detection clear

        worker.notify_cycle_finished()
        assert wait_until(lambda: len(attempts) == 2)  # end of cycle


def test_the_delayed_clear_follows_the_configured_delay(stack) -> None:
    """The second, time-based clear lands roughly trigger_clear_delay_ms after
    the cycle — not on the very next tick with the first one."""
    client, manager, rmap = stack
    worker = PlcPollWorker(manager, poll_interval_ms=10, trigger_clear_delay_ms=DELAY_MS)
    attempts = hold_trigger_high(client, rmap)

    with running(worker):
        assert wait_until(lambda: worker._last_trigger == 0)
        client.set_register(rmap.trigger, 1)
        assert wait_until(lambda: len(attempts) == 1)

        worker.notify_cycle_finished()
        armed_at = time.monotonic()
        assert wait_until(lambda: len(attempts) == 3)

    end_of_cycle, delayed = attempts[1], attempts[2]
    assert end_of_cycle - armed_at < DELAY_MS / 1000.0  # the immediate one
    assert delayed - armed_at >= DELAY_MS / 1000.0      # the timed one


def test_zero_delay_keeps_only_the_end_of_cycle_clear(stack) -> None:
    """0 disables the timed clear; the end-of-cycle one always happens."""
    client, manager, rmap = stack
    worker = PlcPollWorker(manager, poll_interval_ms=10, trigger_clear_delay_ms=0)
    attempts = hold_trigger_high(client, rmap)

    with running(worker):
        assert wait_until(lambda: worker._last_trigger == 0)
        client.set_register(rmap.trigger, 1)
        assert wait_until(lambda: len(attempts) == 1)

        worker.notify_cycle_finished()
        assert wait_until(lambda: len(attempts) == 2)
        time.sleep(DELAY_MS / 1000.0 * 2)  # plenty of ticks for a third

    assert len(attempts) == 2


def test_a_trigger_already_low_is_not_re_zeroed(stack) -> None:
    """A PLC that lowers its own trigger leaves nothing to clear — writing 0
    over a 0 would be a pointless transaction every cycle."""
    client, manager, rmap = stack
    worker = PlcPollWorker(manager, poll_interval_ms=10, trigger_clear_delay_ms=DELAY_MS)
    attempts = hold_trigger_high(client, rmap)

    with running(worker):
        assert wait_until(lambda: worker._last_trigger == 0)
        worker.notify_cycle_finished()
        time.sleep(DELAY_MS / 1000.0 * 2)  # both clears come due

    assert attempts == []


def test_a_fresh_trigger_raised_during_the_delay_window_still_fires(stack) -> None:
    """The safety property: the pending clear must never swallow the next
    cycle's trigger. The edge branch runs first on every tick, so the 1 is
    always claimed by a cycle before any post-cycle clear looks at it."""
    client, manager, rmap = stack
    fired: list[int] = []
    worker = PlcPollWorker(manager, poll_interval_ms=10, trigger_clear_delay_ms=DELAY_MS)
    worker.trigger_detected.connect(fired.append, Qt.ConnectionType.DirectConnection)
    client.set_register(rmap.machine_number, 77)

    with running(worker):
        assert wait_until(lambda: worker._last_trigger == 0)
        worker.notify_cycle_finished()          # previous cycle just ended
        client.set_register(rmap.trigger, 1)    # PLC raises the next one
        assert wait_until(lambda: fired == [77])
        time.sleep(DELAY_MS / 1000.0 * 2)       # the delayed clear comes due

    assert fired == [77]  # fired exactly once, and was not lost


def test_a_pending_clear_is_dropped_when_the_link_drops(stack) -> None:
    """Writing 0 after a reconnect would acknowledge a handshake this worker
    never completed — the same rule as a trigger found high at connect."""
    client, manager, rmap = stack
    worker = PlcPollWorker(manager, poll_interval_ms=10, trigger_clear_delay_ms=DELAY_MS)
    worker.notify_cycle_finished()
    assert worker._clear_trigger_at is not None

    worker._discard_pending_trigger_clears()

    assert worker._clear_trigger_now is False
    assert worker._clear_trigger_at is None


def test_notify_accepts_the_finished_cycle_object(stack) -> None:
    """It is connected straight to inspection_finished, which carries the
    cycle, so it has to tolerate the argument."""
    _client, manager, _rmap = stack
    worker = PlcPollWorker(manager, trigger_clear_delay_ms=DELAY_MS)

    worker.notify_cycle_finished(object())

    assert worker._clear_trigger_now is True


def test_re_arming_restarts_the_delay_from_the_latest_cycle(stack) -> None:
    """Back-to-back cycles: the timed clear belongs to the most recent one."""
    _client, manager, _rmap = stack
    worker = PlcPollWorker(manager, trigger_clear_delay_ms=DELAY_MS)

    worker.notify_cycle_finished()
    first = worker._clear_trigger_at
    time.sleep(0.05)
    worker.notify_cycle_finished()

    assert worker._clear_trigger_at > first

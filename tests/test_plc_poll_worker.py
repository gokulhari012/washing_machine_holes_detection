"""Trigger edge detection and acknowledgement in the PLC poll loop.

The worker is exercised without starting its QThread: ``run()`` is driven on a
plain Python thread and stopped through the same stop event, which keeps these
tests free of a Qt event loop while running the real loop body. Signals are
connected ``DirectConnection`` for the same reason — with no event loop
spinning, a queued delivery would never arrive.
"""

import threading
import time
from contextlib import contextmanager

import pytest
from PySide6.QtCore import Qt

from core.plc import PlcManager, RegisterMap, SimulatedPlc
from workers.plc_poll_worker import PlcPollWorker

from tests.test_register_map import make_config


@pytest.fixture()
def stack() -> tuple[SimulatedPlc, PlcManager, RegisterMap]:
    config = make_config()
    config["registers"]["camera_triggers"] = {"1": 132, "2": 133}
    rmap = RegisterMap.from_config(config)
    client = SimulatedPlc(register_map=rmap)
    manager = PlcManager(client, rmap)
    manager.connect()
    return client, manager, rmap


@contextmanager
def running(worker: PlcPollWorker):
    """Drive the real loop body on a plain thread for the duration of the block."""
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        worker._stop_event.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# ------------------------------------------------------------ global trigger
def test_global_trigger_is_cleared_on_detection(stack) -> None:
    client, manager, rmap = stack
    fired: list[int] = []
    worker = PlcPollWorker(manager, poll_interval_ms=10)
    worker.trigger_detected.connect(fired.append, Qt.ConnectionType.DirectConnection)
    client.set_register(rmap.machine_number, 4242)

    with running(worker):
        assert wait_until(lambda: worker._last_trigger == 0)  # baselined
        client.set_register(rmap.trigger, 1)
        assert wait_until(lambda: client.get_register(rmap.trigger) == 0)

    assert fired == [4242]


def test_a_second_trigger_fires_again_after_the_clear(stack) -> None:
    """Baselining on the 0 we wrote is what lets the next 1 read as an edge."""
    client, manager, rmap = stack
    fired: list[int] = []
    worker = PlcPollWorker(manager, poll_interval_ms=10)
    worker.trigger_detected.connect(fired.append, Qt.ConnectionType.DirectConnection)

    with running(worker):
        assert wait_until(lambda: worker._last_trigger == 0)
        for _ in range(3):
            client.set_register(rmap.trigger, 1)
            assert wait_until(lambda: client.get_register(rmap.trigger) == 0)

    assert len(fired) == 3


def test_trigger_high_at_connect_is_baselined_not_cleared(stack) -> None:
    """A trigger already high on the first read is not an edge, so it is not
    acknowledged either — clearing it would fake a handshake that never ran."""
    client, manager, rmap = stack
    fired: list[int] = []
    worker = PlcPollWorker(manager, poll_interval_ms=10)
    worker.trigger_detected.connect(fired.append, Qt.ConnectionType.DirectConnection)
    worker._last_trigger = None  # as after a (re)connect
    client.set_register(rmap.trigger, 1)

    with running(worker):
        assert wait_until(lambda: worker._last_trigger == 1)
        time.sleep(0.05)  # several more ticks, still no edge

    assert fired == []
    assert client.get_register(rmap.trigger) == 1


# ------------------------------------------------------- per-camera triggers
def test_camera_trigger_fires_but_is_not_cleared_on_detection(stack) -> None:
    """Unlike the global trigger, a camera trigger stays high while it works —
    it is released by write_camera_inspection_output at the end of the cycle."""
    client, manager, _rmap = stack
    fired: list[tuple[int, int]] = []
    worker = PlcPollWorker(manager, poll_interval_ms=10)
    worker.camera_trigger_detected.connect(
        lambda i, m: fired.append((i, m)), Qt.ConnectionType.DirectConnection
    )

    worker._poll_camera_triggers()  # baseline both cameras at 0
    client.set_register(132, 1)
    worker._poll_camera_triggers()

    assert fired == [(1, 0)]
    assert client.get_register(132) == 1


def test_camera_trigger_does_not_re_fire_while_it_stays_high(stack) -> None:
    """Holding the trigger through the cycle must not start it over and over."""
    client, manager, _rmap = stack
    fired: list[tuple[int, int]] = []
    worker = PlcPollWorker(manager, poll_interval_ms=10)
    worker.camera_trigger_detected.connect(
        lambda i, m: fired.append((i, m)), Qt.ConnectionType.DirectConnection
    )

    worker._poll_camera_triggers()
    client.set_register(132, 1)
    for _ in range(5):
        worker._poll_camera_triggers()

    assert len(fired) == 1


def test_camera_trigger_fires_again_after_the_cycle_releases_it(stack) -> None:
    """The release reads as a falling edge, so the PLC's next 1 is a fresh one."""
    client, manager, _rmap = stack
    fired: list[tuple[int, int]] = []
    worker = PlcPollWorker(manager, poll_interval_ms=10)
    worker.camera_trigger_detected.connect(
        lambda i, m: fired.append((i, m)), Qt.ConnectionType.DirectConnection
    )

    worker._poll_camera_triggers()
    for _ in range(3):
        client.set_register(132, 1)
        worker._poll_camera_triggers()
        manager.clear_camera_trigger(1)  # what the end of the cycle does
        worker._poll_camera_triggers()

    assert len(fired) == 3


def test_camera_trigger_edge_does_not_touch_the_global_trigger(stack) -> None:
    client, manager, rmap = stack
    worker = PlcPollWorker(manager, poll_interval_ms=10)

    worker._poll_camera_triggers()
    client.set_register(rmap.trigger, 1)  # a global trigger the loop never sees
    client.set_register(132, 1)
    worker._poll_camera_triggers()

    assert client.get_register(rmap.trigger) == 1


def test_camera_trigger_high_at_connect_is_baselined_not_cleared(stack) -> None:
    client, manager, _rmap = stack
    fired: list[tuple[int, int]] = []
    worker = PlcPollWorker(manager, poll_interval_ms=10)
    worker.camera_trigger_detected.connect(
        lambda i, m: fired.append((i, m)), Qt.ConnectionType.DirectConnection
    )

    client.set_register(132, 1)
    worker._poll_camera_triggers()  # first read after connect: baseline

    assert fired == []
    assert client.get_register(132) == 1

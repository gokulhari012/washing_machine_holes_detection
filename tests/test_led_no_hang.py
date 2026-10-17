"""A silent LED controller must never stall the station or freeze the UI.

The LED link is a convenience, not a gate: the lights being wrong is worth a
log line, never a stalled capture, a stalled inspection cycle, or a frozen
window. Four separate mechanisms make that true, and this file pins all of
them, because each one is easy to undo by accident:

1. **Fire-and-forget dispatch.** Every one-way command (brightness pushes,
   strobe on/off) writes its bytes and returns without awaiting the
   controller acknowledgement, so there is no timeout to wait out in the
   first place. Only the raw command tester waits.
2. **A timeout does not drop the link.** With no poll loop on this link,
   tearing it down for one silent reply would leave the lights dead until an
   operator noticed and pressed Connect. A genuinely broken port still drops.
3. **The one waiting command waits off the GUI thread**, on
   ``LedCommandWorker``.
4. **No serial call runs on the caller's thread.** A USB-serial driver that
   wedges a write (or an open, or the port list) costs a caller one short
   deadline, and closing the port - application close included - cancels
   the stuck write instead of waiting on it forever.

That the *callers* carry on regardless is pinned separately, at the level
that matters for each: ``test_inspection_strobe.py`` for the trigger paths
and ``test_camera_service.py`` for Test Camera / Continuous Capture.
"""

import threading
import time

import pytest

from core.led import LedControllerSettings, LedManager, SimulatedLedClient
from core.led.led_client_base import LedClientBase
from core.utilities.enums import ConnectionState
from core.utilities.exceptions import LedConnectionError, LedTimeoutError, LedWriteError
from workers.led_command_worker import LedCommandWorker


class RecordingClient(LedClientBase):
    """Records how each command was dispatched, and can fail on demand."""

    def __init__(self, raises: Exception | None = None) -> None:
        self._connected = False
        self._raises = raises
        self.calls: list[tuple[str, bool]] = []  # (command, expect_response)

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def send(self, command, *, append_terminator=False, expect_response=True) -> str:
        self.calls.append((command, expect_response))
        if self._raises is not None:
            raise self._raises
        return "!" if expect_response else ""


def _manager(client: LedClientBase) -> LedManager:
    manager = LedManager(client, LedControllerSettings())
    manager.connect()
    return manager


# ------------------------------------------------- 1. fire-and-forget dispatch
def test_one_way_commands_do_not_await_a_reply() -> None:
    client = RecordingClient()
    manager = _manager(client)

    manager.send_channel(1, 200)
    manager.send_multichannel([(100, True), (0, False), (0, False), (0, False)])

    assert [expect for _command, expect in client.calls] == [False, False]


def test_the_raw_tester_is_the_one_command_that_waits() -> None:
    client = RecordingClient()
    manager = _manager(client)

    assert manager.send_raw("SA0200#") == "!"
    assert client.calls == [("SA0200#", True)]


def test_a_strobe_is_never_asked_to_wait() -> None:
    """The real adapter cannot time out with ``expect_response`` False, so
    asserting the manager requests that mode is what pins the guarantee."""
    client = RecordingClient(raises=LedTimeoutError("no reply"))
    manager = _manager(client)

    with pytest.raises(LedTimeoutError):
        manager.send_channel(1, 200)
    assert client.calls == [("SA0200#", False)]


# --------------------------------------------- 2. a timeout must not latch off
def test_timeout_leaves_the_link_connected() -> None:
    client = RecordingClient(raises=LedTimeoutError("no reply within 100 ms"))
    manager = _manager(client)

    with pytest.raises(LedTimeoutError):
        manager.send_raw("SA0200#")

    assert manager.state is ConnectionState.CONNECTED
    assert client.connected is True
    assert "no reply" in manager.last_error


def test_a_broken_link_still_drops_to_error() -> None:
    client = RecordingClient(raises=LedWriteError("Write failed: Write timeout"))
    manager = _manager(client)

    with pytest.raises(LedWriteError):
        manager.send_channel(1, 200)

    assert manager.state is ConnectionState.ERROR
    assert client.connected is False


def test_the_link_keeps_working_after_a_timeout() -> None:
    """The point of not latching: the very next command still goes out."""
    client = SimulatedLedClient()
    manager = _manager(client)

    def mute(command, *, append_terminator=False, expect_response=True):
        raise LedTimeoutError("no reply")

    real_send, client.send = client.send, mute
    with pytest.raises(LedTimeoutError):
        manager.send_raw("SA0200#")
    client.send = real_send

    manager.send_channel(1, 200)
    assert client.sent == ["SA0200#"]


# ------------------------------------------- serial adapter skips the read loop
class FakeSerial:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.is_open = True
        self.written = bytearray()
        self.in_waiting_reads = 0

    def close(self) -> None:
        self.is_open = False

    def reset_input_buffer(self) -> None:
        pass

    def write(self, data) -> int:
        self.written += data
        return len(data)

    def flush(self) -> None:
        pass

    @property
    def in_waiting(self) -> int:
        self.in_waiting_reads += 1
        return 0

    def read(self, size: int) -> bytes:
        return b""


def _fake_serial_client(monkeypatch, timeout_s=0.1):
    import core.led.serial_led_client as mod

    created: list[FakeSerial] = []

    def factory(**kwargs):
        fake = FakeSerial(**kwargs)
        created.append(fake)
        return fake

    monkeypatch.setattr(mod.serial, "Serial", factory)
    client = mod.SerialLedClient("COM_TEST", 19200, timeout_s)
    client.connect()
    return client, created


def test_serial_adapter_never_polls_for_a_reply_it_was_not_promised(monkeypatch) -> None:
    client, created = _fake_serial_client(monkeypatch)

    assert client.send("SA0200#", expect_response=False) == ""

    assert created[0].written == b"SA0200#"
    # The read loop is what costs the timeout; it must not have run at all.
    assert created[0].in_waiting_reads == 0


def test_serial_adapter_still_times_out_when_a_reply_was_expected(monkeypatch) -> None:
    client, _created = _fake_serial_client(monkeypatch)

    with pytest.raises(LedTimeoutError):
        client.send("SA0200#", expect_response=True)


def test_write_timeout_never_follows_a_tight_response_timeout(monkeypatch) -> None:
    """A station tuning the ack wait down must not also shorten the window the
    USB-serial driver gets to flush bytes -- the bug that produced
    'Write failed: Write timeout' on an otherwise healthy link."""
    _client, created = _fake_serial_client(monkeypatch, timeout_s=0.1)

    assert created[0].kwargs["timeout"] == 0.1
    assert created[0].kwargs["write_timeout"] == 1.0


def test_a_wedged_output_buffer_fails_the_write_instead_of_hanging(monkeypatch) -> None:
    """pyserial's Windows ``flush()`` loops on ``out_waiting`` with no
    deadline; a bridge that never drains must surface as a write failure
    within the write timeout, not hold the inspection thread forever."""
    import core.led.serial_led_client as mod

    monkeypatch.setattr(mod, "_MIN_WRITE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(FakeSerial, "out_waiting", property(lambda self: 8), raising=False)
    client, _created = _fake_serial_client(monkeypatch, timeout_s=0.05)

    with pytest.raises(LedWriteError):
        client.send("SA0200#", expect_response=False)


# ----------------------------------------------- 3. the waiting call is off-GUI
def test_worker_reports_a_response_without_raising() -> None:
    seen: list[str] = []
    worker = LedCommandWorker(lambda cmd, **kw: "!", "SA0200#", append_terminator=False)
    worker.succeeded.connect(seen.append)
    worker.run()
    assert seen == ["!"]


def test_worker_turns_a_timeout_into_a_signal_not_an_exception() -> None:
    def mute(command, **kwargs):
        raise LedTimeoutError("no reply within 100 ms")

    failures: list[str] = []
    worker = LedCommandWorker(mute, "SA0200#", append_terminator=False)
    worker.failed.connect(failures.append)
    worker.run()  # must not raise -- a frozen or crashed page helps nobody
    assert failures and "no reply" in failures[0]


# ------------------------------------- 4. a wedged serial driver hangs nobody
class WedgedSerial(FakeSerial):
    """A port whose driver never completes a write until it is cancelled -
    what a surprise-removed USB-RS232 bridge can do to pyserial's Windows
    ``write()``, whatever its ``write_timeout`` says."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.release = threading.Event()
        self.wedged = True
        self.cancelled = False

    def write(self, data) -> int:
        if self.wedged:
            self.release.wait(10)
        return super().write(data)

    def cancel_write(self) -> None:
        self.cancelled = True
        self.release.set()


def _wedged_client(monkeypatch):
    """A connected client plus a getter for the port it currently has open
    (every connect opens a fresh one, as a real reconnect would)."""
    import core.led.serial_led_client as mod

    monkeypatch.setattr(mod, "SEND_DEADLINE_S", 0.1)
    ports: list[WedgedSerial] = []

    def factory(**kwargs):
        ports.append(WedgedSerial(**kwargs))
        return ports[-1]

    monkeypatch.setattr(mod.serial, "Serial", factory)
    client = mod.SerialLedClient("COM_TEST", 19200, 0.1)
    client.connect()
    return client, lambda: ports[-1]


def _timed(fn) -> tuple[float, BaseException | None]:
    started = time.monotonic()
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - the test inspects it
        return time.monotonic() - started, exc
    return time.monotonic() - started, None


def _eventually(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_wedged_write_costs_the_caller_one_short_deadline(monkeypatch) -> None:
    """The GUI thread (Test Camera, a brightness push) used to sit in the
    driver's write itself; now it waits one short deadline, and a command
    behind the stuck one is not waited on at all - but still goes out, in
    order, once the driver recovers."""
    client, current_port = _wedged_client(monkeypatch)
    port = current_port()

    took, exc = _timed(lambda: client.send("SA0200#", expect_response=False))
    assert isinstance(exc, LedTimeoutError)
    assert took < 0.5

    time.sleep(0.05)  # comfortably past the deadline: the port is now "stuck"
    took, exc = _timed(lambda: client.send("SA0000#", expect_response=False))
    assert isinstance(exc, LedTimeoutError)
    assert took < 0.05

    port.wedged = False
    port.release.set()
    assert _eventually(lambda: port.written == b"SA0200#SA0000#")
    client.disconnect()


def test_close_cancels_a_wedged_write_instead_of_hanging(monkeypatch) -> None:
    """Application close used to wait on the port lock with no deadline, so a
    write stuck in the driver froze shutdown for good."""
    client, current_port = _wedged_client(monkeypatch)
    port = current_port()
    with pytest.raises(LedTimeoutError):
        client.send("SA0200#", expect_response=False)

    took, exc = _timed(client.disconnect)

    assert exc is None
    assert took < 1.5
    assert port.cancelled  # the stuck write was cancelled ...
    assert not port.is_open  # ... so the queued close could run
    assert not client.connected


def test_close_and_reconnect_stay_bounded_when_cancel_does_nothing(monkeypatch) -> None:
    """Some drivers ignore the cancel too. Close then gives up waiting rather
    than hanging, and Connect refuses at once - Windows will not open a COM
    port whose old handle is still busy."""
    client, current_port = _wedged_client(monkeypatch)
    port = current_port()
    monkeypatch.setattr(port, "cancel_write", lambda: None)
    with pytest.raises(LedTimeoutError):
        client.send("SA0200#", expect_response=False)

    took, exc = _timed(client.disconnect)
    assert exc is None
    assert took < 1.5
    assert not client.connected

    took, exc = _timed(client.connect)
    assert isinstance(exc, LedConnectionError)
    assert took < 0.5

    port.release.set()


def test_a_port_that_never_opens_fails_connect_in_bounded_time(monkeypatch) -> None:
    """Startup, the LED page's Connect and a settings Save all connect on the
    GUI thread. A port that opens only after the caller gave up is closed
    again: a leaked handle would lock the COM port until the app exits."""
    import core.led.serial_led_client as mod

    monkeypatch.setattr(mod, "CONNECT_DEADLINE_S", 0.1)
    release = threading.Event()
    opened: list[FakeSerial] = []

    def slow_open(**kwargs):
        release.wait(10)
        opened.append(FakeSerial(**kwargs))
        return opened[-1]

    monkeypatch.setattr(mod.serial, "Serial", slow_open)
    client = mod.SerialLedClient("COM_TEST", 19200, 0.1)

    took, exc = _timed(client.connect)
    assert isinstance(exc, LedConnectionError)
    assert took < 0.5

    release.set()
    assert _eventually(lambda: bool(opened) and not opened[0].is_open)
    assert not client.connected


def test_a_failed_write_nobody_waited_for_still_drops_the_link(monkeypatch) -> None:
    """The caller stopped waiting on a stuck write. When the driver finally
    fails it, the port is closed anyway, so the *next* command reports the
    broken link and LedManager goes to ERROR instead of claiming CONNECTED."""
    client, current_port = _wedged_client(monkeypatch)
    manager = _manager(client)
    port = current_port()

    def fails_once_released(data):
        port.release.wait(10)
        raise OSError("device removed")

    monkeypatch.setattr(port, "write", fails_once_released)
    with pytest.raises(LedTimeoutError):
        manager.send_channel(1, 200)
    assert manager.state is ConnectionState.CONNECTED  # slow is not yet broken

    port.release.set()
    assert _eventually(lambda: not client.connected)
    with pytest.raises(LedConnectionError):
        manager.send_channel(1, 0)
    assert manager.state is ConnectionState.ERROR


def test_a_port_that_dies_mid_reply_drops_the_link(monkeypatch) -> None:
    """``in_waiting`` failing used to escape as a raw SerialException and
    leave the link reported as connected."""
    client, _created = _fake_serial_client(monkeypatch)
    manager = _manager(client)

    def dead(_self):
        raise OSError("ClearCommError failed")

    monkeypatch.setattr(FakeSerial, "in_waiting", property(dead))

    with pytest.raises(LedConnectionError):
        manager.send_raw("SA0200#")
    assert manager.state is ConnectionState.ERROR


def test_disconnect_ends_a_reply_wait_at_once(monkeypatch) -> None:
    """Closing the app mid raw-command must not sit out the reply timeout."""
    client, _created = _fake_serial_client(monkeypatch, timeout_s=5.0)
    outcome: list[BaseException | None] = []
    sender = threading.Thread(
        target=lambda: outcome.append(_timed(lambda: client.send("SA0200#"))[1])
    )
    sender.start()
    time.sleep(0.1)

    took, exc = _timed(client.disconnect)

    assert exc is None
    assert took < 0.3
    sender.join(1)
    assert not sender.is_alive()
    assert isinstance(outcome[0], LedConnectionError)


def test_listing_ports_never_hangs_the_page(monkeypatch) -> None:
    """The LED page lists COM ports while it is built, at startup."""
    import serial.tools.list_ports as list_ports

    import core.led.serial_led_client as mod

    release = threading.Event()
    monkeypatch.setattr(list_ports, "comports", lambda: release.wait(10) and [])

    took, exc = _timed(lambda: mod.list_serial_ports(timeout_s=0.1))

    assert exc is None
    assert took < 0.5
    release.set()

"""A silent LED controller must never stall the station or freeze the UI.

The LED link is a convenience, not a gate: the lights being wrong is worth a
log line, never a stalled capture, a stalled inspection cycle, or a frozen
window. Three separate mechanisms make that true, and this file pins all of
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

That the *callers* carry on regardless is pinned separately, at the level
that matters for each: ``test_inspection_strobe.py`` for the trigger paths
and ``test_camera_service.py`` for Test Camera / Continuous Capture.
"""

import pytest

from core.led import LedControllerSettings, LedManager, SimulatedLedClient
from core.led.led_client_base import LedClientBase
from core.utilities.enums import ConnectionState
from core.utilities.exceptions import LedTimeoutError, LedWriteError
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

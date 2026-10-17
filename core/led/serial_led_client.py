"""RS232 adapter for the KDC-24V60W-4T LED controller, built on pyserial.

Fixed serial parameters per the controller's documented RS232 settings:

    bytesize = 8, parity = NONE, stopbits = 1, flow control = None

Two independent timeouts, deliberately not the same number. ``timeout_ms``
from ``led.json`` is the *response* timeout - how long to wait for the
controller's documented "!" acknowledgement - and tuning it down to a
hundred milliseconds or so is a reasonable thing for a station to do, since
the reply comes back in one frame. The *write* timeout is a different
quantity: it bounds how long the driver may take to report the outgoing
bytes as transferred, which on a USB-RS232 bridge is dominated by the
adapter's latency timer (~16 ms) and USB frame scheduling rather than by
wire time (an 8-byte command at 19200 baud is ~4 ms of line time). Under
load - a full-resolution GigE grab on the same box - that completion report
can take far longer than a tight response timeout allows, surfacing as
``SerialTimeoutException('Write timeout')`` even though the link is
healthy. So the write timeout takes ``_MIN_WRITE_TIMEOUT_S`` as a floor and
never follows ``timeout_ms`` below it.

**No serial call ever runs on the caller's thread.** Callers are the GUI
thread (Camera page Test Camera, brightness pushes, the LED page, startup,
application close) and the inspection pipeline, and none of them may hang on
this port. pyserial's Windows ``write`` waits on the driver with
``GetOverlappedResult(..., bWait=True)``, so a USB-RS232 bridge whose driver
wedges - surprise removal, an EMI reset on some clone chipsets - can hold it
indefinitely despite ``write_timeout``; opening or closing the port can
stall the same way. So every port operation (open, write, drain, read,
close) runs in order on this client's own ``led-io``
:class:`~core.led.command_thread.CommandThread`, and the caller only ever
waits a bounded time:

- a one-way command (brightness push, strobe) waits at most
  :data:`SEND_DEADLINE_S` - a healthy write takes a few ms. Past that the
  command finishes in the background and the caller gets ``LedTimeoutError``,
  which leaves the link up, exactly as a missing reply does;
- a command arriving while an earlier one is already past that deadline is
  queued behind it, so on/off order survives, without being waited on at all
  (a wedged port costs one deadline, not one per command); past
  :data:`_MAX_BACKLOG` it is dropped;
- the raw tester's request/response waits for the write, the reply timeout
  and :data:`_RESPONSE_MARGIN_S` - on ``LedCommandWorker``, never the GUI;
- ``connect`` waits at most :data:`CONNECT_DEADLINE_S` for the port to open
  (a port that opens later is closed again rather than leaked, which would
  lock the COM port until exit), and refuses at once while a command is
  still stuck, since Windows will not open a COM port twice;
- ``disconnect`` waits at most ``_CLOSE_GRACE_S + _CLOSE_AFTER_CANCEL_S``:
  the close is queued behind whatever is in flight, and if that is a write
  the driver never completed, the pending I/O is cancelled (pyserial allows
  that from another thread) so the queued close can run. If even that does
  not land, the port is left to close in the background. The port is never
  closed from a second thread while the I/O thread may still be using it.

A write or read that *fails* closes the port from the I/O thread itself, so
the next command reports "Not connected" and ``LedManager`` drops the link to
ERROR - even when the caller had already stopped waiting for the failed one.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from concurrent.futures import wait as wait_futures

import serial

from core.led.command_thread import CommandThread
from core.led.led_client_base import LedClientBase
from core.led.protocol import ACK, DEFAULT_BAUD_RATE
from core.logging import get_logger
from core.utilities.enums import LogSource
from core.utilities.exceptions import LedConnectionError, LedTimeoutError, LedWriteError

logger = get_logger(LogSource.LED)

_ACK_BYTES = ACK.encode("ascii")
_POLL_INTERVAL_S = 0.01

#: Floor for the serial write timeout (see the module docstring). A station
#: may tune ``timeout_ms`` down for a snappier ack wait; that must not also
#: shorten the window the USB-serial driver gets to flush bytes.
_MIN_WRITE_TIMEOUT_S = 1.0

#: Longest a caller waits for a one-way command to reach the wire.
SEND_DEADLINE_S = 0.5
#: Longest a caller waits for the port to open.
CONNECT_DEADLINE_S = 2.0
#: Added to write + reply timeouts for the raw tester's own deadline.
_RESPONSE_MARGIN_S = 0.5
#: Wait for an orderly close, then the further wait after cancelling stuck I/O.
_CLOSE_GRACE_S = 0.2
_CLOSE_AFTER_CANCEL_S = 0.8
#: Commands allowed to queue behind a stuck one before new ones are dropped.
_MAX_BACKLOG = 8
#: Longest the LED page waits for the COM port list.
_LIST_PORTS_TIMEOUT_S = 2.0


class SerialLedClient(LedClientBase):
    """RS232 implementation of :class:`LedClientBase` whose callers never
    block on the port for longer than a fixed deadline."""

    def __init__(
        self,
        port: str,
        baud_rate: int = DEFAULT_BAUD_RATE,
        timeout_s: float = 1.0,
    ) -> None:
        self._port_name = port
        self._baud_rate = baud_rate
        self._timeout_s = timeout_s
        self._write_timeout_s = max(timeout_s, _MIN_WRITE_TIMEOUT_S)
        self._serial: serial.Serial | None = None
        # Guards the ``_serial`` reference only - never held across I/O, so
        # nothing can queue behind a stuck port operation on it.
        self._lock = threading.Lock()
        self._io = CommandThread("led-io", max_backlog=_MAX_BACKLOG)

    # ------------------------------------------------------------ connection
    @property
    def connected(self) -> bool:
        with self._lock:
            return self._serial is not None and self._serial.is_open

    def connect(self) -> None:
        self.disconnect()
        if self._io.running_for() > SEND_DEADLINE_S:
            raise LedConnectionError(
                f"Cannot open {self._port_name}: the serial driver has not finished "
                "an earlier command"
            )
        future = self._io.submit(self._open, force=True)
        if not _settled(future, CONNECT_DEADLINE_S):
            future.add_done_callback(_close_if_opened)
            raise LedConnectionError(
                f"Cannot open {self._port_name}: the serial driver did not respond "
                f"within {CONNECT_DEADLINE_S * 1000:.0f} ms"
            )
        try:
            ser = future.result()
        except Exception as exc:
            raise LedConnectionError(f"Cannot open {self._port_name}: {exc}") from exc
        with self._lock:
            self._serial = ser
        logger.info(
            "Connected to LED controller on %s at %d baud", self._port_name, self._baud_rate
        )

    def disconnect(self) -> None:
        with self._lock:
            ser, self._serial = self._serial, None
        if ser is None:
            return
        closed = self._io.submit(lambda: _close_quietly(ser), force=True)
        if _settled(closed, _CLOSE_GRACE_S):
            return
        _cancel_pending_io(ser)
        if _settled(closed, _CLOSE_AFTER_CANCEL_S):
            return
        logger.warning(
            "LED controller port %s did not close within %.0f ms - the serial driver "
            "is not responding; it will close in the background if the driver recovers",
            self._port_name, (_CLOSE_GRACE_S + _CLOSE_AFTER_CANCEL_S) * 1000,
        )

    # ------------------------------------------------------------------- I/O
    def send(
        self,
        command: str,
        *,
        append_terminator: bool = False,
        expect_response: bool = True,
    ) -> str:
        data = command.encode("ascii", errors="replace")
        if append_terminator:
            data += b"\r\n"

        def transact() -> str:
            return self._transact(data, expect_response)

        if self._io.running_for() > SEND_DEADLINE_S:
            # An earlier command is stuck in the driver. A one-way command
            # still goes in line behind it - a late light-on must still be
            # followed by its light-off - but nobody waits for it. A request
            # for a reply is pointless to queue: nobody would read it.
            if not expect_response and self._io.submit(transact) is not None:
                raise LedTimeoutError(
                    "LED controller has not finished an earlier command; "
                    "this one is queued behind it"
                )
            raise LedTimeoutError(
                "LED controller has not finished an earlier command; command not sent"
            )

        future = self._io.submit(transact)
        if future is None:
            raise LedTimeoutError(
                f"LED command dropped: {_MAX_BACKLOG} commands already waiting on the port"
            )
        deadline = (
            self._write_timeout_s + self._timeout_s + _RESPONSE_MARGIN_S
            if expect_response
            else SEND_DEADLINE_S
        )
        if not _settled(future, deadline):
            raise LedTimeoutError(
                f"LED command not completed within {deadline * 1000:.0f} ms; "
                "it will finish in the background"
            )
        response = future.result()
        if expect_response and not response:
            raise LedTimeoutError(
                f"No response from LED controller within {self._timeout_s * 1000:.0f} ms"
            )
        return response

    # ---------------------------------------------- runs on the led-io thread
    def _open(self) -> serial.Serial:
        return serial.Serial(
            port=self._port_name,
            baudrate=self._baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self._timeout_s,
            write_timeout=self._write_timeout_s,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

    def _transact(self, data: bytes, expect_response: bool) -> str:
        with self._lock:
            ser = self._serial
        if ser is None or not ser.is_open:
            raise LedConnectionError("Not connected to LED controller")
        try:
            ser.reset_input_buffer()
            ser.write(data)
            self._drain(ser)
        except Exception as exc:
            self._drop(ser)
            raise LedWriteError(f"Write failed: {exc}") from exc

        if not expect_response:
            # Fire-and-forget (see LedClientBase.send): the bytes are on the
            # wire, which is all a strobe or a brightness push needs. Any
            # reply the controller does send is discarded by the
            # reset_input_buffer() of the next command.
            return ""
        try:
            return self._read_response(ser)
        except Exception as exc:
            self._drop(ser)
            raise LedConnectionError(f"Read failed: {exc}") from exc

    def _drain(self, ser: serial.Serial) -> None:
        """Wait for the output buffer to empty, bounded by the write timeout.

        Replaces ``ser.flush()``, which on Windows is an *unbounded*
        ``while out_waiting: sleep`` loop: a USB-RS232 bridge that has
        wedged (controller powered off, cable pulled mid-cycle) would hold
        the I/O thread forever. Here it raises instead, which ``send``
        reports as a write failure - and stops at once on a disconnect."""
        deadline = time.monotonic() + self._write_timeout_s
        while getattr(ser, "out_waiting", 0):
            if not self._is_current(ser):
                raise LedConnectionError("disconnected while sending")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"output not drained within {self._write_timeout_s * 1000:.0f} ms"
                )
            time.sleep(_POLL_INTERVAL_S)

    def _read_response(self, ser: serial.Serial) -> str:
        """Read whatever the controller sends back within the configured
        timeout. The manual only documents "!" as the success
        acknowledgement, so this reads until the ack byte is seen or the
        timeout elapses, returning exactly what was received (never
        fabricated). A disconnect ends the wait at once, so closing the
        port never queues behind a long reply timeout."""
        buf = bytearray()
        deadline = time.monotonic() + self._timeout_s
        while time.monotonic() < deadline:
            if not self._is_current(ser):
                raise LedConnectionError("disconnected while waiting for a reply")
            waiting = ser.in_waiting
            if waiting:
                buf += ser.read(waiting)
                if _ACK_BYTES in buf:
                    break
            else:
                time.sleep(_POLL_INTERVAL_S)
        return buf.decode("ascii", errors="replace")

    def _is_current(self, ser: serial.Serial) -> bool:
        with self._lock:
            return self._serial is ser

    def _drop(self, ser: serial.Serial) -> None:
        """A failed write or read means the port is gone: close it here, on
        the I/O thread, so the next command reports "Not connected" even if
        the caller of this one had already stopped waiting."""
        with self._lock:
            if self._serial is ser:
                self._serial = None
        _close_quietly(ser)


def _settled(future: Future, timeout_s: float) -> bool:
    """True once *future* has finished, either way, within *timeout_s*."""
    done, _pending = wait_futures([future], timeout=timeout_s)
    return bool(done)


def _close_quietly(ser: serial.Serial) -> None:
    try:
        ser.close()
    except Exception:  # closing a dead port must never raise
        pass


def _close_if_opened(future: Future) -> None:
    """Done-callback for an open the caller gave up on: close the port it
    eventually produced instead of leaking the handle."""
    if not future.cancelled() and future.exception() is None:
        _close_quietly(future.result())


def _cancel_pending_io(ser: serial.Serial) -> None:
    """Abort a write/read the driver never completed. pyserial documents
    ``cancel_write``/``cancel_read`` as callable from another thread; the
    stuck call then returns on the I/O thread, which can run the close."""
    for name in ("cancel_write", "cancel_read"):
        cancel = getattr(ser, name, None)
        if cancel is None:
            continue
        try:
            cancel()
        except Exception:
            pass


def list_serial_ports(timeout_s: float = _LIST_PORTS_TIMEOUT_S) -> list[str]:
    """Available COM port device names, for the LED Controller page's picker.

    Enumerated on a throwaway daemon thread and bounded by *timeout_s*: the
    page builds this list at startup, and a misbehaving USB-serial driver
    must not be able to hold the window up while it does."""
    found: list[str] = []
    done = threading.Event()

    def enumerate_ports() -> None:
        try:
            from serial.tools import list_ports

            found.extend(p.device for p in list_ports.comports())
        except Exception:
            logger.exception("Could not list serial ports")
        finally:
            done.set()

    threading.Thread(target=enumerate_ports, name="led-list-ports", daemon=True).start()
    if not done.wait(timeout_s):
        logger.warning(
            "Serial port list not available within %.0f ms - the USB-serial driver "
            "is not responding", timeout_s * 1000,
        )
        return []
    return list(found)

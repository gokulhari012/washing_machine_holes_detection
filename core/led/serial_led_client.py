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

Thread ownership: every call site in this application (``LedService``, the
LED Controller page) invokes this client directly from the GUI thread, the
same way the PLC page's manual register write and ``PlcService.test_connection``
briefly block on the GUI thread rather than through a dedicated worker - a
one-shot RS232 round trip bounded by ``timeout_ms`` (default 1000 ms) is the
same order of magnitude as those. An internal lock keeps the adapter safe if
that ever changes.
"""

from __future__ import annotations

import threading
import time

import serial

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


class SerialLedClient(LedClientBase):
    """Synchronous RS232 implementation of :class:`LedClientBase`."""

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
        self._lock = threading.RLock()

    # ------------------------------------------------------------ connection
    @property
    def connected(self) -> bool:
        with self._lock:
            return self._serial is not None and self._serial.is_open

    def connect(self) -> None:
        with self._lock:
            self.disconnect()
            try:
                self._serial = serial.Serial(
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
            except Exception as exc:
                self._serial = None
                raise LedConnectionError(
                    f"Cannot open {self._port_name}: {exc}"
                ) from exc
        logger.info(
            "Connected to LED controller on %s at %d baud", self._port_name, self._baud_rate
        )

    def disconnect(self) -> None:
        with self._lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:  # closing a dead port must never raise
                    pass
                self._serial = None

    # ------------------------------------------------------------------- I/O
    def send(
        self,
        command: str,
        *,
        append_terminator: bool = False,
        expect_response: bool = True,
    ) -> str:
        with self._lock:
            ser = self._require_serial()
            data = command.encode("ascii", errors="replace")
            if append_terminator:
                data += b"\r\n"
            try:
                ser.reset_input_buffer()
                ser.write(data)
                ser.flush()
            except Exception as exc:
                raise LedWriteError(f"Write failed: {exc}") from exc

            if not expect_response:
                # Fire-and-forget (see LedClientBase.send): the bytes are on
                # the wire, which is all a strobe or a brightness push needs.
                # Any reply the controller does send is discarded by the
                # reset_input_buffer() of the next command.
                return ""

            response = self._read_response(ser)

        if not response:
            raise LedTimeoutError(
                f"No response from LED controller within {self._timeout_s * 1000:.0f} ms"
            )
        return response

    def _read_response(self, ser: serial.Serial) -> str:
        """Read whatever the controller sends back within the configured
        timeout. The manual only documents "!" as the success
        acknowledgement, so this reads until the ack byte is seen or the
        timeout elapses, returning exactly what was received (never
        fabricated)."""
        buf = bytearray()
        deadline = time.monotonic() + self._timeout_s
        while time.monotonic() < deadline:
            waiting = ser.in_waiting
            if waiting:
                buf += ser.read(waiting)
                if _ACK_BYTES in buf:
                    break
            else:
                time.sleep(_POLL_INTERVAL_S)
        return buf.decode("ascii", errors="replace")

    # -------------------------------------------------------------- internal
    def _require_serial(self) -> serial.Serial:
        if self._serial is None or not self._serial.is_open:
            raise LedConnectionError("Not connected to LED controller")
        return self._serial


def list_serial_ports() -> list[str]:
    """Available COM port device names, for the LED Controller page's picker."""
    from serial.tools import list_ports

    return [p.device for p in list_ports.comports()]

"""SLMP adapter (MELSEC Communication protocol, 3E frame, binary) built on sockets.

Talks straight to a MELSEC iQ-R / Q CPU's built-in Ethernet port -- no vendor
library and no extra rack module, unlike Modbus TCP which those CPUs do not
speak natively.

PLC-side prerequisites (GX Works3):

- Module Parameter -> Ethernet Port -> External Device Configuration -> add an
  "SLMP Connection Module", protocol TCP, port matching ``connection.port``.
- Communication data code must be *binary* (ASCII is rejected with 0xC050).
- Writing while the CPU is in RUN needs the RUN-time write permission, else
  every write comes back as end code 0x0055.

Register model: the :class:`PlcClientBase` contract addresses registers by
plain integer, so this adapter maps them onto data registers -- address 104
is ``D104``, exactly like the Modbus adapter's holding register 104.

Frame variant: iQ-R encodes a 4-byte device number + 2-byte device code
(subcommand 0x0002); Q/L uses 3-byte + 1-byte (subcommand 0x0000). Chosen by
``connection.slmp_frame`` -- ``"iq_r"`` (default) or ``"q"``.

Thread ownership: as with the Modbus adapter all traffic normally flows
through the single PLC worker thread; the internal lock keeps a stray call
from another thread from interleaving two frames on one socket.
"""

from __future__ import annotations

import socket
import threading

from core.logging import get_logger
from core.plc.plc_client_base import PlcClientBase
from core.utilities.enums import LogSource
from core.utilities.exceptions import (
    PlcConnectionError,
    PlcError,
    PlcReadError,
    PlcTimeoutError,
    PlcWriteError,
)

logger = get_logger(LogSource.PLC)

UINT16_MAX = 65535
MAX_POINTS = 960  # 3E batch word-access limit

# --- 3E frame constants (MELSEC Communication Protocol Reference Manual) ---
REQUEST_SUBHEADER = b"\x50\x00"
RESPONSE_SUBHEADER = b"\xd0\x00"
RESPONSE_HEADER_LEN = 9  # subheader .. response data length
NETWORK_NO = 0x00  # own network
PC_NO = 0xFF  # own station
IO_NUMBER = 0x03FF  # own CPU
MULTIDROP_NO = 0x00

CMD_BATCH_READ_WORDS = 0x0401
CMD_BATCH_WRITE_WORDS = 0x1401
DEVICE_CODE_D = 0xA8  # data register: 0xA8 as one byte, 0x00A8 as two

# Only the end codes worth acting on; anything else is reported as raw hex and
# has to be looked up in the MC protocol manual.
END_CODE_HINTS = {
    0x0055: (
        "writing during RUN is disabled on the CPU "
        "(enable the RUN-time write permission in GX Works3)"
    ),
    0xC050: "the connection is set to ASCII; this adapter speaks binary",
    0xC059: (
        "command/subcommand not supported by this CPU "
        "(try the other connection.slmp_frame variant)"
    ),
}


class SlmpPlcClient(PlcClientBase):
    """Synchronous SLMP/MC-3E implementation of :class:`PlcClientBase`."""

    def __init__(
        self,
        host: str,
        port: int = 5007,
        timeout_s: float = 1.0,
        frame: str = "iq_r",
        device_code: int = DEVICE_CODE_D,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._iq_r = str(frame).lower().replace("-", "_") in ("iq_r", "iqr", "r")
        self._device_code = device_code
        self._sock: socket.socket | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------ connection
    @property
    def connected(self) -> bool:
        with self._lock:
            return self._sock is not None

    def connect(self) -> None:
        with self._lock:
            self.disconnect()
            try:
                self._sock = socket.create_connection(
                    (self._host, self._port), timeout=self._timeout_s
                )
            except OSError as exc:
                self._sock = None
                raise PlcConnectionError(
                    f"Cannot connect to PLC {self._host}:{self._port}: {exc}"
                ) from exc
        logger.info(
            "Connected to PLC %s:%d (SLMP 3E binary, %s frame)",
            self._host,
            self._port,
            "iQ-R" if self._iq_r else "Q/L",
        )

    def disconnect(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:  # closing a dead socket must never raise
                    pass
                self._sock = None

    # ------------------------------------------------------------------- I/O
    def read_registers(self, address: int, count: int = 1) -> list[int]:
        if not 1 <= count <= MAX_POINTS:
            raise PlcReadError(f"Point count {count} outside 1..{MAX_POINTS} for D{address}")

        payload = self._device_spec(address) + count.to_bytes(2, "little")
        with self._lock:
            data = self._transact(CMD_BATCH_READ_WORDS, payload, address, PlcReadError)
        if len(data) < count * 2:
            raise PlcReadError(
                f"Short reply for D{address}: wanted {count * 2} data bytes, got {len(data)}"
            )
        return [int.from_bytes(data[i : i + 2], "little") for i in range(0, count * 2, 2)]

    def write_register(self, address: int, value: int) -> None:
        self.write_registers(address, [value])

    def write_registers(self, address: int, values: list[int]) -> None:
        points = [int(v) for v in values]
        if not 1 <= len(points) <= MAX_POINTS:
            raise PlcWriteError(
                f"Point count {len(points)} outside 1..{MAX_POINTS} for D{address}"
            )
        for value in points:
            if not 0 <= value <= UINT16_MAX:
                raise PlcWriteError(f"Value {value} out of uint16 range for D{address}")

        payload = (
            self._device_spec(address)
            + len(points).to_bytes(2, "little")
            + b"".join(v.to_bytes(2, "little") for v in points)
        )
        with self._lock:
            self._transact(CMD_BATCH_WRITE_WORDS, payload, address, PlcWriteError)

    # ----------------------------------------------------------- frame codec
    def _device_spec(self, address: int) -> bytes:
        """Encode the head device as the frame's device-number + device-code pair."""
        if self._iq_r:
            return address.to_bytes(4, "little") + self._device_code.to_bytes(2, "little")
        return address.to_bytes(3, "little") + self._device_code.to_bytes(1, "little")

    def _monitoring_timer(self) -> int:
        """CPU-side wait, in 250 ms units, clamped to the 16-bit field."""
        return max(1, min(0xFFFF, round(self._timeout_s / 0.25)))

    def _build_frame(self, command: int, payload: bytes) -> bytes:
        body = (
            self._monitoring_timer().to_bytes(2, "little")
            + command.to_bytes(2, "little")
            + (0x0002 if self._iq_r else 0x0000).to_bytes(2, "little")
            + payload
        )
        header = (
            REQUEST_SUBHEADER
            + bytes([NETWORK_NO, PC_NO])
            + IO_NUMBER.to_bytes(2, "little")
            + bytes([MULTIDROP_NO])
            + len(body).to_bytes(2, "little")  # counts from the monitoring timer on
        )
        return header + body

    # -------------------------------------------------------------- internal
    def _require_socket(self) -> socket.socket:
        if self._sock is None:
            raise PlcConnectionError("Not connected to PLC")
        return self._sock

    def _transact(
        self,
        command: int,
        payload: bytes,
        address: int,
        error_cls: type[PlcError],
    ) -> bytes:
        """Send one request; return the response data that follows the end code.

        Any transport-level failure drops the socket: a half-read reply would
        desynchronise every frame after it, so the link is rebuilt rather than
        reused. A clean end-code rejection leaves the link up.
        """
        sock = self._require_socket()
        try:
            sock.sendall(self._build_frame(command, payload))
            header = self._recv_exact(sock, RESPONSE_HEADER_LEN)
            if header[:2] != RESPONSE_SUBHEADER:
                raise PlcConnectionError(
                    f"Unexpected SLMP response subheader {header[:2].hex()} for D{address}"
                )
            body = self._recv_exact(sock, int.from_bytes(header[7:9], "little"))
        except socket.timeout as exc:
            self.disconnect()
            raise PlcTimeoutError(f"Timeout on D{address}: {exc}") from exc
        except PlcError:
            self.disconnect()
            raise
        except OSError as exc:
            self.disconnect()
            raise PlcConnectionError(f"Connection lost on D{address}: {exc}") from exc

        end_code = int.from_bytes(body[:2], "little")
        if end_code != 0x0000:
            hint = END_CODE_HINTS.get(end_code)
            detail = f" -- {hint}" if hint else ""
            raise error_cls(f"PLC returned end code 0x{end_code:04X} for D{address}{detail}")
        return body[2:]

    @staticmethod
    def _recv_exact(sock: socket.socket, count: int) -> bytes:
        chunks: list[bytes] = []
        remaining = count
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise PlcConnectionError("PLC closed the connection mid-reply")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

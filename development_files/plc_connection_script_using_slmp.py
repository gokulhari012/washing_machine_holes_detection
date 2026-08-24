"""Standalone SLMP check: write 300 to D104, then read D120.

Speaks MC protocol 3E (binary) straight to the R08 CPU's built-in Ethernet
port over a plain TCP socket -- stdlib only, no pymodbus / pymcprotocol.

The CPU must expose an SLMP connection first, in GX Works3:
    Module Parameter -> Ethernet Port -> External Device Configuration
    -> add "SLMP Connection Module", protocol TCP, host station port 5007.

Run directly:

    python development_files/plc_connection_script_using_slmp.py
"""

from __future__ import annotations

import socket

PLC_IP = "192.168.3.20"
PLC_PORT = 5007  # whatever port the SLMP connection was given in GX Works3
# PLC_PORT = 7920  # whatever port the SLMP connection was given in GX Works3
TIMEOUT_S = 3.0

WRITE_ADDRESS = 121  # D104
WRITE_VALUE = 400
READ_ADDRESS = 120  # D120

# iQ-R extended frame: 4-byte device number + 2-byte device code (subcommand
# 0x0002). Set False for the Q/L-compatible layout (3-byte number + 1-byte
# code, subcommand 0x0000) if the CPU answers 0xC059 "command error".
IQ_R_FRAME = True

# --- 3E frame constants (MELSEC Communication Protocol Reference Manual) ---
REQUEST_SUBHEADER = b"\x50\x00"
RESPONSE_SUBHEADER = b"\xd0\x00"
NETWORK_NO = 0x00  # own network
PC_NO = 0xFF  # own station
IO_NUMBER = 0x03FF  # own CPU
MULTIDROP_NO = 0x00
MONITORING_TIMER = 0x0010  # 16 * 250 ms = 4 s

CMD_BATCH_READ_WORDS = 0x0401
CMD_BATCH_WRITE_WORDS = 0x1401
DEVICE_CODE_D = 0xA8  # data register; 0xA8 as 1 byte, 0x00A8 as 2 bytes

UINT16_MAX = 65535


class SlmpError(Exception):
    """The PLC rejected a request or replied with a malformed frame."""


def _subcommand() -> int:
    return 0x0002 if IQ_R_FRAME else 0x0000


def _device_spec(address: int, device_code: int = DEVICE_CODE_D) -> bytes:
    """Encode the head device as the frame's device-number + device-code pair."""
    if IQ_R_FRAME:
        return address.to_bytes(4, "little") + device_code.to_bytes(2, "little")
    return address.to_bytes(3, "little") + device_code.to_bytes(1, "little")


def _build_frame(command: int, payload: bytes) -> bytes:
    body = (
        MONITORING_TIMER.to_bytes(2, "little")
        + command.to_bytes(2, "little")
        + _subcommand().to_bytes(2, "little")
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


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise SlmpError("PLC closed the connection mid-reply")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _transact(sock: socket.socket, command: int, payload: bytes) -> bytes:
    """Send one request; return the response data that follows the end code."""
    sock.sendall(_build_frame(command, payload))

    header = _recv_exact(sock, 9)  # subheader..response data length
    if header[:2] != RESPONSE_SUBHEADER:
        raise SlmpError(f"Unexpected response subheader {header[:2].hex()}")

    body = _recv_exact(sock, int.from_bytes(header[7:9], "little"))
    end_code = int.from_bytes(body[:2], "little")
    if end_code != 0x0000:
        raise SlmpError(f"PLC returned end code 0x{end_code:04X}")
    return body[2:]


def read_words(sock: socket.socket, address: int, count: int = 1) -> list[int]:
    """Batch-read *count* consecutive D registers starting at *address*."""
    payload = _device_spec(address) + count.to_bytes(2, "little")
    data = _transact(sock, CMD_BATCH_READ_WORDS, payload)
    if len(data) < count * 2:
        raise SlmpError(f"Short reply: wanted {count * 2} data bytes, got {len(data)}")
    return [int.from_bytes(data[i : i + 2], "little") for i in range(0, count * 2, 2)]


def write_words(sock: socket.socket, address: int, values: list[int]) -> None:
    """Batch-write consecutive D registers starting at *address*."""
    for value in values:
        if not 0 <= value <= UINT16_MAX:
            raise SlmpError(f"Value {value} out of uint16 range for D{address}")
    payload = (
        _device_spec(address)
        + len(values).to_bytes(2, "little")
        + b"".join(v.to_bytes(2, "little") for v in values)
    )
    _transact(sock, CMD_BATCH_WRITE_WORDS, payload)


def main() -> None:
    try:
        sock = socket.create_connection((PLC_IP, PLC_PORT), timeout=TIMEOUT_S)
    except OSError as exc:
        print(f"Cannot connect to PLC {PLC_IP}:{PLC_PORT}: {exc}")
        return

    try:
        print(f"Connected to PLC {PLC_IP}:{PLC_PORT} (SLMP 3E binary)")

        write_words(sock, WRITE_ADDRESS, [WRITE_VALUE])
        print(f"Wrote {WRITE_VALUE} to D{WRITE_ADDRESS}")

        value = read_words(sock, READ_ADDRESS)[0]
        print(f"D{READ_ADDRESS} = {value}")
    except (SlmpError, OSError) as exc:
        print(f"SLMP request failed: {exc}")
    finally:
        sock.close()
        print("Disconnected")


if __name__ == "__main__":
    main()

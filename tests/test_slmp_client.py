"""SlmpPlcClient frame encoding and error mapping against a loopback CPU stub.

The stub decodes real 3E binary frames rather than replaying canned bytes, so
a malformed request fails the test instead of quietly matching a fixture.
"""

import socket
import threading

import pytest

from core.plc.slmp_client import SlmpPlcClient
from core.utilities.exceptions import PlcReadError, PlcWriteError

RESPONSE_SUBHEADER = b"\xd0\x00"


class FakeCpu:
    """Minimal SLMP/3E binary server: batch word read + write over one D area."""

    def __init__(self, end_code: int = 0x0000) -> None:
        self.end_code = end_code
        self.devices: dict[int, int] = {}
        self.last_subcommand: int | None = None
        self._server = socket.socket()
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._server.accept()
        except OSError:
            return
        with conn:
            while True:
                header = self._recv_exact(conn, 9)
                if header is None:
                    return
                body = self._recv_exact(conn, int.from_bytes(header[7:9], "little"))
                if body is None:
                    return
                conn.sendall(self._handle(body))

    def _handle(self, body: bytes) -> bytes:
        command = int.from_bytes(body[2:4], "little")
        self.last_subcommand = int.from_bytes(body[4:6], "little")
        payload = body[6:]

        if self.last_subcommand == 0x0002:  # iQ-R: 4-byte number + 2-byte code
            address = int.from_bytes(payload[0:4], "little")
            assert int.from_bytes(payload[4:6], "little") == 0xA8
            payload = payload[6:]
        else:  # Q/L: 3-byte number + 1-byte code
            address = int.from_bytes(payload[0:3], "little")
            assert payload[3] == 0xA8
            payload = payload[4:]

        count = int.from_bytes(payload[0:2], "little")
        data = b""
        if self.end_code == 0x0000:
            if command == 0x0401:
                data = b"".join(
                    self.devices.get(address + i, 0).to_bytes(2, "little")
                    for i in range(count)
                )
            elif command == 0x1401:
                for i in range(count):
                    word = payload[2 + i * 2 : 4 + i * 2]
                    self.devices[address + i] = int.from_bytes(word, "little")

        response = self.end_code.to_bytes(2, "little") + data
        return (
            RESPONSE_SUBHEADER
            + b"\x00\xff\xff\x03\x00"
            + len(response).to_bytes(2, "little")
            + response
        )

    @staticmethod
    def _recv_exact(conn: socket.socket, count: int) -> bytes | None:
        chunks = []
        while count > 0:
            chunk = conn.recv(count)
            if not chunk:
                return None
            chunks.append(chunk)
            count -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        self._server.close()


@pytest.fixture()
def cpu() -> FakeCpu:
    server = FakeCpu()
    yield server
    server.close()


def _client(cpu: FakeCpu, frame: str = "iq_r") -> SlmpPlcClient:
    client = SlmpPlcClient("127.0.0.1", port=cpu.port, timeout_s=2.0, frame=frame)
    client.connect()
    return client


def test_write_then_read_round_trip(cpu) -> None:
    client = _client(cpu)
    try:
        client.write_register(104, 300)
        assert cpu.devices[104] == 300
        cpu.devices[120] = 4711
        assert client.read_registers(120) == [4711]
    finally:
        client.disconnect()


def test_iq_r_frame_uses_extended_subcommand(cpu) -> None:
    client = _client(cpu, frame="iq_r")
    try:
        client.read_registers(120)
        assert cpu.last_subcommand == 0x0002
    finally:
        client.disconnect()


def test_q_frame_uses_classic_subcommand(cpu) -> None:
    client = _client(cpu, frame="q")
    try:
        client.read_registers(120)
        assert cpu.last_subcommand == 0x0000
    finally:
        client.disconnect()


def test_multi_word_write_is_one_transaction(cpu) -> None:
    client = _client(cpu)
    try:
        client.write_registers(110, [11, 22, 33])
        assert [cpu.devices[a] for a in (110, 111, 112)] == [11, 22, 33]
        assert client.read_registers(110, 3) == [11, 22, 33]
    finally:
        client.disconnect()


def test_run_write_disabled_maps_to_write_error(cpu) -> None:
    cpu.end_code = 0x0055
    client = _client(cpu)
    try:
        with pytest.raises(PlcWriteError, match="0x0055"):
            client.write_register(104, 300)
    finally:
        client.disconnect()


def test_end_code_on_read_maps_to_read_error(cpu) -> None:
    cpu.end_code = 0xC059
    client = _client(cpu)
    try:
        with pytest.raises(PlcReadError, match="0xC059"):
            client.read_registers(120)
    finally:
        client.disconnect()


def test_out_of_range_value_rejected_before_send(cpu) -> None:
    client = _client(cpu)
    try:
        with pytest.raises(PlcWriteError, match="uint16"):
            client.write_register(104, 70000)
        assert cpu.devices == {}
    finally:
        client.disconnect()

"""SlmpPlcClient frame encoding and error mapping against a loopback CPU stub.

The stub decodes real 3E binary frames rather than replaying canned bytes, so
a malformed request fails the test instead of quietly matching a fixture.
"""

import socket
import threading

import pytest

from core.plc.slmp_client import SlmpPlcClient, _pack_bits, _unpack_bits
from core.utilities.exceptions import PlcReadError, PlcWriteError

RESPONSE_SUBHEADER = b"\xd0\x00"


class FakeCpu:
    """Minimal SLMP/3E binary server: batch word read/write over one D area,
    and batch bit read/write over one M area — decodes real frames (device
    code + bit/word subcommand flag) rather than assuming which one was
    sent, so a mixed-up device code or subcommand fails the test."""

    def __init__(self, end_code: int = 0x0000) -> None:
        self.end_code = end_code
        self.devices: dict[int, int] = {}
        self.coils: dict[int, bool] = {}
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
        is_iq_r = bool(self.last_subcommand & 0x0002)
        is_bits = bool(self.last_subcommand & 0x0001)

        if is_iq_r:  # 4-byte device number + 2-byte device code
            address = int.from_bytes(payload[0:4], "little")
            device_code = int.from_bytes(payload[4:6], "little")
            payload = payload[6:]
        else:  # Q/L: 3-byte device number + 1-byte device code
            address = int.from_bytes(payload[0:3], "little")
            device_code = payload[3]
            payload = payload[4:]
        assert device_code == (0x90 if is_bits else 0xA8)

        count = int.from_bytes(payload[0:2], "little")
        data = b""
        if self.end_code == 0x0000:
            if command == 0x0401:
                if is_bits:
                    data = _pack_bits([self.coils.get(address + i, False) for i in range(count)])
                else:
                    data = b"".join(
                        self.devices.get(address + i, 0).to_bytes(2, "little")
                        for i in range(count)
                    )
            elif command == 0x1401:
                if is_bits:
                    for i, bit in enumerate(_unpack_bits(payload[2:], count)):
                        self.coils[address + i] = bit
                else:
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


def test_write_coil_then_read_round_trip(cpu) -> None:
    client = _client(cpu)
    try:
        client.write_coil(200, True)
        assert cpu.coils[200] is True
        cpu.coils[210] = True
        assert client.read_coils(210) == [True]
    finally:
        client.disconnect()


def test_read_coils_multiple_points(cpu) -> None:
    client = _client(cpu)
    try:
        cpu.coils.update({50: True, 51: False, 52: True})
        assert client.read_coils(50, 3) == [True, False, True]
    finally:
        client.disconnect()


def test_coil_write_uses_bit_subcommand_and_m_device_code(cpu) -> None:
    client = _client(cpu, frame="iq_r")
    try:
        client.write_coil(200, True)
        assert cpu.last_subcommand == 0x0003  # iQ-R (0x0002) | bit units (0x0001)
    finally:
        client.disconnect()


def test_register_write_still_uses_word_subcommand(cpu) -> None:
    client = _client(cpu, frame="iq_r")
    try:
        client.write_register(104, 1)
        assert cpu.last_subcommand == 0x0002  # unaffected by the new bit-unit path
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


@pytest.mark.parametrize(
    "bits",
    [
        [True],
        [False],
        [True, False],
        [True, True, False],  # odd count exercises the zero-pad path
        [False, True, False, True, True],
    ],
)
def test_pack_unpack_bits_round_trip(bits) -> None:
    packed = _pack_bits(bits)
    assert len(packed) == (len(bits) + 1) // 2
    assert _unpack_bits(packed, len(bits)) == bits


def test_pack_bits_nibble_layout() -> None:
    # point 0 -> high nibble, point 1 -> low nibble, per the module docstring
    assert _pack_bits([True, False]) == bytes([0x10])
    assert _pack_bits([False, True]) == bytes([0x01])
    assert _pack_bits([True, True]) == bytes([0x11])

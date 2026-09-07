"""Modbus TCP adapter built on pymodbus 3.x.

Version tolerance: pymodbus renamed the unit-id keyword (``slave`` →
``device_id``) during the 3.x series; the correct name is detected once at
import time so the adapter works across 3.7 – 3.x.

Thread ownership: by design all traffic flows through the single PLC worker
thread, but an internal lock keeps the adapter safe if a UI action (manual
register read on the PLC page) is ever mis-wired onto another thread.
"""

from __future__ import annotations

import inspect
import threading

from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ConnectionException, ModbusException, ModbusIOException

from core.logging import get_logger
from core.plc.plc_client_base import PlcClientBase
from core.utilities.enums import LogSource
from core.utilities.exceptions import (
    PlcConnectionError,
    PlcReadError,
    PlcTimeoutError,
    PlcWriteError,
)

logger = get_logger(LogSource.PLC)

UINT16_MAX = 65535


def _unit_id_keyword() -> str:
    """Return the unit-id keyword name used by the installed pymodbus."""
    params = inspect.signature(ModbusTcpClient.read_holding_registers).parameters
    return "device_id" if "device_id" in params else "slave"


class ModbusTcpPlcClient(PlcClientBase):
    """Synchronous Modbus TCP implementation of :class:`PlcClientBase`."""

    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = 1,
        timeout_s: float = 1.0,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._unit_kwargs = {_unit_id_keyword(): unit_id}
        self._client: ModbusTcpClient | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------ connection
    @property
    def connected(self) -> bool:
        with self._lock:
            if self._client is None:
                return False
            try:
                return bool(self._client.is_socket_open())
            except Exception:
                return False

    def connect(self) -> None:
        with self._lock:
            self.disconnect()
            self._client = ModbusTcpClient(
                self._host, port=self._port, timeout=self._timeout_s
            )
            try:
                ok = self._client.connect()
            except Exception as exc:
                self._client = None
                raise PlcConnectionError(
                    f"Cannot connect to PLC {self._host}:{self._port}: {exc}"
                ) from exc
            if not ok:
                self._client = None
                raise PlcConnectionError(
                    f"PLC {self._host}:{self._port} is unreachable"
                )
        logger.info("Connected to PLC %s:%d (Modbus TCP)", self._host, self._port)

    def disconnect(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:  # closing a dead socket must never raise
                    pass
                self._client = None

    # ------------------------------------------------------------------- I/O
    def read_registers(self, address: int, count: int = 1) -> list[int]:
        with self._lock:
            client = self._require_client()
            try:
                response = client.read_holding_registers(
                    address, count=count, **self._unit_kwargs
                )
            except ModbusIOException as exc:
                raise PlcTimeoutError(f"Timeout reading register {address}: {exc}") from exc
            except ConnectionException as exc:
                raise PlcConnectionError(f"Connection lost reading {address}: {exc}") from exc
            except ModbusException as exc:
                raise PlcReadError(f"Read failed at register {address}: {exc}") from exc
            if response.isError():
                raise PlcReadError(f"PLC rejected read at register {address}: {response}")
            return list(response.registers)

    def write_register(self, address: int, value: int) -> None:
        self._write(address, value, single=True)

    def write_registers(self, address: int, values: list[int]) -> None:
        self._write(address, values, single=False)

    def read_coils(self, address: int, count: int = 1) -> list[bool]:
        with self._lock:
            client = self._require_client()
            try:
                response = client.read_coils(address, count=count, **self._unit_kwargs)
            except ModbusIOException as exc:
                raise PlcTimeoutError(f"Timeout reading coil {address}: {exc}") from exc
            except ConnectionException as exc:
                raise PlcConnectionError(f"Connection lost reading coil {address}: {exc}") from exc
            except ModbusException as exc:
                raise PlcReadError(f"Read failed at coil {address}: {exc}") from exc
            if response.isError():
                raise PlcReadError(f"PLC rejected read at coil {address}: {response}")
            return list(response.bits[:count])

    def write_coil(self, address: int, value: bool) -> None:
        with self._lock:
            client = self._require_client()
            try:
                response = client.write_coil(address, bool(value), **self._unit_kwargs)
            except ModbusIOException as exc:
                raise PlcTimeoutError(f"Timeout writing coil {address}: {exc}") from exc
            except ConnectionException as exc:
                raise PlcConnectionError(f"Connection lost writing coil {address}: {exc}") from exc
            except ModbusException as exc:
                raise PlcWriteError(f"Write failed at coil {address}: {exc}") from exc
            if response.isError():
                raise PlcWriteError(f"PLC rejected write at coil {address}: {response}")

    # -------------------------------------------------------------- internal
    def _require_client(self) -> ModbusTcpClient:
        if self._client is None:
            raise PlcConnectionError("Not connected to PLC")
        return self._client

    def _write(self, address: int, payload: int | list[int], *, single: bool) -> None:
        values = [payload] if single else list(payload)  # type: ignore[list-item]
        for value in values:
            if not 0 <= int(value) <= UINT16_MAX:
                raise PlcWriteError(f"Value {value} out of uint16 range for register {address}")

        with self._lock:
            client = self._require_client()
            try:
                if single:
                    response = client.write_register(address, int(values[0]), **self._unit_kwargs)
                else:
                    response = client.write_registers(
                        address, [int(v) for v in values], **self._unit_kwargs
                    )
            except ModbusIOException as exc:
                raise PlcTimeoutError(f"Timeout writing register {address}: {exc}") from exc
            except ConnectionException as exc:
                raise PlcConnectionError(f"Connection lost writing {address}: {exc}") from exc
            except ModbusException as exc:
                raise PlcWriteError(f"Write failed at register {address}: {exc}") from exc
            if response.isError():
                raise PlcWriteError(f"PLC rejected write at register {address}: {response}")

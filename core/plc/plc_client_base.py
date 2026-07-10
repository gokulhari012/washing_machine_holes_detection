"""Abstract PLC client interface (adapter pattern).

Concrete adapters (Modbus TCP today, EtherNet/IP via pycomm3 tomorrow)
implement this interface; everything above it — ``PlcManager``, workers,
services, UI — depends only on this contract.

Register model: 16-bit unsigned holding registers addressed by integer.

Error contract (all from ``core.utilities.exceptions``):
- ``PlcConnectionError`` — connect failed / connection lost
- ``PlcTimeoutError``    — device did not answer in time
- ``PlcReadError`` / ``PlcWriteError`` — request rejected or malformed reply
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class PlcClientBase(ABC):
    """Contract every PLC protocol adapter must fulfil."""

    @property
    @abstractmethod
    def connected(self) -> bool:
        """True while the transport link is usable."""

    @abstractmethod
    def connect(self) -> None:
        """Open the connection.

        Raises:
            PlcConnectionError: the device is unreachable.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection. Must be safe to call repeatedly."""

    @abstractmethod
    def read_registers(self, address: int, count: int = 1) -> list[int]:
        """Read *count* consecutive holding registers starting at *address*.

        Returns:
            List of ``count`` unsigned 16-bit values.

        Raises:
            PlcConnectionError | PlcTimeoutError | PlcReadError
        """

    @abstractmethod
    def write_register(self, address: int, value: int) -> None:
        """Write one unsigned 16-bit value.

        Raises:
            PlcConnectionError | PlcTimeoutError | PlcWriteError
        """

    @abstractmethod
    def write_registers(self, address: int, values: list[int]) -> None:
        """Write consecutive unsigned 16-bit values starting at *address*.

        Raises:
            PlcConnectionError | PlcTimeoutError | PlcWriteError
        """

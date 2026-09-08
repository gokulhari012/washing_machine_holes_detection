"""Abstract PLC client interface (adapter pattern).

Concrete adapters (Modbus TCP today, EtherNet/IP via pycomm3 tomorrow)
implement this interface; everything above it — ``PlcManager``, workers,
services, UI — depends only on this contract.

Register model: 16-bit unsigned holding registers addressed by integer.

Coil model: single-bit registers (Modbus coils / SLMP "M" internal relays)
addressed by their own, separate integer space — coil address 5 and holding
register address 5 are different physical memory on the PLC, not the same
address read two ways. No named register in this application is a coil today;
the path exists for the PLC page's manual coil read/write and for whichever
handshake bit a station wires up next.

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

    @abstractmethod
    def read_coils(self, address: int, count: int = 1) -> list[bool]:
        """Read *count* consecutive coils starting at *address*.

        Raises:
            PlcConnectionError | PlcTimeoutError | PlcReadError
        """

    @abstractmethod
    def write_coil(self, address: int, value: bool) -> None:
        """Write one coil.

        Raises:
            PlcConnectionError | PlcTimeoutError | PlcWriteError
        """

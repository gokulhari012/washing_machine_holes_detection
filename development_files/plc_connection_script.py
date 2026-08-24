"""Standalone Modbus TCP check: write 300 to D104, then read D120.

Run directly (no project imports needed):

    python development_files/plc_connection_script.py
"""

from __future__ import annotations

import inspect

from pymodbus.client import ModbusTcpClient

PLC_IP = "192.168.3.20"
PLC_PORT = 502
UNIT_ID = 1
TIMEOUT_S = 1.0

WRITE_ADDRESS = 104  # D104
WRITE_VALUE = 300
READ_ADDRESS = 120  # D120


def _unit_kwargs() -> dict[str, int]:
    """pymodbus renamed the unit-id keyword (slave -> device_id) during 3.x."""
    params = inspect.signature(ModbusTcpClient.read_holding_registers).parameters
    name = "device_id" if "device_id" in params else "slave"
    return {name: UNIT_ID}


def main() -> None:
    kwargs = _unit_kwargs()
    client = ModbusTcpClient(PLC_IP, port=PLC_PORT, timeout=TIMEOUT_S)

    if not client.connect():
        print(f"Cannot connect to PLC {PLC_IP}:{PLC_PORT}")
        return

    try:
        print(f"Connected to PLC {PLC_IP}:{PLC_PORT}")

        response = client.write_register(WRITE_ADDRESS, WRITE_VALUE, **kwargs)
        if response.isError():
            print(f"Write failed at D{WRITE_ADDRESS}: {response}")
            return
        print(f"Wrote {WRITE_VALUE} to D{WRITE_ADDRESS}")

        response = client.read_holding_registers(READ_ADDRESS, count=1, **kwargs)
        if response.isError():
            print(f"Read failed at D{READ_ADDRESS}: {response}")
            return
        print(f"D{READ_ADDRESS} = {response.registers[0]}")
    finally:
        client.close()
        print("Disconnected")


if __name__ == "__main__":
    main()

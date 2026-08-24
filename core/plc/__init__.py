"""PLC communication: abstract client, adapters, register map, manager."""

from core.plc.plc_client_base import PlcClientBase
from core.plc.plc_manager import PlcManager, PositionMap
from core.plc.register_map import RegisterMap
from core.plc.simulated_plc import SimulatedPlc

__all__ = [
    "PlcClientBase",
    "PlcManager",
    "PositionMap",
    "RegisterMap",
    "SimulatedPlc",
    "create_plc_client",
]


def create_plc_client(plc_config: dict, register_map: RegisterMap) -> PlcClientBase:
    """Factory: build the client named by ``connection.protocol``.

    ``modbus_tcp`` imports pymodbus lazily so the simulator (and tests) run
    without the dependency installed; ``slmp`` needs no third-party package
    at all.
    """
    connection = plc_config.get("connection", {})
    protocol = str(connection.get("protocol", "modbus_tcp")).lower()

    if protocol == "simulated":
        return SimulatedPlc(
            register_map=register_map,
            auto_cycle_interval_s=float(connection.get("auto_cycle_interval_s", 0.0)),
        )
    if protocol == "modbus_tcp":
        from core.plc.modbus_client import ModbusTcpPlcClient

        return ModbusTcpPlcClient(
            host=str(connection.get("ip", "192.168.0.10")),
            port=int(connection.get("port", 502)),
            unit_id=int(connection.get("unit_id", 1)),
            timeout_s=int(connection.get("timeout_ms", 1000)) / 1000.0,
        )
    if protocol == "slmp":
        from core.plc.slmp_client import SlmpPlcClient

        return SlmpPlcClient(
            host=str(connection.get("ip", "192.168.0.10")),
            port=int(connection.get("port", 5007)),
            timeout_s=int(connection.get("timeout_ms", 1000)) / 1000.0,
            frame=str(connection.get("slmp_frame", "iq_r")),
        )

    from core.utilities.exceptions import ConfigurationError

    raise ConfigurationError(f"Unknown PLC protocol: {protocol!r}")

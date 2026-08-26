"""PLC facade for the UI: configuration persistence, tests, manual access.

Live polling belongs to the PLC worker; this service covers everything the
PLC Configuration page does — saving settings (JSON + DB audit mirror),
"Test Connection" against arbitrary parameters, manual/raw register access
and reconnect requests.
"""

from __future__ import annotations

from core.logging import get_logger
from core.plc import PlcManager, RegisterMap, create_plc_client
from core.utilities import ConfigManager
from core.utilities.enums import ConnectionState, LogSource
from core.utilities.exceptions import ConfigurationError, PlcError
from services.database_service import DatabaseService

logger = get_logger(LogSource.PLC)


class PlcService:
    """Everything the PLC page needs; owns no thread, no polling."""

    def __init__(
        self,
        plc_manager: PlcManager,
        config_manager: ConfigManager,
        database_service: DatabaseService,
    ) -> None:
        self._manager = plc_manager
        self._config = config_manager
        self._database = database_service

    # ---------------------------------------------------------------- state
    @property
    def state(self) -> ConnectionState:
        return self._manager.state

    @property
    def last_error(self) -> str:
        return self._manager.last_error

    def request_reconnect(self) -> None:
        """Drop the link; the poll worker's ensure_connected() re-establishes it."""
        self._manager.disconnect()
        logger.info("Manual reconnect requested")

    # ---------------------------------------------------------------- config
    def get_config(self) -> dict:
        return self._config.load("plc")

    def save_config(self, plc_config: dict) -> None:
        """Validate, persist to plc.json and mirror to the audit table.

        The runtime client/worker rebuild is wired in the composition root via
        ``ConfigManager.subscribe("plc", ...)``.

        Raises:
            ConfigurationError: register map malformed.
        """
        RegisterMap.from_config(plc_config)  # validate before persisting
        self._config.save("plc", plc_config)
        self._mirror_to_database(plc_config)

    @staticmethod
    def test_connection(plc_config: dict) -> tuple[bool, str]:
        """Try the given settings with a throwaway client; returns (ok, message)."""
        try:
            register_map = RegisterMap.from_config(plc_config)
            client = create_plc_client(plc_config, register_map)
            client.connect()
            try:
                value = client.read_registers(register_map.trigger, 1)[0]
            finally:
                client.disconnect()
            return True, f"Connected. Trigger register {register_map.trigger} = {value}"
        except (PlcError, ConfigurationError) as exc:
            return False, str(exc)

    # -------------------------------------------------------- manual access
    def read_register(self, address: int, count: int = 1) -> list[int]:
        """Raises PlcError when the link is down or the read is rejected."""
        return self._manager.read_raw(address, count)

    def write_register(self, address: int, value: int) -> None:
        """Raises PlcError; caller (UI) must gate this behind admin login."""
        self._manager.write_raw(address, value)

    # ---------------------------------------------------------- camera jog
    def jog_configured(self, camera_index: int) -> bool:
        return self._manager.jog_configured(camera_index)

    def jog_camera(self, camera_index: int, direction: str) -> tuple[int, int]:
        """Raises ConfigurationError/PlcError; caller (UI) must gate this
        behind admin login, same as write_register."""
        return self._manager.jog_camera(camera_index, direction)

    def home_camera(self, camera_index: int) -> tuple[int, int]:
        """Raises ConfigurationError/PlcError; caller (UI) must gate this
        behind admin login, same as write_register."""
        return self._manager.home_camera(camera_index)

    def read_camera_position(self, camera_index: int) -> tuple[int, int]:
        """Raises ConfigurationError/PlcError."""
        return self._manager.read_camera_jog_position(camera_index)

    def set_camera_position(self, camera_index: int, x: int, y: int) -> tuple[int, int]:
        """Raises ConfigurationError/PlcError; caller (UI) must gate this
        behind admin login, same as write_register."""
        return self._manager.set_camera_jog_position(camera_index, x, y)

    # -------------------------------------------------------------- internal
    def _mirror_to_database(self, plc_config: dict) -> None:
        connection = plc_config.get("connection", {})
        registers = plc_config.get("registers", {})
        scaling = plc_config.get("scaling", {})
        camera_positions = registers.get("camera_positions", {})

        values: dict = {
            "ip": connection.get("ip", ""),
            "port": int(connection.get("port", 502)),
            "protocol": connection.get("protocol", "modbus_tcp"),
            "unit_id": int(connection.get("unit_id", 1)),
            "timeout_ms": int(connection.get("timeout_ms", 1000)),
            "poll_interval_ms": int(connection.get("poll_interval_ms", 50)),
            "trigger_register": int(registers.get("trigger", 0)),
            "machine_number_register": int(registers.get("machine_number", 0)),
            "heartbeat_register": int(registers.get("heartbeat", 0)),
            "result_register": int(registers.get("result", 0)),
            "vision_complete_register": int(registers.get("vision_complete", 0)),
            "position_scale": int(scaling.get("position_scale", 10)),
            "position_offset": int(scaling.get("position_offset", 10000)),
        }
        for index in (1, 2, 3, 4):
            addresses = camera_positions.get(str(index), {})
            values[f"cam{index}_x_register"] = int(addresses.get("x", 0))
            values[f"cam{index}_y_register"] = int(addresses.get("y", 0))
        self._database.plc_config.save(values)

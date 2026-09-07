"""In-memory PLC simulator.

Purposes:
- run the full application (dashboard, inspection cycle, DB writes) with no
  hardware attached — the factory function selects it when
  ``connection.protocol == "simulated"``;
- deterministic unit tests for ``PlcManager`` and the poll worker.

When constructed with a :class:`RegisterMap` and ``auto_cycle_interval_s > 0``
it emulates the production handshake on a background thread:

    every N s: machine_number += 1, trigger := 1
    → waits for the PC to set vision_complete = 1 (or times out)
    → clears trigger and vision_complete, cycle complete
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from core.logging import get_logger
from core.plc.plc_client_base import PlcClientBase
from core.plc.register_map import RegisterMap
from core.utilities.enums import LogSource
from core.utilities.exceptions import PlcConnectionError, PlcWriteError

logger = get_logger(LogSource.PLC)

UINT16_MAX = 65535


class SimulatedPlc(PlcClientBase):
    """Thread-safe fake PLC with an optional automatic trigger cycle.

    Holding registers and coils are kept in separate dicts (``_registers`` /
    ``_coils``), matching the real protocols where they are distinct address
    spaces — coil 5 and register 5 are unrelated memory here too.
    """

    RESPONSE_TIMEOUT_S = 10.0  # how long the fake PLC waits for vision_complete
    RESULT_HOLD_S = 0.2        # how long results stay latched before reset

    def __init__(
        self,
        register_map: RegisterMap | None = None,
        auto_cycle_interval_s: float = 0.0,
        initial_machine_number: int = 1000,
    ) -> None:
        self._map = register_map
        self._interval_s = auto_cycle_interval_s
        self._machine_number = initial_machine_number
        self._registers: dict[int, int] = defaultdict(int)
        self._coils: dict[int, bool] = defaultdict(bool)
        self._lock = threading.Lock()
        self._connected = False
        self._stop = threading.Event()
        self._cycle_thread: threading.Thread | None = None

    # ------------------------------------------------------------ connection
    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True
        if self._map is not None and self._interval_s > 0 and self._cycle_thread is None:
            self._stop.clear()
            self._cycle_thread = threading.Thread(
                target=self._cycle_loop, name="SimulatedPlcCycle", daemon=True
            )
            self._cycle_thread.start()
        logger.info(
            "Simulated PLC connected (auto cycle: %s)",
            f"every {self._interval_s}s" if self._interval_s > 0 else "off",
        )

    def disconnect(self) -> None:
        self._connected = False
        self._stop.set()
        if self._cycle_thread is not None:
            self._cycle_thread.join(timeout=2.0)
            self._cycle_thread = None

    # ------------------------------------------------------------------- I/O
    def read_registers(self, address: int, count: int = 1) -> list[int]:
        self._require_connected()
        with self._lock:
            return [self._registers[address + offset] for offset in range(count)]

    def write_register(self, address: int, value: int) -> None:
        self._require_connected()
        if not 0 <= int(value) <= UINT16_MAX:
            raise PlcWriteError(f"Value {value} out of uint16 range for register {address}")
        with self._lock:
            self._registers[address] = int(value)

    def write_registers(self, address: int, values: list[int]) -> None:
        for offset, value in enumerate(values):
            self.write_register(address + offset, value)

    def read_coils(self, address: int, count: int = 1) -> list[bool]:
        self._require_connected()
        with self._lock:
            return [self._coils[address + offset] for offset in range(count)]

    def write_coil(self, address: int, value: bool) -> None:
        self._require_connected()
        with self._lock:
            self._coils[address] = bool(value)

    # -------------------------------------------------- test/demo assistance
    def set_register(self, address: int, value: int) -> None:
        """Backdoor for tests/demo: set a register regardless of connection."""
        with self._lock:
            self._registers[address] = int(value) & UINT16_MAX

    def get_register(self, address: int) -> int:
        with self._lock:
            return self._registers[address]

    def set_coil(self, address: int, value: bool) -> None:
        """Backdoor for tests/demo: set a coil regardless of connection."""
        with self._lock:
            self._coils[address] = bool(value)

    def get_coil(self, address: int) -> bool:
        with self._lock:
            return self._coils[address]

    def fire_trigger(self, machine_number: int | None = None) -> None:
        """Manually raise one trigger (used by a 'Simulate Trigger' UI button)."""
        if self._map is None:
            raise PlcWriteError("SimulatedPlc has no register map; cannot fire trigger")
        with self._lock:
            if machine_number is None:
                self._machine_number += 1
                machine_number = self._machine_number
            self._registers[self._map.machine_number] = machine_number & UINT16_MAX
            self._registers[self._map.trigger] = 1

    # -------------------------------------------------------------- internal
    def _require_connected(self) -> None:
        if not self._connected:
            raise PlcConnectionError("Simulated PLC is not connected")

    def _cycle_loop(self) -> None:
        assert self._map is not None
        while not self._stop.wait(self._interval_s):
            self.fire_trigger()
            logger.debug("Simulated PLC raised trigger (machine %d)", self._machine_number)

            deadline = time.monotonic() + self.RESPONSE_TIMEOUT_S
            while time.monotonic() < deadline and not self._stop.is_set():
                with self._lock:
                    if self._registers[self._map.vision_complete] == 1:
                        break
                time.sleep(0.02)

            time.sleep(self.RESULT_HOLD_S)
            with self._lock:  # PLC-side reset ends the handshake
                self._registers[self._map.trigger] = 0
                self._registers[self._map.vision_complete] = 0

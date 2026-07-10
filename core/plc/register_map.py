"""Typed view of the PLC register layout defined in ``config/plc.json``.

Position encoding
-----------------
Hole positions are millimetres with one decimal, carried in a 16-bit
unsigned register:

    raw = round(mm * position_scale) + position_offset      (default ×10 +10000)

so with defaults the representable range is -1000.0 mm .. +5553.5 mm.
Raw ``0`` is reserved as the **no-hole sentinel** (it would decode to
-1000.0 mm, far outside any physical field of view).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.utilities.exceptions import ConfigurationError

UINT16_MAX = 65535


@dataclass(frozen=True)
class RegisterMap:
    """Immutable register layout + position codec."""

    NO_HOLE_RAW = 0  # class constant, written when a camera finds no hole

    trigger: int
    machine_number: int
    heartbeat: int
    result: int
    vision_complete: int
    camera_positions: dict[int, tuple[int, int]] = field(default_factory=dict)
    position_scale: int = 10
    position_offset: int = 10000

    @classmethod
    def from_config(cls, plc_config: dict) -> "RegisterMap":
        """Build from the parsed ``plc.json`` dict.

        Raises:
            ConfigurationError: required keys missing or malformed.
        """
        try:
            registers = plc_config["registers"]
            scaling = plc_config.get("scaling", {})
            camera_positions = {
                int(index): (int(addrs["x"]), int(addrs["y"]))
                for index, addrs in registers["camera_positions"].items()
            }
            return cls(
                trigger=int(registers["trigger"]),
                machine_number=int(registers["machine_number"]),
                heartbeat=int(registers["heartbeat"]),
                result=int(registers["result"]),
                vision_complete=int(registers["vision_complete"]),
                camera_positions=camera_positions,
                position_scale=int(scaling.get("position_scale", 10)),
                position_offset=int(scaling.get("position_offset", 10000)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid PLC register configuration: {exc}") from exc

    # ----------------------------------------------------------------- codec
    def encode_position(self, mm: float) -> int:
        """Millimetres → raw register value (clamped to the uint16 range)."""
        raw = round(mm * self.position_scale) + self.position_offset
        return max(0, min(UINT16_MAX, raw))

    def decode_position(self, raw: int) -> float:
        """Raw register value → millimetres."""
        return (raw - self.position_offset) / self.position_scale

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
    # Per-camera GOOD/NG/ERROR result register — camera index -> address.
    # Optional/independent of camera_positions: a camera absent here simply
    # gets no individual result register written, same as an absent entry
    # in camera_positions gets no X/Y written.
    camera_results: dict[int, int] = field(default_factory=dict)
    position_scale: int = 10
    position_offset: int = 10000
    # Machine-model select register: which part/model is mounted, written by
    # the PLC. Optional — None means the feature is inert (no address wired
    # up yet), never a config error, so existing plc.json files keep working.
    model_select: int | None = None
    # Physical camera-position jog control — unrelated to camera_positions
    # above (that's the *detected hole* coordinate the app writes out as an
    # inspection result; this is the camera *mount's* position, driven by
    # PLC-controlled actuators). camera_jog: camera index -> (x_addr, y_addr)
    # register addresses; camera_jog_home: camera index -> (home_x, home_y)
    # *values* written by the Home action. Both optional/per-camera — a
    # camera absent from either dict simply has no jog control available.
    camera_jog: dict[int, tuple[int, int]] = field(default_factory=dict)
    camera_jog_home: dict[int, tuple[int, int]] = field(default_factory=dict)
    jog_step: int = 10

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
            camera_results = {
                int(index): int(address)
                for index, address in registers.get("camera_results", {}).items()
            }
            model_select = registers.get("model_select")

            jog_cfg = plc_config.get("camera_jog", {})
            jog_registers = jog_cfg.get("registers", {})
            camera_jog = {
                int(index): (int(entry["x"]), int(entry["y"]))
                for index, entry in jog_registers.items()
            }
            camera_jog_home = {
                int(index): (int(entry.get("home_x", 0)), int(entry.get("home_y", 0)))
                for index, entry in jog_registers.items()
            }

            return cls(
                trigger=int(registers["trigger"]),
                machine_number=int(registers["machine_number"]),
                heartbeat=int(registers["heartbeat"]),
                result=int(registers["result"]),
                vision_complete=int(registers["vision_complete"]),
                camera_positions=camera_positions,
                camera_results=camera_results,
                position_scale=int(scaling.get("position_scale", 10)),
                position_offset=int(scaling.get("position_offset", 10000)),
                model_select=int(model_select) if model_select is not None else None,
                camera_jog=camera_jog,
                camera_jog_home=camera_jog_home,
                jog_step=int(jog_cfg.get("step", 10)),
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

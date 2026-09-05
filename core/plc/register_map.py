"""Typed view of the PLC register layout defined in ``config/plc.json``.

Position encoding
-----------------
Hole positions are millimetres with one decimal. A 16-bit holding register
is unsigned, so each coordinate is carried as **two** registers: a magnitude
and a sign.

    magnitude = round(abs(mm) * position_scale)             (default ×10)
    sign      = SIGN_NEGATIVE (1) | SIGN_POSITIVE (2)

With the default scale that covers 0.0 .. 6553.5 mm either side of zero,
which is far more range than any of these fields of view need.

Splitting the sign out this way, rather than biasing the magnitude by a
fixed offset, means the PLC program reads a plain unsigned millimetre value
it can display or compare directly, with no arithmetic to undo first.

``SIGN_NONE`` (0) is the **no-hole sentinel**: written to both sign
registers, with the magnitudes zeroed, when a camera found no hole. Zero is
a legitimate magnitude (a hole exactly on centre), so the sign register is
the only unambiguous place to say "no measurement" — a station that has not
wired up sign registers therefore reports no-hole as a plain 0/0, which is
indistinguishable from a centred hole. Wire the sign registers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.utilities.exceptions import ConfigurationError

UINT16_MAX = 65535


@dataclass(frozen=True)
class RegisterMap:
    """Immutable register layout + position codec."""

    NO_HOLE_RAW = 0  # magnitude written when a camera finds no hole

    # Sign-register codes. The PLC reads an unsigned magnitude plus one of
    # these; 0 is not a sign at all, it means "this camera reported no hole
    # this cycle", so a PLC that sees it must not treat the 0 magnitude
    # beside it as a measured position.
    SIGN_NONE = 0
    SIGN_NEGATIVE = 1
    SIGN_POSITIVE = 2

    # Values written to a camera_status register. "Available" rather than
    # merely "socket open": a camera that is connected but failing to grab is
    # just as unusable to the line, so it reports UNAVAILABLE too — the safe
    # direction for a PLC deciding whether to run the station.
    CAMERA_AVAILABLE = 1
    CAMERA_UNAVAILABLE = 0

    trigger: int
    machine_number: int
    heartbeat: int
    result: int
    vision_complete: int
    camera_positions: dict[int, tuple[int, int]] = field(default_factory=dict)
    # Sign registers for the coordinates above — camera index -> (x_sign_addr,
    # y_sign_addr). Optional per camera: a camera absent here still gets its
    # magnitudes written, it just cannot express a negative coordinate (see
    # the module docstring). Kept separate from camera_positions so an
    # existing plc.json with no sign block keeps loading.
    camera_position_signs: dict[int, tuple[int, int]] = field(default_factory=dict)
    # Per-camera GOOD/NG/ERROR result register — camera index -> address.
    # Optional/independent of camera_positions: a camera absent here simply
    # gets no individual result register written, same as an absent entry
    # in camera_positions gets no X/Y written.
    camera_results: dict[int, int] = field(default_factory=dict)
    # Per-camera handshake, mirroring `trigger`/`vision_complete` but scoped to
    # a single camera: the PLC raises camera_triggers[i] to inspect *only*
    # camera i, and the PC answers on camera_vision_complete[i] once that
    # camera's X/Y and result registers hold the new values. Both optional and
    # independent per camera — a camera absent from camera_triggers simply
    # cannot be triggered on its own, and one absent from
    # camera_vision_complete gets no completion flag written.
    camera_triggers: dict[int, int] = field(default_factory=dict)
    camera_vision_complete: dict[int, int] = field(default_factory=dict)
    # Per-camera availability register — camera index -> address. Written by
    # the PC (CAMERA_AVAILABLE / CAMERA_UNAVAILABLE) whenever a camera's state
    # changes, so the PLC can refuse to run the station with a dead camera.
    # Optional per camera, like every other block here.
    camera_status: dict[int, int] = field(default_factory=dict)
    position_scale: int = 10
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
            camera_position_signs = {
                int(index): (int(addrs["x"]), int(addrs["y"]))
                for index, addrs in registers.get("camera_position_signs", {}).items()
            }
            camera_results = {
                int(index): int(address)
                for index, address in registers.get("camera_results", {}).items()
            }
            camera_triggers = {
                int(index): int(address)
                for index, address in registers.get("camera_triggers", {}).items()
            }
            camera_vision_complete = {
                int(index): int(address)
                for index, address in registers.get("camera_vision_complete", {}).items()
            }
            camera_status = {
                int(index): int(address)
                for index, address in registers.get("camera_status", {}).items()
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
                camera_position_signs=camera_position_signs,
                camera_results=camera_results,
                camera_triggers=camera_triggers,
                camera_vision_complete=camera_vision_complete,
                camera_status=camera_status,
                position_scale=int(scaling.get("position_scale", 10)),
                model_select=int(model_select) if model_select is not None else None,
                camera_jog=camera_jog,
                camera_jog_home=camera_jog_home,
                jog_step=int(jog_cfg.get("step", 10)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid PLC register configuration: {exc}") from exc

    # ----------------------------------------------------------------- codec
    def encode_position(self, mm: float) -> tuple[int, int]:
        """Millimetres → ``(magnitude, sign_code)``.

        The magnitude is clamped to the uint16 range; the sign is
        :attr:`SIGN_NEGATIVE` or :attr:`SIGN_POSITIVE`. Exactly ``0.0`` is
        reported positive — it is a real measurement, and only
        :attr:`SIGN_NONE` means "no hole".
        """
        magnitude = round(abs(mm) * self.position_scale)
        magnitude = max(0, min(UINT16_MAX, magnitude))
        sign = self.SIGN_NEGATIVE if mm < 0 else self.SIGN_POSITIVE
        return magnitude, sign

    def decode_position(self, magnitude: int, sign: int = SIGN_POSITIVE) -> float:
        """``(magnitude, sign_code)`` → millimetres.

        Any sign code that is not :attr:`SIGN_NEGATIVE` decodes positive, so
        a station with no sign registers wired up (which reads back 0) still
        gets sensible positive values rather than an exception.
        """
        value = magnitude / self.position_scale
        return -value if sign == self.SIGN_NEGATIVE else value

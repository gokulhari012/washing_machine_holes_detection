"""Typed view of the PLC register layout defined in ``config/plc.json``.

Position encoding
-----------------
A hole position is the offset in millimetres from the centre of the analysed
image. A 16-bit holding register is unsigned, so a negative offset cannot be
written directly; instead the value is expressed **relative to that axis's
servo home position**, which the PLC publishes in its own register and the PC
reads back each time it writes a result:

    raw = servo_home + round(mm * position_scale)     (clamped to uint16)

e.g. servo home 6000, hole 2.0 mm right of centre, scale 100 -> 6200. A hole
2.0 mm the other way writes 5800. The PLC therefore reads one absolute servo
target per axis, in the servo's own units, with no sign handling and no
arithmetic to undo — and because home is read live rather than baked into a
config offset, moving the axis needs no change on the PC side.

Raw ``0`` remains the **no-hole sentinel** (see :attr:`RegisterMap.NO_HOLE_RAW`).
An axis whose home position sits at or very near 0 would make that sentinel
ambiguous with a real measurement; every servo home in this station is far
from zero, which is what makes the sentinel safe.

A camera with no servo-home registers configured falls back to a home of 0,
i.e. plain ``mm * position_scale``, and can then only express positions on the
positive side of centre.

Gantry gating
-------------
``gantry_status`` is the other PLC→PC per-camera input. The PLC publishes 1
there while a camera's gantry is in position; the PC inspects only those
cameras and leaves every other camera's position/result registers exactly as
the last real cycle left them. A camera with no gantry-status register is
always inspected, so an existing plc.json keeps behaving as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.utilities.exceptions import ConfigurationError

UINT16_MAX = 65535


@dataclass(frozen=True)
class RegisterMap:
    """Immutable register layout + position codec."""

    NO_HOLE_RAW = 0  # written to both position registers when no hole is found

    # Values written to a camera_status register. "Available" rather than
    # merely "socket open": a camera that is connected but failing to grab is
    # just as unusable to the line, so it reports UNAVAILABLE too — the safe
    # direction for a PLC deciding whether to run the station.
    CAMERA_AVAILABLE = 1
    CAMERA_UNAVAILABLE = 0

    # Values read from a gantry_status register. 1 means that camera's gantry
    # is in position and the camera takes part in this cycle; anything else
    # means it does not and the camera is skipped. Only 1 counts as active, so
    # a garbled or half-written value fails towards "don't inspect" rather
    # than towards inspecting a part the gantry is not presenting.
    GANTRY_ACTIVE = 1

    trigger: int
    machine_number: int
    heartbeat: int
    result: int
    vision_complete: int
    camera_positions: dict[int, tuple[int, int]] = field(default_factory=dict)
    # Servo home position registers — camera index -> (x_addr, y_addr). These
    # are PLC→PC *inputs*: the PLC publishes where each axis's home sits, and
    # every position written to camera_positions is measured from it (see the
    # module docstring). Optional per camera: a camera absent here encodes
    # against a home of 0, so an existing plc.json keeps loading.
    servo_home_positions: dict[int, tuple[int, int]] = field(default_factory=dict)
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
    # Per-camera light-brightness register — camera index -> address. The PC
    # writes the 0-255 brightness level configured for that camera (see
    # CameraSettings.brightness) whenever its settings are applied or saved,
    # so a PLC-driven light source (not the camera's own ISP) tracks it.
    # Optional per camera, like every other block here.
    camera_brightness: dict[int, int] = field(default_factory=dict)
    # Per-camera gantry-status register — camera index -> address. A PLC→PC
    # *input*, like servo_home_positions: the PLC publishes whether that
    # camera's gantry is active, and the PC inspects only the cameras whose
    # gantry reads GANTRY_ACTIVE. Read live at the start of every cycle (both
    # the global one and a single-camera one), never cached — the PLC may park
    # a gantry between cycles. Optional per camera: a camera absent here is
    # always inspected, which is exactly what every station did before this
    # register existed.
    gantry_status: dict[int, int] = field(default_factory=dict)
    position_scale: int = 10
    # Machine-model select register: which part/model is mounted, written by
    # the PLC. Optional — None means the feature is inert (no address wired
    # up yet), never a config error, so existing plc.json files keep working.
    model_select: int | None = None
    # Serial-number register: the serial of the machine on the station this
    # cycle, written by the PLC. Optional — None means the feature is inert
    # and the serial falls back to the machine number, so an existing
    # plc.json keeps working. Like every register here it is 16-bit, so the
    # PLC can publish 0-65535; the configured serial prefix is added on the
    # PC side and is never read from or written to the PLC.
    serial_number: int | None = None

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
            servo_home_positions = {
                int(index): (int(addrs["x"]), int(addrs["y"]))
                for index, addrs in registers.get("servo_home_positions", {}).items()
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
            camera_brightness = {
                int(index): int(address)
                for index, address in registers.get("camera_brightness", {}).items()
            }
            gantry_status = {
                int(index): int(address)
                for index, address in registers.get("gantry_status", {}).items()
            }
            model_select = registers.get("model_select")
            serial_number = registers.get("serial_number")

            return cls(
                trigger=int(registers["trigger"]),
                machine_number=int(registers["machine_number"]),
                heartbeat=int(registers["heartbeat"]),
                result=int(registers["result"]),
                vision_complete=int(registers["vision_complete"]),
                camera_positions=camera_positions,
                servo_home_positions=servo_home_positions,
                camera_results=camera_results,
                camera_triggers=camera_triggers,
                camera_vision_complete=camera_vision_complete,
                camera_status=camera_status,
                camera_brightness=camera_brightness,
                gantry_status=gantry_status,
                position_scale=int(scaling.get("position_scale", 10)),
                model_select=int(model_select) if model_select is not None else None,
                serial_number=int(serial_number) if serial_number is not None else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid PLC register configuration: {exc}") from exc

    # ----------------------------------------------------------------- codec
    def encode_position(self, mm: float, servo_home: int = 0) -> int:
        """Millimetres from image centre → raw register value.

        *servo_home* is the value just read from that axis's servo home
        register; the result is that home biased by the scaled offset, clamped
        to the uint16 range. A negative offset therefore encodes as a raw
        value *below* home rather than needing a sign.
        """
        raw = servo_home + round(mm * self.position_scale)
        return max(0, min(UINT16_MAX, raw))

    def decode_position(self, raw: int, servo_home: int = 0) -> float:
        """Raw register value → millimetres from image centre.

        The inverse of :meth:`encode_position`, and it needs the same
        *servo_home* the value was written against.
        """
        return (raw - servo_home) / self.position_scale

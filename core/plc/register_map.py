"""Typed view of the PLC register layout defined in ``config/plc.json``.

Position encoding
-----------------
A hole position is the offset in millimetres from the centre of the analysed
image. A holding register is unsigned, so a negative offset cannot be written
directly; instead the value is expressed **relative to that axis's servo home
position**, which the PLC publishes in its own register(s) and the PC reads
back each time it writes a result:

    raw = servo_home + round(mm * position_scale)     (clamped to uint32)

e.g. servo home 6000, hole 2.0 mm right of centre, scale 100 -> 6200. A hole
2.0 mm the other way writes 5800. The PLC therefore reads one absolute servo
target per axis, in the servo's own units, with no sign handling and no
arithmetic to undo — and because home is read live rather than baked into a
config offset, moving the axis needs no change on the PC side.

**The scale is per axis**: ``scaling.position_scale_x`` and
``scaling.position_scale_y`` in ``plc.json`` are independent, because the two
servos on a gantry need not count in the same units (one may be geared
differently, or run in 0.01 mm steps while the other runs in 0.1 mm). Every
encode/decode therefore names the axis it is working on — see
:meth:`RegisterMap.encode_position`. A file carrying only the older single
``position_scale`` key is read as "both axes at that scale", so an existing
plc.json keeps encoding exactly as it did.

Raw ``0`` remains the **no-hole sentinel** (see :attr:`RegisterMap.NO_HOLE_RAW`).
An axis whose home position sits at or very near 0 would make that sentinel
ambiguous with a real measurement; every servo home in this station is far
from zero, which is what makes the sentinel safe.

A camera with no servo-home registers configured falls back to a home of 0,
i.e. plain ``mm * that axis's position scale``, and can then only express
positions on the positive side of centre.

32-bit (double-word) registers
-------------------------------
Every *positional* value — each entry of ``camera_positions`` (the hole X/Y
written to the PLC) and ``servo_home_positions`` (the home X/Y read back from
it) — is 32-bit, not the 16-bit width every other register in this map uses.
A single Modbus/SLMP holding register only ever carries 16 bits, so a 32-bit
value spans **two consecutive registers**, addressed by the *base* (lower)
address configured for that axis in ``plc.json``: e.g. ``camera_positions.1.x
= 200`` means camera 1's X occupies registers 200 and 201, not just 200.

Word order is **low word first**: the base address holds the low 16 bits, the
next address the high 16 bits (:meth:`split_dword`/:meth:`join_dword`) — the
convention Mitsubishi D-register double-word (32-bit) access uses, which
matches this application's SLMP link (the non-simulated PLC protocol
documented in CLAUDE.md). A Modbus-only station whose PLC program expects the
opposite word order would need its own client-level swap; nothing here
assumes Modbus specifically.

Configuring an axis's X and Y addresses 2 apart (e.g. ``x=200, y=202``) lets
``PlcManager`` batch the whole 4-register X+Y pair into one read/write
transaction, the same optimisation the old 16-bit, 1-apart (``x=110,
y=111``) layout used — see :meth:`PlcManager._write_position`/
:meth:`PlcManager.read_servo_home`. It is not required: any two non-
overlapping base addresses work, just as two separate transactions instead
of one.

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
from typing import Literal

from core.utilities.exceptions import ConfigurationError

UINT32_MAX = 0xFFFFFFFF

#: Which of a camera's two position axes a scale or an encode/decode applies
#: to. The axes are scaled independently, so nothing here takes "the" scale.
Axis = Literal["x", "y"]

#: Scale used when neither the per-axis key nor the legacy shared one is
#: configured (one decimal place of a millimetre).
DEFAULT_POSITION_SCALE = 10


def axis_scale(scaling: dict, axis: Axis) -> int:
    """Read one axis's position scale out of the ``scaling`` config block.

    ``position_scale_x``/``position_scale_y`` are the current keys. A file
    written before the axes were split carries a single ``position_scale``
    instead, which stands in for both — so an older plc.json (and the
    "Restore Defaults" copy of one) keeps encoding exactly as it did.
    """
    shared = scaling.get("position_scale", DEFAULT_POSITION_SCALE)
    return int(scaling.get(f"position_scale_{axis}", shared))


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
    # Per-camera gantry-status register — camera index -> address. A PLC→PC
    # *input*, like servo_home_positions: the PLC publishes whether that
    # camera's gantry is active, and the PC inspects only the cameras whose
    # gantry reads GANTRY_ACTIVE. Read live at the start of every cycle (both
    # the global one and a single-camera one), never cached — the PLC may park
    # a gantry between cycles. Optional per camera: a camera absent here is
    # always inspected, which is exactly what every station did before this
    # register existed.
    gantry_status: dict[int, int] = field(default_factory=dict)
    # Millimetres-to-register-units scale, independent per axis: the two
    # servos behind a camera's X and Y need not count in the same units.
    # ``from_config`` fills both from the legacy single ``position_scale``
    # key when the per-axis keys are absent.
    position_scale_x: int = 10
    position_scale_y: int = 10
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
                gantry_status=gantry_status,
                position_scale_x=axis_scale(scaling, "x"),
                position_scale_y=axis_scale(scaling, "y"),
                model_select=int(model_select) if model_select is not None else None,
                serial_number=int(serial_number) if serial_number is not None else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid PLC register configuration: {exc}") from exc

    # ----------------------------------------------------------------- codec
    def scale_for(self, axis: Axis) -> int:
        """The millimetre scale for *axis* (``"x"`` or ``"y"``).

        Raises:
            ValueError: *axis* is neither ``"x"`` nor ``"y"``.
        """
        if axis == "x":
            return self.position_scale_x
        if axis == "y":
            return self.position_scale_y
        raise ValueError(f"Unknown axis {axis!r}: expected 'x' or 'y'")

    def encode_position(self, mm: float, servo_home: int = 0, *, axis: Axis) -> int:
        """Millimetres from image centre → raw 32-bit register value.

        *servo_home* is the value just read from that axis's servo home
        register pair; the result is that home biased by the scaled offset,
        clamped to the uint32 range. A negative offset therefore encodes as a
        raw value *below* home rather than needing a sign. The raw value
        returned here is the *combined* 32-bit number — see
        :meth:`split_dword` for how it becomes the two register words
        actually written to the PLC.

        *axis* selects which of the two configured scales applies and is
        keyword-only and **required**: the axes may be scaled differently, so
        a call that forgot to say which one it meant would silently encode a
        Y measurement in X units.
        """
        raw = servo_home + round(mm * self.scale_for(axis))
        return max(0, min(UINT32_MAX, raw))

    def decode_position(self, raw: int, servo_home: int = 0, *, axis: Axis) -> float:
        """Raw (combined 32-bit) register value → millimetres from image centre.

        The inverse of :meth:`encode_position`, and it needs the same
        *servo_home* and *axis* the value was written against.
        """
        return (raw - servo_home) / self.scale_for(axis)

    @staticmethod
    def split_dword(raw: int) -> tuple[int, int]:
        """Combined 32-bit value → ``(low_word, high_word)``, the order the
        pair is written to/read from the wire in (see the module docstring).
        The base address configured for an axis gets *low_word*, base+1 gets
        *high_word* — low word first, the Mitsubishi D-register convention.
        """
        raw &= UINT32_MAX
        return raw & 0xFFFF, (raw >> 16) & 0xFFFF

    @staticmethod
    def join_dword(low: int, high: int) -> int:
        """The inverse of :meth:`split_dword`: two register words → the
        combined 32-bit value."""
        return ((high & 0xFFFF) << 16) | (low & 0xFFFF)

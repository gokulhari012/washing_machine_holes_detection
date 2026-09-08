"""Shared enumerations used across all layers of the application.

String enums are used for anything persisted to JSON/SQLite so values stay
human-readable in configuration files and database rows. ``PlcResultCode`` is
an ``IntEnum`` because it is written directly into a PLC holding register.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class ConnectionState(StrEnum):
    """Lifecycle state of an external connection (PLC or camera)."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class InspectionResult(StrEnum):
    """Outcome of a single camera detection or of a whole inspection cycle.

    ``SKIPPED`` is not a judgement — it records that a camera was deliberately
    *not* inspected because the PLC reported its gantry inactive (see
    ``gantry_status`` in :mod:`core.plc.register_map`). It is stored and shown
    like any other per-camera outcome, but is excluded from the cycle's overall
    verdict and never written to a PLC result register: nothing was measured,
    so that camera's registers keep whatever the last real cycle left in them.
    """

    GOOD = "GOOD"
    NG = "NG"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class PlcResultCode(IntEnum):
    """Numeric result written to the PLC result register.

    ``ERROR`` guarantees the PLC never dead-waits when the vision side fails
    (camera fault, detection exception, timeout).
    """

    GOOD = 1
    NG = 2
    ERROR = 3


class TriggerMode(StrEnum):
    """How a camera acquires its inspection frame."""

    SOFTWARE = "software"
    HARDWARE = "hardware"
    CONTINUOUS = "continuous"


class CameraDriver(StrEnum):
    """Registered camera adapter implementations (see ``core/camera/``)."""

    SIMULATED = "simulated"
    IMAGE_FILE = "image_file"  # replays an uploaded picture / folder of pictures
    USB = "usb"
    HIKROBOT = "hikrobot"
    BASLER = "basler"
    DAHENG = "daheng"
    IDS = "ids"


class DetectorType(StrEnum):
    """Registered hole-detection strategies (see ``core/vision/``)."""

    OPENCV = "opencv"
    TEMPLATE_MATCHING = "template_matching"
    YOLO = "yolo"
    DARK_HOLE = "dark_hole"  # local-contrast; copes with partially visible bores


class LogSource(StrEnum):
    """Subsystem tag attached to every log record, filterable on the Logs page."""

    PLC = "PLC"
    CAMERA = "CAMERA"
    VISION = "VISION"
    DATABASE = "DATABASE"
    SYSTEM = "SYSTEM"
    UI = "UI"


class UserRole(StrEnum):
    """Access level for password-protected areas and nav-rail pages.

    Declared least- to most-privileged, and *ordered*: the roles nest rather
    than sit side by side, so a developer can do everything an admin can and
    an admin everything an operator can. Gates therefore ask "does this role
    cover the one I need" (:meth:`covers`) rather than comparing for
    equality — an ``== ADMIN`` test would lock a developer out of the very
    pages their role is meant to be a superset of.

    - ``operator``  — the default, logged-out view: Dashboard, Database, Logs
    - ``admin``     — adds the day-to-day engineering consoles (Cameras, PLC,
      Detection, Settings)
    - ``developer`` — adds the commissioning consoles (Calibration,
      Machine Models); every page in the app
    """

    OPERATOR = "operator"
    ADMIN = "admin"
    DEVELOPER = "developer"

    @property
    def rank(self) -> int:
        """Privilege level; higher covers lower. Not persisted — the string
        value is what reaches the database, so re-ranking a role never
        invalidates existing rows."""
        return _ROLE_RANKS[self]

    def covers(self, required: UserRole) -> bool:
        """True when this role grants everything *required* grants."""
        return self.rank >= required.rank


_ROLE_RANKS: dict[UserRole, int] = {
    UserRole.OPERATOR: 0,
    UserRole.ADMIN: 1,
    UserRole.DEVELOPER: 2,
}

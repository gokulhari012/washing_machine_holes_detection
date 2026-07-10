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
    """Outcome of a single camera detection or of a whole inspection cycle."""

    GOOD = "GOOD"
    NG = "NG"
    ERROR = "ERROR"


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


class LogSource(StrEnum):
    """Subsystem tag attached to every log record, filterable on the Logs page."""

    PLC = "PLC"
    CAMERA = "CAMERA"
    VISION = "VISION"
    DATABASE = "DATABASE"
    SYSTEM = "SYSTEM"
    UI = "UI"


class UserRole(StrEnum):
    """Access level for password-protected areas (Settings, PLC writes)."""

    ADMIN = "admin"
    OPERATOR = "operator"

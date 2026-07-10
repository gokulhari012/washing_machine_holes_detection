"""Custom exception hierarchy for the vision system.

Every subsystem raises exceptions derived from :class:`VisionSystemError` so
callers can catch domain failures in one clause without swallowing genuine
programming errors (``TypeError``, ``AttributeError`` ...), which deliberately
stay outside this tree.

Catch policy used throughout the codebase:

- Workers catch ``VisionSystemError`` subclasses, log them, raise a UI alarm
  and keep running (a production line must not stop on a recoverable fault).
- Anything else propagates to the global excepthook installed in ``main.py``.
"""

from __future__ import annotations


class VisionSystemError(Exception):
    """Base class for every recoverable, domain-specific failure."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class ConfigurationError(VisionSystemError):
    """A configuration file is missing, unreadable, or fails validation."""


# --------------------------------------------------------------------------- #
# PLC
# --------------------------------------------------------------------------- #
class PlcError(VisionSystemError):
    """Base class for PLC communication failures."""


class PlcConnectionError(PlcError):
    """TCP connection to the PLC could not be established or was lost."""


class PlcReadError(PlcError):
    """A register read returned an error response or malformed data."""


class PlcWriteError(PlcError):
    """A register write was rejected or not acknowledged."""


class PlcTimeoutError(PlcError):
    """The PLC did not answer within the configured timeout."""


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
class CameraError(VisionSystemError):
    """Base class for camera failures."""


class CameraConnectionError(CameraError):
    """The camera could not be opened or dropped its connection."""


class CameraCaptureError(CameraError):
    """A frame grab failed or timed out."""


class CameraConfigurationError(CameraError):
    """A parameter (exposure, gain, ROI ...) was rejected by the device."""


# --------------------------------------------------------------------------- #
# Vision / calibration
# --------------------------------------------------------------------------- #
class DetectionError(VisionSystemError):
    """The detection algorithm failed on an image (not a plain NG result)."""


class CalibrationError(VisionSystemError):
    """Calibration data is missing, invalid, or could not be computed."""


# --------------------------------------------------------------------------- #
# Persistence / misc
# --------------------------------------------------------------------------- #
class DatabaseError(VisionSystemError):
    """A database operation failed after retries."""


class ExportError(VisionSystemError):
    """CSV/Excel/PDF export failed."""


class AuthenticationError(VisionSystemError):
    """Login failed or the user lacks the required role."""

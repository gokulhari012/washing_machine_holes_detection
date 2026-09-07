"""Basler camera adapter (GigE Vision over Ethernet) built on ``pypylon``.

Aimed at the ace / ace 2 GigE models that reach the station over a network
cable, but the same code drives a Basler USB3 camera — only the transport
layer differs, and the device class is chosen automatically.

Station setup
-------------
1. Install the **Basler pylon Camera Software Suite** (ships the GigE filter
   driver and the pylon IP Configurator), then ``pip install pypylon``.
2. Give the camera an address on the same subnet as the PC with the pylon IP
   Configurator — a camera on a foreign subnet enumerates but cannot be
   opened. A static IP is preferred on a production line.
3. On the NIC facing the cameras: enable **jumbo frames** (MTU 9014), raise
   the receive buffers, and let the pylon filter driver bind to it. Then set
   ``packet_size`` to 8192 below — at the 1500-byte default a 1280x1024 frame
   costs about five times more interrupts.
4. In camera.json set ``driver`` to ``"basler"`` and ``connection_id`` to the
   serial number (preferred), the IP address, the user-defined name, or a
   plain enumeration index such as ``"0"``.

Optional per-camera tuning goes in a ``"basler"`` block of the same
camera.json entry — every key may be omitted::

    "basler": {
      "packet_size": 8192,            // GigE payload bytes; 8192 needs jumbo frames
      "inter_packet_delay": 0,        // GevSCPD ticks — raise when several
      "frame_transmission_delay": 0,  // GevSCFTD ticks — cameras share one NIC
      "heartbeat_timeout_ms": 5000,   // raise while debugging with breakpoints
      "grab_timeout_ms": 5000,
      "num_buffers": 5,
      "pixel_format": "Mono8",        // device default when omitted
      "color_conversion": "bgr8",     // or "native" to keep Mono8 2-D frames
      "offset_x": 0,                  // sensor window origin for width/height
      "offset_y": 0,
      "frame_rate": 0,                // >0 caps the free-run frame rate (Hz)
      "trigger_source": "Line1",      // hardware trigger input
      "trigger_activation": "RisingEdge",
      "device_class": ""              // "BaslerGigE" to refuse a USB fallback
    }

Trigger modes map to the SFNC ``FrameStart`` trigger: ``software`` issues
``ExecuteSoftwareTrigger`` per capture (the inspection pipeline default),
``hardware`` waits for the configured input line, ``continuous`` free-runs
and always hands back the newest frame.

Parameter names differ between SFNC 1.x (ace classic GigE: ``ExposureTimeAbs``,
``GainRaw``) and SFNC 2.x (ace 2, dart, USB3: ``ExposureTime``, ``Gain``), so
every write below tries the alternatives in order and skips what the model
does not expose.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.camera.camera_base import CameraBase, CameraSettings
from core.logging import get_logger
from core.utilities.enums import LogSource, TriggerMode
from core.utilities.exceptions import (
    CameraCaptureError,
    CameraConfigurationError,
    CameraConnectionError,
)

try:  # pylon is machine-installed software; keep the import optional
    from pypylon import genicam, pylon

    PYLON_AVAILABLE = True
except ImportError:
    PYLON_AVAILABLE = False

logger = get_logger(LogSource.CAMERA)

_SDK_HELP = (
    "pypylon is not installed. Install the Basler pylon Camera Software Suite "
    "and run 'pip install pypylon' (see basler_camera.py header)."
)

GIGE_DEVICE_CLASS = "BaslerGigE"
DEFAULT_GRAB_TIMEOUT_MS = 5000
DEFAULT_PACKET_SIZE = 1500
DEFAULT_NUM_BUFFERS = 5


def _require_sdk(context: str) -> None:
    """Raise a helpful connection error when pypylon is missing."""
    if not PYLON_AVAILABLE:
        raise CameraConnectionError(f"{context}: {_SDK_HELP}")


def _info_value(info: Any, key: str) -> str:
    """Read one transport-layer property of a device, ``""`` when absent."""
    try:
        if not info.GetPropertyAvailable(key):
            return ""
        value = info.GetPropertyValue(key)
    except Exception:  # property sets vary per transport layer
        return ""
    if isinstance(value, tuple):  # some pypylon builds return (ok, value)
        return str(value[1]) if value[0] else ""
    return str(value)


def describe_device(info: Any) -> dict[str, str]:
    """Flatten a pylon ``DeviceInfo`` into printable fields."""
    return {
        "serial": info.GetSerialNumber(),
        "model": info.GetModelName(),
        "user_name": info.GetUserDefinedName(),
        "device_class": info.GetDeviceClass(),
        "ip": _info_value(info, "IpAddress"),
        "mac": _info_value(info, "MacAddress"),
        "friendly_name": info.GetFriendlyName(),
    }


def enumerate_devices(gige_only: bool = False) -> list[dict[str, str]]:
    """List the Basler cameras this PC can see, in enumeration order.

    Args:
        gige_only: keep only the cameras reached over Ethernet.

    Raises:
        CameraConnectionError: pypylon is not installed.
    """
    _require_sdk("Basler")
    infos = pylon.TlFactory.GetInstance().EnumerateDevices()
    devices = [describe_device(info) for info in infos]
    if gige_only:
        devices = [d for d in devices if d["device_class"] == GIGE_DEVICE_CLASS]
    return devices


class BaslerCamera(CameraBase):
    """Adapter for Basler GigE Vision / USB3 Vision cameras via pypylon."""

    def __init__(self, settings: CameraSettings) -> None:
        super().__init__(settings)
        self._camera: Any = None
        self._converter: Any = None
        self._options: dict[str, Any] = self._read_options(settings)

    # ---------------------------------------------------------- driver hooks
    def _connect_device(self) -> None:
        _require_sdk(self.name)
        self._options = self._read_options(self._settings)

        info = self._find_device()
        described = describe_device(info)
        try:
            camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
            camera.Open()
        except genicam.GenericException as exc:
            raise CameraConnectionError(
                f"{self.name}: cannot open {described['friendly_name']}: {exc}"
            ) from exc

        self._camera = camera
        camera.MaxNumBuffer = int(self._options.get("num_buffers", DEFAULT_NUM_BUFFERS))

        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        if described["device_class"] == GIGE_DEVICE_CLASS:
            self._tune_gige_link()
        logger.info(
            "%s: opened %s sn=%s ip=%s (%s)",
            self.name,
            described["model"],
            described["serial"],
            described["ip"] or "n/a",
            described["device_class"],
        )

    def _disconnect_device(self) -> None:
        camera, self._camera, self._converter = self._camera, None, None
        if camera is None:
            return
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()

    def _detect_resolution(self) -> tuple[int, int]:
        """Sensor's native (max) resolution — WidthMax/HeightMax — independent
        of whatever Width/Height are currently configured to."""
        width = self._read_int("WidthMax")
        height = self._read_int("HeightMax")
        if width is None or height is None:
            raise CameraConfigurationError(
                f"{self.name}: camera does not report WidthMax/HeightMax"
            )
        return width, height

    def _apply_to_device(self, settings: CameraSettings) -> None:
        camera = self._camera
        if camera is None:
            raise CameraConfigurationError(f"{self.name}: camera is not open")
        self._options = self._read_options(settings)

        if camera.IsGrabbing():
            camera.StopGrabbing()  # geometry and trigger nodes lock while grabbing
        try:
            self._apply_pixel_format()
            self._apply_geometry(settings)
            self._apply_exposure_gain(settings)
            self._apply_trigger(settings)
        except genicam.GenericException as exc:
            raise CameraConfigurationError(
                f"{self.name}: device rejected settings: {exc}"
            ) from exc
        self._start_grabbing(settings)

    def _grab(self) -> np.ndarray:
        camera = self._camera
        if camera is None or not camera.IsGrabbing():
            raise CameraCaptureError(f"{self.name}: acquisition is not running")
        timeout_ms = int(self._options.get("grab_timeout_ms", DEFAULT_GRAB_TIMEOUT_MS))

        try:
            if self._settings.trigger_mode is TriggerMode.SOFTWARE:
                self._flush_pending()
                if not camera.WaitForFrameTriggerReady(timeout_ms, pylon.TimeoutHandling_Return):
                    raise CameraCaptureError(
                        f"{self.name}: camera not ready for a software trigger "
                        f"within {timeout_ms} ms"
                    )
                camera.ExecuteSoftwareTrigger()
            result = camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_Return)
        except genicam.GenericException as exc:
            raise CameraCaptureError(f"{self.name}: grab failed: {exc}") from exc

        if result is None or not result.IsValid():
            raise CameraCaptureError(f"{self.name}: {self._timeout_hint(timeout_ms)}")
        try:
            if not result.GrabSucceeded():
                raise CameraCaptureError(
                    f"{self.name}: grab error 0x{result.GetErrorCode():08x} "
                    f"({result.GetErrorDescription()})"
                )
            return self._to_frame(result)
        finally:
            result.Release()

    # ------------------------------------------------------ device selection
    def _find_device(self) -> Any:
        """Pick the configured camera out of the enumeration.

        ``connection_id`` matches a serial number, IP address or user-defined
        name; a bare number is used as an enumeration index when nothing
        matched, which keeps ``"0"`` working for a single-camera station.
        """
        infos = list(pylon.TlFactory.GetInstance().EnumerateDevices())
        device_class = str(self._options.get("device_class", "")).strip()
        if device_class:
            infos = [i for i in infos if i.GetDeviceClass() == device_class]
        if not infos:
            raise CameraConnectionError(
                f"{self.name}: no Basler camera found. Check the Ethernet cable, "
                f"that the camera holds an IP on this subnet (pylon IP Configurator), "
                f"and that the firewall allows pylon."
            )

        wanted = self._settings.connection_id.strip()
        if not wanted:
            gige = [i for i in infos if i.GetDeviceClass() == GIGE_DEVICE_CLASS]
            return (gige or infos)[0]

        for info in infos:
            described = describe_device(info)
            if wanted in (described["serial"], described["ip"], described["user_name"]):
                return info
        if wanted.isdigit() and int(wanted) < len(infos):
            return infos[int(wanted)]

        visible = ", ".join(
            f"{d['model']} sn={d['serial']} ip={d['ip'] or 'n/a'}"
            for d in (describe_device(i) for i in infos)
        )
        raise CameraConnectionError(
            f"{self.name}: no Basler camera matches connection_id {wanted!r}. "
            f"Visible: {visible}"
        )

    def _timeout_hint(self, timeout_ms: int) -> str:
        """Timeout message that names the likely cause for this trigger mode."""
        if self._settings.trigger_mode is TriggerMode.HARDWARE:
            source = self._options.get("trigger_source", "Line1")
            return f"no frame within {timeout_ms} ms — no trigger signal on {source}?"
        return (
            f"no frame within {timeout_ms} ms — check the link, the packet size "
            f"and the exposure time"
        )

    # -------------------------------------------------------------- settings
    @staticmethod
    def _read_options(settings: CameraSettings) -> dict[str, Any]:
        """Driver-specific ``basler`` block of the camera.json entry."""
        block = settings.extra.get("basler", {})
        return dict(block) if isinstance(block, dict) else {}

    def _tune_gige_link(self) -> None:
        """Packet size and delays — what separates a stable link from dropped frames."""
        options = self._options
        self._write_int(int(options.get("packet_size", DEFAULT_PACKET_SIZE)), "GevSCPSPacketSize")
        self._write_int(int(options.get("inter_packet_delay", 0)), "GevSCPD")
        if "frame_transmission_delay" in options:
            self._write_int(int(options["frame_transmission_delay"]), "GevSCFTD")
        if "heartbeat_timeout_ms" in options:
            self._write_int(int(options["heartbeat_timeout_ms"]), "GevHeartbeatTimeout")

    def _apply_pixel_format(self) -> None:
        pixel_format = str(self._options.get("pixel_format", "")).strip()
        if pixel_format:
            self._write_enum(pixel_format, "PixelFormat")

    def _apply_geometry(self, settings: CameraSettings) -> None:
        """Sensor window: zero the offsets first, so Width/Height see their full range."""
        self._write_int(0, "OffsetX")
        self._write_int(0, "OffsetY")
        self._write_int(settings.width, "Width")
        self._write_int(settings.height, "Height")
        self._write_int(int(self._options.get("offset_x", 0)), "OffsetX")
        self._write_int(int(self._options.get("offset_y", 0)), "OffsetY")

    def _apply_exposure_gain(self, settings: CameraSettings) -> None:
        self._write_enum("Off", "ExposureAuto")
        self._write_enum("Timed", "ExposureMode")
        self._write_float(float(settings.exposure_us), "ExposureTime", "ExposureTimeAbs")

        self._write_enum("Off", "GainAuto")
        if not self._write_float(settings.gain_db, "Gain", "GainAbs"):
            # SFNC 1.x GigE exposes gain only as raw device units.
            self._write_int(int(round(settings.gain_db)), "GainRaw")

        self._write_bool(True, "GammaEnable")
        self._write_enum("User", "GammaSelector")
        self._write_float(settings.gamma, "Gamma")
        # settings.brightness is a PLC-driven external light level, not an
        # in-camera setting — see CameraSettings.brightness and CameraService.

    def _apply_trigger(self, settings: CameraSettings) -> None:
        self._write_enum("Continuous", "AcquisitionMode")
        self._write_enum("FrameStart", "TriggerSelector")

        if settings.trigger_mode is TriggerMode.CONTINUOUS:
            self._write_enum("Off", "TriggerMode")
            frame_rate = float(self._options.get("frame_rate", 0.0))
            self._write_bool(frame_rate > 0.0, "AcquisitionFrameRateEnable")
            if frame_rate > 0.0:
                self._write_float(frame_rate, "AcquisitionFrameRate", "AcquisitionFrameRateAbs")
            return

        self._write_bool(False, "AcquisitionFrameRateEnable")
        if settings.trigger_mode is TriggerMode.SOFTWARE:
            self._write_enum("Software", "TriggerSource")
        else:
            self._write_enum(str(self._options.get("trigger_source", "Line1")), "TriggerSource")
            self._write_enum(
                str(self._options.get("trigger_activation", "RisingEdge")), "TriggerActivation"
            )
        self._write_enum("On", "TriggerMode")

    def _start_grabbing(self, settings: CameraSettings) -> None:
        """Software trigger grabs strictly in order; the others take the newest frame."""
        strategy = (
            pylon.GrabStrategy_OneByOne
            if settings.trigger_mode is TriggerMode.SOFTWARE
            else pylon.GrabStrategy_LatestImageOnly
        )
        try:
            self._camera.StartGrabbing(strategy)
        except genicam.GenericException as exc:
            raise CameraConfigurationError(
                f"{self.name}: could not start acquisition: {exc}"
            ) from exc

    # -------------------------------------------------------------- grabbing
    def _flush_pending(self) -> None:
        """Drop frames queued before this trigger, so ``capture()`` is never stale."""
        while True:
            result = self._camera.RetrieveResult(0, pylon.TimeoutHandling_Return)
            if result is None or not result.IsValid():
                return
            result.Release()

    def _to_frame(self, result: Any) -> np.ndarray:
        """Grab result to an owned ndarray (BGR, or 2-D mono in ``native`` mode)."""
        native = str(self._options.get("color_conversion", "bgr8")).lower() == "native"
        if native or self._converter.ImageHasDestinationFormat(result):
            return result.GetArray().copy()  # copy: the buffer goes back to the queue
        converted = self._converter.Convert(result)
        try:
            return converted.GetArray().copy()
        finally:
            converted.Release()

    # --------------------------------------------------------- node plumbing
    def _node(self, *names: str) -> Any:
        """First node of ``names`` this model exposes, or ``None``."""
        for name in names:
            try:
                node = getattr(self._camera, name)
            except Exception:  # absent on this model / SFNC version
                continue
            if genicam.IsAvailable(node):
                return node
        return None

    def _writable(self, names: tuple[str, ...]) -> Any:
        node = self._node(*names)
        if node is None or not genicam.IsWritable(node):
            logger.debug("%s: node %s not writable, skipped", self.name, "/".join(names))
            return None
        return node

    def _write_float(self, value: float, *names: str) -> bool:
        """Write a float node, clamped to its range. False when unsupported."""
        node = self._writable(names)
        if node is None:
            return False
        node.SetValue(min(max(float(value), node.GetMin()), node.GetMax()))
        return True

    def _read_int(self, *names: str) -> int | None:
        """Current value of an integer node, or ``None`` when unavailable."""
        node = self._node(*names)
        return int(node.GetValue()) if node is not None else None

    def _write_int(self, value: int, *names: str) -> bool:
        """Write an integer node, clamped to its range and snapped to its increment."""
        node = self._writable(names)
        if node is None:
            return False
        low, high, inc = node.GetMin(), node.GetMax(), max(1, node.GetInc())
        value = min(max(int(value), low), high)
        node.SetValue(low + ((value - low) // inc) * inc)
        return True

    def _write_enum(self, value: str, *names: str) -> bool:
        """Select an enum entry, skipping values this model does not offer."""
        node = self._writable(names)
        if node is None:
            return False
        if value not in node.Symbolics:
            logger.debug(
                "%s: %s has no entry %r (offers %s)",
                self.name,
                "/".join(names),
                value,
                ", ".join(node.Symbolics),
            )
            return False
        node.SetValue(value)
        return True

    def _write_bool(self, value: bool, *names: str) -> bool:
        node = self._writable(names)
        if node is None:
            return False
        node.SetValue(bool(value))
        return True


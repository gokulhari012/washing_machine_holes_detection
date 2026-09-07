"""Abstract camera interface (adapter + template method).

Concrete drivers implement the four ``_device`` hooks; the public API
(:meth:`connect`, :meth:`capture`, :meth:`apply_settings`) lives here so
every driver gets identical locking, ROI cropping, timing and error
translation for free.

Thread safety: ``capture()`` serialises on an internal lock because two
threads legitimately grab from the same camera — the live-preview
acquisition worker and the inspection pipeline.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.logging import get_logger
from core.utilities.enums import CameraDriver, LogSource, TriggerMode
from core.utilities.exceptions import (
    CameraCaptureError,
    CameraConfigurationError,
    CameraConnectionError,
    ConfigurationError,
)

logger = get_logger(LogSource.CAMERA)


# Rate assumed for a camera whose ``fps`` is missing or nonsensical — an
# entry written before ``fps`` existed, or hand-edited to 0. It is a guard
# against a divide-by-zero, not a cadence any call site is meant to rely on:
# every continuous view is paced by the camera's own configured rate.
DEFAULT_VIEW_FPS = 4.0


def frame_interval_ms(fps: float) -> int:
    """Loop/timer interval in ms for a continuous view running at ``fps``.

    Every continuous viewing mode in the app — live preview, the Camera
    page's Continuous Capture, Auto Calibrate's board scan — sizes its
    cadence through here from :attr:`CameraSettings.fps`, so the one
    Camera-tab setting drives all of them and no site keeps a fixed rate of
    its own. A missing/zero rate falls back to :data:`DEFAULT_VIEW_FPS`.
    """
    if fps <= 0:
        fps = DEFAULT_VIEW_FPS
    return max(1, int(round(1000.0 / fps)))


@dataclass
class CameraSettings:
    """Driver-independent camera parameters (mirrors one entry of camera.json)."""

    index: int
    name: str
    driver: CameraDriver
    connection_id: str = ""
    enabled: bool = True
    exposure_us: int = 10000
    gain_db: float = 0.0
    gamma: float = 1.0
    # 0-255 light-brightness level for an external, PLC-controlled light
    # source — not an in-camera image adjustment. No driver applies this to
    # the device or the captured image; CameraService pushes it to that
    # camera's PLC brightness register on every apply/save instead (see
    # core.plc.register_map.RegisterMap.camera_brightness).
    brightness: int = 0
    # Frames per second requested from this camera by every *continuous*
    # viewing mode — live preview, the Camera page's Continuous Capture and
    # the Calibration page's Auto Calibrate scan all pace themselves by this
    # one value (see :func:`frame_interval_ms`). It throttles how often the
    # app asks for a frame; it does not program an acquisition frame rate
    # into the device. Whether the *background* preview threads run at all is
    # a separate station-wide switch (``app_config.ui.live_preview_fps``);
    # this is the rate they use once they do.
    fps: float = DEFAULT_VIEW_FPS
    width: int = 1280
    height: int = 1024
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h; w/h 0 = full frame
    trigger_mode: TriggerMode = TriggerMode.SOFTWARE
    extra: dict[str, Any] = field(default_factory=dict)  # driver-specific keys

    @classmethod
    def from_config(cls, cfg: dict) -> "CameraSettings":
        """Build from one ``cameras[]`` entry of camera.json.

        Raises:
            ConfigurationError: required keys missing or invalid.
        """
        try:
            roi_cfg = cfg.get("roi", {})
            return cls(
                index=int(cfg["index"]),
                name=str(cfg.get("name", f"Camera {cfg['index']}")),
                driver=CameraDriver(str(cfg["driver"]).lower()),
                connection_id=str(cfg.get("connection_id", "")),
                enabled=bool(cfg.get("enabled", True)),
                exposure_us=int(cfg.get("exposure_us", 10000)),
                gain_db=float(cfg.get("gain_db", 0.0)),
                gamma=float(cfg.get("gamma", 1.0)),
                brightness=int(cfg.get("brightness", 0)),
                fps=float(cfg.get("fps", DEFAULT_VIEW_FPS)),
                width=int(cfg.get("width", 1280)),
                height=int(cfg.get("height", 1024)),
                roi=(
                    int(roi_cfg.get("x", 0)),
                    int(roi_cfg.get("y", 0)),
                    int(roi_cfg.get("width", 0)),
                    int(roi_cfg.get("height", 0)),
                ),
                trigger_mode=TriggerMode(str(cfg.get("trigger_mode", "software")).lower()),
                extra=dict(cfg),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ConfigurationError(f"Invalid camera configuration: {exc}") from exc


class CameraBase(ABC):
    """Contract + shared behaviour for every camera driver."""

    def __init__(self, settings: CameraSettings) -> None:
        self._settings = settings
        self._connected = False
        self._capture_lock = threading.Lock()
        self.last_capture_ms: float = 0.0  # duration of the most recent grab

    # ------------------------------------------------------------ properties
    @property
    def index(self) -> int:
        return self._settings.index

    @property
    def name(self) -> str:
        return self._settings.name

    @property
    def settings(self) -> CameraSettings:
        return self._settings

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------ public API
    def connect(self) -> None:
        """Open the device and push the current settings.

        Raises:
            CameraConnectionError
        """
        try:
            self._connect_device()
            self._apply_to_device(self._settings)
        except CameraConnectionError:
            raise
        except Exception as exc:  # driver SDKs raise their own types
            raise CameraConnectionError(f"{self.name}: connect failed: {exc}") from exc
        self._connected = True
        logger.info("%s connected (%s)", self.name, self._settings.driver.value)

    def disconnect(self) -> None:
        """Close the device. Safe to call repeatedly."""
        try:
            self._disconnect_device()
        except Exception:  # closing a dead device must never raise
            logger.exception("%s: error during disconnect (ignored)", self.name)
        finally:
            self._connected = False
            logger.info("%s disconnected", self.name)

    def capture(self) -> np.ndarray:
        """Grab one frame (BGR or mono ndarray), ROI-cropped if configured.

        Raises:
            CameraConnectionError: camera not connected.
            CameraCaptureError: the grab failed or timed out.
        """
        with self._capture_lock:
            if not self._connected:
                raise CameraConnectionError(f"{self.name} is not connected")
            started = time.perf_counter()
            try:
                frame = self._grab()
            except CameraCaptureError:
                raise
            except Exception as exc:
                raise CameraCaptureError(f"{self.name}: capture failed: {exc}") from exc
            self.last_capture_ms = (time.perf_counter() - started) * 1000.0

        if frame is None or frame.size == 0:
            raise CameraCaptureError(f"{self.name}: empty frame")
        return self._crop_roi(frame)

    def apply_settings(self, settings: CameraSettings) -> None:
        """Adopt new settings; pushed to the device immediately when connected.

        Raises:
            CameraConfigurationError: the device rejected a parameter.
        """
        self._settings = settings
        if self._connected:
            self._apply_to_device(settings)
            logger.info("%s: settings applied", self.name)

    def detect_resolution(self) -> tuple[int, int]:
        """Actual (width, height) reported by the device or file source,
        independent of whatever Width/Height are currently configured to.

        Raises:
            CameraConnectionError: camera not connected.
            CameraConfigurationError: this driver has none to report.
        """
        if not self._connected:
            raise CameraConnectionError(f"{self.name} is not connected")
        return self._detect_resolution()

    # ---------------------------------------------------------- driver hooks
    @abstractmethod
    def _connect_device(self) -> None:
        """Open the physical/virtual device."""

    @abstractmethod
    def _disconnect_device(self) -> None:
        """Release the device."""

    @abstractmethod
    def _grab(self) -> np.ndarray:
        """Acquire and return one frame."""

    @abstractmethod
    def _apply_to_device(self, settings: CameraSettings) -> None:
        """Push exposure/gain/gamma/resolution/trigger-mode to the device."""

    def _detect_resolution(self) -> tuple[int, int]:
        """Driver hook for :meth:`detect_resolution`; default: nothing to report."""
        raise CameraConfigurationError(
            f"{self.name}: {self._settings.driver.value} cameras have no "
            f"resolution to auto-detect — set Width/Height manually"
        )

    # -------------------------------------------------------------- internal
    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """Software ROI crop (drivers with hardware ROI may pre-crop instead)."""
        x, y, w, h = self._settings.roi
        if w <= 0 or h <= 0:
            return frame
        frame_h, frame_w = frame.shape[:2]
        x0 = max(0, min(x, frame_w - 1))
        y0 = max(0, min(y, frame_h - 1))
        x1 = max(x0 + 1, min(x + w, frame_w))
        y1 = max(y0 + 1, min(y + h, frame_h))
        return frame[y0:y1, x0:x1]

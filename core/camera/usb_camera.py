"""USB / UVC camera adapter built on ``cv2.VideoCapture``.

Suitable for webcams and USB machine-vision cameras exposing a DirectShow
interface. ``connection_id`` in camera.json is the numeric device index.

Note on exposure: UVC exposure control is driver-dependent. OpenCV's
``CAP_PROP_EXPOSURE`` on DirectShow expects log2(seconds) (e.g. -13 ≈ 122 µs);
the conversion below is best-effort and harmless where unsupported.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from core.camera.camera_base import CameraBase, CameraSettings
from core.utilities.exceptions import CameraCaptureError, CameraConnectionError


class UsbCamera(CameraBase):
    """OpenCV VideoCapture implementation of :class:`CameraBase`."""

    GRAB_RETRIES = 2

    def __init__(self, settings: CameraSettings) -> None:
        super().__init__(settings)
        self._capture: cv2.VideoCapture | None = None

    # ---------------------------------------------------------- driver hooks
    def _connect_device(self) -> None:
        try:
            device_index = int(self._settings.connection_id or self._settings.index - 1)
        except ValueError as exc:
            raise CameraConnectionError(
                f"{self.name}: connection_id must be a device index, "
                f"got {self._settings.connection_id!r}"
            ) from exc

        capture = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            raise CameraConnectionError(f"{self.name}: no device at index {device_index}")
        self._capture = capture

    def _disconnect_device(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _apply_to_device(self, settings: CameraSettings) -> None:
        assert self._capture is not None
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        self._capture.set(cv2.CAP_PROP_BRIGHTNESS, settings.brightness)
        self._capture.set(cv2.CAP_PROP_GAIN, settings.gain_db)
        self._capture.set(cv2.CAP_PROP_GAMMA, settings.gamma * 100.0)  # UVC gamma is x100
        if settings.exposure_us > 0:
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual mode (DirectShow)
            self._capture.set(
                cv2.CAP_PROP_EXPOSURE, math.log2(settings.exposure_us / 1_000_000.0)
            )

    def _grab(self) -> np.ndarray:
        assert self._capture is not None
        for _ in range(self.GRAB_RETRIES + 1):
            ok, frame = self._capture.read()
            if ok and frame is not None and frame.size:
                return frame
        raise CameraCaptureError(f"{self.name}: VideoCapture.read() failed")

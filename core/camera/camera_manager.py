"""Lifecycle, health and parallel capture for the 4-camera station.

The manager owns one :class:`CameraBase` per configured camera, tracks a
:class:`CameraHealth` record for the dashboard/status indicators, and fans
state changes out to Qt-free observer callbacks (the UI bridges them to
signals).

``capture_all()`` grabs every connected camera in parallel and **returns
partial results** — one dead camera must not abort the other three; the
inspection service decides how a missing frame affects the overall result.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from core.camera.camera_base import CameraBase, CameraSettings
from core.logging import get_logger
from core.utilities.enums import ConnectionState, LogSource
from core.utilities.exceptions import CameraError

logger = get_logger(LogSource.CAMERA)

StateCallback = "callable[[int, ConnectionState], None]"


@dataclass
class CameraHealth:
    """Rolling health record per camera (drives the UI health indicator)."""

    connected: bool = False
    last_ok_monotonic: float | None = None
    last_error: str = ""
    frames_captured: int = 0
    capture_failures: int = 0

    @property
    def healthy(self) -> bool:
        return self.connected and not self.last_error


class CameraManager:
    """Builds, connects and supervises all configured cameras."""

    def __init__(self, camera_configs: list[dict]) -> None:
        from core.camera import create_camera  # local import: factory lives in package root

        self._cameras: dict[int, CameraBase] = {}
        self._health: dict[int, CameraHealth] = {}
        self._callbacks: list = []
        self._lock = threading.Lock()

        for cfg in camera_configs:
            settings = CameraSettings.from_config(cfg)
            self._cameras[settings.index] = create_camera(settings)
            self._health[settings.index] = CameraHealth()

    # ------------------------------------------------------------ inventory
    @property
    def cameras(self) -> dict[int, CameraBase]:
        return dict(self._cameras)

    def get(self, index: int) -> CameraBase:
        try:
            return self._cameras[index]
        except KeyError:
            raise CameraError(f"No camera with index {index}") from None

    def subscribe_state(self, callback) -> None:
        """Register ``callback(camera_index, ConnectionState)`` (any thread)."""
        with self._lock:
            self._callbacks.append(callback)

    # ------------------------------------------------------------ lifecycle
    def connect(self, index: int) -> None:
        """Connect one camera. Raises CameraConnectionError on failure."""
        camera = self.get(index)
        try:
            camera.connect()
        except CameraError as exc:
            self._update_health(index, connected=False, error=str(exc))
            raise
        self._update_health(index, connected=True, error="")

    def disconnect(self, index: int) -> None:
        self.get(index).disconnect()
        self._update_health(index, connected=False, error="")

    def connect_all(self) -> dict[int, str]:
        """Connect every enabled camera; returns {index: error} for failures."""
        errors: dict[int, str] = {}
        for index, camera in self._cameras.items():
            if not camera.settings.enabled:
                continue
            try:
                self.connect(index)
            except CameraError as exc:
                errors[index] = str(exc)
                logger.error("Camera %d failed to connect: %s", index, exc)
        return errors

    def disconnect_all(self) -> None:
        for index in self._cameras:
            self.disconnect(index)

    # -------------------------------------------------------------- capture
    def capture(self, index: int) -> np.ndarray:
        """Grab one frame from one camera; updates health; re-raises CameraError."""
        camera = self.get(index)
        try:
            frame = camera.capture()
        except CameraError as exc:
            self._record_failure(index, str(exc))
            raise
        self._record_success(index)
        return frame

    def capture_all(
        self, indexes: list[int] | None = None
    ) -> dict[int, np.ndarray | None]:
        """Parallel grab from the given (default: all enabled+connected) cameras.

        Returns {index: frame or None-on-failure}; never raises for a single
        camera fault — failures are logged and recorded in health.
        """
        if indexes is None:
            indexes = [
                index
                for index, camera in self._cameras.items()
                if camera.settings.enabled and camera.connected
            ]
        if not indexes:
            return {}

        def _safe_capture(index: int) -> np.ndarray | None:
            try:
                return self.capture(index)
            except CameraError:
                return None  # already logged + health-recorded by capture()

        with ThreadPoolExecutor(max_workers=len(indexes), thread_name_prefix="grab") as pool:
            frames = list(pool.map(_safe_capture, indexes))
        return dict(zip(indexes, frames))

    # ---------------------------------------------------------------- health
    def health(self, index: int) -> CameraHealth:
        return self._health[index]

    def all_health(self) -> dict[int, CameraHealth]:
        return dict(self._health)

    # ---------------------------------------------------- settings / rebuild
    def apply_settings(self, index: int, settings: CameraSettings) -> None:
        """Push new settings to one camera (Camera Configuration page)."""
        self.get(index).apply_settings(settings)

    def rebuild(self, camera_configs: list[dict]) -> None:
        """Tear down and rebuild all cameras after a configuration change."""
        from core.camera import create_camera

        self.disconnect_all()
        self._cameras.clear()
        self._health.clear()
        for cfg in camera_configs:
            settings = CameraSettings.from_config(cfg)
            self._cameras[settings.index] = create_camera(settings)
            self._health[settings.index] = CameraHealth()
        logger.info("Camera manager rebuilt with %d cameras", len(self._cameras))

    # -------------------------------------------------------------- internal
    def _record_success(self, index: int) -> None:
        health = self._health[index]
        health.frames_captured += 1
        health.last_ok_monotonic = time.monotonic()
        if health.last_error:
            health.last_error = ""
            self._notify(index, ConnectionState.CONNECTED)  # recovered

    def _record_failure(self, index: int, error: str) -> None:
        health = self._health[index]
        health.capture_failures += 1
        first_failure = not health.last_error
        health.last_error = error
        logger.error("Camera %d capture failure: %s", index, error)
        if first_failure:
            self._notify(index, ConnectionState.ERROR)

    def _update_health(self, index: int, *, connected: bool, error: str) -> None:
        health = self._health[index]
        health.connected = connected
        health.last_error = error
        state = (
            ConnectionState.CONNECTED
            if connected
            else (ConnectionState.ERROR if error else ConnectionState.DISCONNECTED)
        )
        self._notify(index, state)

    def _notify(self, index: int, state: ConnectionState) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(index, state)
            except Exception:  # observers must never break acquisition
                logger.exception("Camera state callback raised")

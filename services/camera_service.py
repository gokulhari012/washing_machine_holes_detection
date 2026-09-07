"""Camera facade for the UI: add/remove/edit cameras, live control, tests.

Owns the persistence of camera.json (plus the DB audit mirror) and delegates
device operations to the :class:`CameraManager`. The acquisition workers are
rebuilt by the composition root when the configuration is saved (via
``ConfigManager.subscribe("camera", ...)``).

``brightness`` (0-255) is a light-brightness level for an external,
PLC-controlled light source, not an in-camera image adjustment — every apply/
save also pushes it to that camera's PLC brightness register (best-effort:
a PLC communication failure is logged, not raised, so it never blocks the
camera settings themselves from applying — see :meth:`_push_brightness`).
"""

from __future__ import annotations

import numpy as np

from core.camera import DEFAULT_VIEW_FPS, CameraHealth, CameraManager, CameraSettings
from core.logging import get_logger
from core.utilities import ConfigManager
from core.utilities.enums import ConnectionState, LogSource
from core.utilities.exceptions import CameraError, ConfigurationError, PlcError
from services.database_service import DatabaseService
from services.plc_service import PlcService

logger = get_logger(LogSource.CAMERA)


class CameraService:
    """Everything the Camera Configuration page needs."""

    def __init__(
        self,
        camera_manager: CameraManager,
        config_manager: ConfigManager,
        database_service: DatabaseService,
        plc_service: PlcService,
    ) -> None:
        self._manager = camera_manager
        self._config = config_manager
        self._database = database_service
        self._plc = plc_service

    # -------------------------------------------------------------- queries
    def get_configs(self) -> list[dict]:
        return self._config.load("camera").get("cameras", [])

    def camera_fps(self, index: int) -> float:
        """Configured frame rate for camera ``index``, as set on the Camera page.

        The cadence every continuous viewing mode paces itself by — the
        Camera page's Continuous Capture and the Calibration page's Auto
        Calibrate scan both size their loop from this (via
        :func:`core.camera.frame_interval_ms`), and it is the rate the live
        preview workers run at too. Falls back to :data:`DEFAULT_VIEW_FPS`
        for an unknown camera or an entry predating the setting.
        """
        for cfg in self.get_configs():
            if int(cfg.get("index", -1)) == index:
                return float(cfg.get("fps", DEFAULT_VIEW_FPS) or DEFAULT_VIEW_FPS)
        return DEFAULT_VIEW_FPS

    def health(self, index: int) -> CameraHealth:
        return self._manager.health(index)

    def all_health(self) -> dict[int, CameraHealth]:
        return self._manager.all_health()

    def connection_states(self) -> dict[int, ConnectionState]:
        return {
            index: (
                ConnectionState.CONNECTED if camera.connected else ConnectionState.DISCONNECTED
            )
            for index, camera in self._manager.cameras.items()
        }

    # ------------------------------------------------------------ lifecycle
    def connect(self, index: int) -> None:
        """Raises CameraConnectionError on failure."""
        self._manager.connect(index)

    def disconnect(self, index: int) -> None:
        self._manager.disconnect(index)

    def connect_all(self) -> dict[int, str]:
        return self._manager.connect_all()

    def test_capture(self, index: int) -> np.ndarray:
        """One frame for the 'Test Camera' button. Raises CameraError."""
        return self._manager.capture(index)

    def detect_resolution(self, index: int) -> tuple[int, int]:
        """(width, height) for the 'Detect Resolution' button. Raises CameraError."""
        return self._manager.detect_resolution(index)

    # --------------------------------------------------------- configuration
    def apply_live(self, camera_config: dict) -> None:
        """Push settings to a connected camera without persisting (preview tuning).

        Also pushes ``brightness`` to that camera's PLC light-brightness
        register (best-effort — see :meth:`_push_brightness`).

        Raises:
            ConfigurationError | CameraConfigurationError
        """
        settings = CameraSettings.from_config(camera_config)
        self._manager.apply_settings(settings.index, settings)
        self._push_brightness(settings)

    def save_camera(self, camera_config: dict) -> None:
        """Insert-or-update one camera entry in camera.json + DB mirror.

        Also pushes ``brightness`` to that camera's PLC light-brightness
        register (best-effort — see :meth:`_push_brightness`).

        Raises:
            ConfigurationError: entry malformed.
        """
        settings = CameraSettings.from_config(camera_config)  # validate first
        document = self._config.load("camera")
        cameras = document.setdefault("cameras", [])
        for position, entry in enumerate(cameras):
            if int(entry.get("index", -1)) == settings.index:
                cameras[position] = camera_config
                break
        else:
            cameras.append(camera_config)
            cameras.sort(key=lambda entry: int(entry.get("index", 0)))
        self._config.save("camera", document)
        self._mirror_to_database(settings)
        self._push_brightness(settings)
        logger.info("Camera %d configuration saved", settings.index)

    def remove_camera(self, index: int) -> None:
        document = self._config.load("camera")
        cameras = document.get("cameras", [])
        remaining = [entry for entry in cameras if int(entry.get("index", -1)) != index]
        if len(remaining) == len(cameras):
            raise ConfigurationError(f"No camera with index {index} to remove")
        document["cameras"] = remaining
        self._config.save("camera", document)
        self._database.camera_configs.delete_by_index(index)
        logger.info("Camera %d removed from configuration", index)

    # -------------------------------------------------------------- internal
    def _push_brightness(self, settings: CameraSettings) -> None:
        """Write this camera's brightness to its PLC register, if configured.

        Best-effort: a communication failure is logged and swallowed rather
        than raised, so a PLC hiccup (or a station whose light isn't wired to
        the PLC at all — ``set_camera_brightness`` then just returns False)
        never blocks the camera settings themselves from applying/saving.
        """
        try:
            self._plc.set_camera_brightness(settings.index, settings.brightness)
        except PlcError as exc:
            logger.warning(
                "Camera %d: brightness not pushed to PLC (%s)", settings.index, exc
            )

    def _mirror_to_database(self, settings: CameraSettings) -> None:
        self._database.camera_configs.upsert(
            {
                "camera_index": settings.index,
                "name": settings.name,
                "driver": settings.driver.value,
                "connection_id": settings.connection_id,
                "enabled": settings.enabled,
                "exposure_us": settings.exposure_us,
                "gain_db": settings.gain_db,
                "gamma": settings.gamma,
                "brightness": settings.brightness,
                "fps": settings.fps,
                "rotation": settings.rotation,
                "width": settings.width,
                "height": settings.height,
                "roi_x": settings.roi[0],
                "roi_y": settings.roi[1],
                "roi_width": settings.roi[2],
                "roi_height": settings.roi[3],
                "trigger_mode": settings.trigger_mode.value,
            }
        )

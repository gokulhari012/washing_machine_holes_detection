"""Camera facade for the UI: add/remove/edit cameras, live control, tests.

Owns the persistence of camera.json (plus the DB audit mirror) and delegates
device operations to the :class:`CameraManager`. The acquisition workers are
rebuilt by the composition root when the configuration is saved (via
``ConfigManager.subscribe("camera", ...)``).
"""

from __future__ import annotations

import numpy as np

from core.camera import CameraHealth, CameraManager, CameraSettings
from core.logging import get_logger
from core.utilities import ConfigManager
from core.utilities.enums import ConnectionState, LogSource
from core.utilities.exceptions import CameraError, ConfigurationError
from services.database_service import DatabaseService

logger = get_logger(LogSource.CAMERA)


class CameraService:
    """Everything the Camera Configuration page needs."""

    def __init__(
        self,
        camera_manager: CameraManager,
        config_manager: ConfigManager,
        database_service: DatabaseService,
    ) -> None:
        self._manager = camera_manager
        self._config = config_manager
        self._database = database_service

    # -------------------------------------------------------------- queries
    def get_configs(self) -> list[dict]:
        return self._config.load("camera").get("cameras", [])

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

    # --------------------------------------------------------- configuration
    def apply_live(self, camera_config: dict) -> None:
        """Push settings to a connected camera without persisting (preview tuning).

        Raises:
            ConfigurationError | CameraConfigurationError
        """
        settings = CameraSettings.from_config(camera_config)
        self._manager.apply_settings(settings.index, settings)

    def save_camera(self, camera_config: dict) -> None:
        """Insert-or-update one camera entry in camera.json + DB mirror.

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
                "width": settings.width,
                "height": settings.height,
                "roi_x": settings.roi[0],
                "roi_y": settings.roi[1],
                "roi_width": settings.roi[2],
                "roi_height": settings.roi[3],
                "trigger_mode": settings.trigger_mode.value,
            }
        )

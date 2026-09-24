"""Camera facade for the UI: add/remove/edit cameras, live control, tests.

Owns the persistence of camera.json (plus the DB audit mirror) and delegates
device operations to the :class:`CameraManager`. The acquisition workers are
rebuilt by the composition root when the configuration is saved (via
``ConfigManager.subscribe("camera", ...)``).

``brightness`` (0-255) is a light-brightness level for an external LED light
source, not an in-camera image adjustment — every apply/save also pushes it
to the LED Controller channel named by that camera's ``led_channel`` (1-4;
0 means "not wired to a channel", so the push is simply skipped). Best-effort:
an LED communication failure is logged, not raised, so it never blocks the
camera settings themselves from applying — see :meth:`_push_brightness`.

``led_strobe`` switches that same channel to an on-only-during-capture model
instead: :meth:`light_on`/:meth:`light_off` are the caller's (the Camera page's)
responsibility to bracket around a capture or a Continuous Capture run — see
their docstrings. While strobe mode is on, :meth:`_push_brightness` is a
no-op, so an unrelated Apply/Save can't leave the light lit at rest.
"""

from __future__ import annotations

import numpy as np

from core.camera import DEFAULT_VIEW_FPS, CameraHealth, CameraManager, CameraSettings
from core.logging import get_logger
from core.utilities import ConfigManager
from core.utilities.enums import ConnectionState, LogSource
from core.utilities.exceptions import CameraError, ConfigurationError, LedError
from services.database_service import DatabaseService
from services.led_service import LedService

logger = get_logger(LogSource.CAMERA)


class CameraService:
    """Everything the Camera Configuration page needs."""

    def __init__(
        self,
        camera_manager: CameraManager,
        config_manager: ConfigManager,
        database_service: DatabaseService,
        led_service: LedService,
    ) -> None:
        self._manager = camera_manager
        self._config = config_manager
        self._database = database_service
        self._led = led_service

    # -------------------------------------------------------------- queries
    def get_configs(self) -> list[dict]:
        """The persisted camera.json entries — the manually maintained baseline."""
        return self._config.load("camera").get("cameras", [])

    def get_effective_configs(self) -> list[dict]:
        """camera.json entries with any *live* overrides merged over them.

        What the cameras are actually running right now, which is not the
        same document as camera.json: :meth:`apply_live` — used by the Camera
        page's "Apply Live" and by every machine-model switch
        (``MachineModelService.apply_profile``) — deliberately never
        persists. A UI that reads :meth:`get_configs` therefore shows the
        pre-switch baseline after a model change; anything displaying what
        the station is *doing* must read this instead.

        A camera present in the file but not in the manager (it failed to
        construct) falls back to its file entry, so the list never loses a
        row just because a device is missing.
        """
        live = self._manager.cameras
        effective: list[dict] = []
        for cfg in self.get_configs():
            camera = live.get(int(cfg.get("index", -1)))
            effective.append(camera.settings.to_config() if camera is not None else cfg)
        return effective

    def effective_config(self, index: int) -> dict | None:
        """One camera's live-effective entry, or None if it isn't configured."""
        for cfg in self.get_effective_configs():
            if int(cfg.get("index", -1)) == index:
                return cfg
        return None

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

    def test_capture(self, index: int, *, full_frame: bool = False) -> np.ndarray:
        """One frame for the 'Test Camera' button. Raises CameraError.

        ``full_frame=True`` returns the whole rotated frame, not the ROI crop
        — the Camera page draws the ROI on top of it and crops its own ROI
        view, so it can follow ROI edits that have not been applied yet.
        """
        return self._manager.capture(index, apply_roi=not full_frame)

    def detect_resolution(self, index: int) -> tuple[int, int]:
        """(width, height) for the 'Detect Resolution' button. Raises CameraError."""
        return self._manager.detect_resolution(index)

    # -------------------------------------------------------------- strobing
    def light_on(self, index: int) -> None:
        """Turn camera *index*'s configured LED channel on at its configured
        brightness — for strobe mode, called immediately before a capture
        (or once, at the start of a Continuous Capture run).

        No-op when the camera has no channel configured (``led_channel``
        <= 0) or isn't found. Best-effort: an LED communication failure is
        logged, never raised, so it can never block a capture.
        """
        self._set_channel_for(index, on=True)

    def light_off(self, index: int) -> None:
        """Turn camera *index*'s configured LED channel off — the other half
        of :meth:`light_on`, called immediately after a capture (or once, at
        the end of a Continuous Capture run). Same no-op/best-effort rules.
        """
        self._set_channel_for(index, on=False)

    def _set_channel_for(self, index: int, *, on: bool) -> None:
        cfg = self.effective_config(index)
        if cfg is None:
            return
        channel = int(cfg.get("led_channel", 0))
        if channel <= 0:
            return
        level = int(cfg.get("brightness", 0)) if on else 0
        try:
            self._led.set_channel_brightness(channel, level)
        except LedError as exc:
            logger.warning(
                "Camera %d: light_%s failed on LED channel %d (%s)",
                index, "on" if on else "off", channel, exc,
            )

    # --------------------------------------------------------- configuration
    def apply_live(self, camera_config: dict) -> None:
        """Push settings to a connected camera without persisting (preview tuning).

        Also pushes ``brightness`` to that camera's configured LED Controller
        channel (best-effort — see :meth:`_push_brightness`).

        Raises:
            ConfigurationError | CameraConfigurationError
        """
        settings = CameraSettings.from_config(camera_config)
        self._manager.apply_settings(settings.index, settings)
        self._push_brightness(settings)

    def save_camera(self, camera_config: dict) -> None:
        """Insert-or-update one camera entry in camera.json + DB mirror.

        Also pushes ``brightness`` to that camera's configured LED Controller
        channel (best-effort — see :meth:`_push_brightness`).

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
        """Write this camera's brightness to its configured LED Controller
        channel, if any.

        ``led_channel`` of 0 means this camera isn't wired to a channel, so
        there is nothing to push — same convention as an unconfigured PLC
        register elsewhere in this app. Also a no-op while ``led_strobe`` is
        on: strobe mode's resting state is *off*, and :meth:`light_on`/
        :meth:`light_off` around an actual capture are what drive the
        channel then — an Apply/Save here must not light it up outside a
        capture just because the brightness field changed. Best-effort: an
        LED communication failure (not connected, no response, ...) is
        logged and swallowed rather than raised, so it never blocks the
        camera settings themselves from applying/saving.
        """
        if settings.led_channel <= 0 or settings.led_strobe:
            return
        try:
            self._led.set_channel_brightness(settings.led_channel, settings.brightness)
        except LedError as exc:
            logger.warning(
                "Camera %d: brightness not pushed to LED channel %d (%s)",
                settings.index, settings.led_channel, exc,
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
                "led_channel": settings.led_channel,
                "led_strobe": settings.led_strobe,
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

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

:meth:`set_brightness` is a second, narrower way in - the LED Controller
page's "Camera Light Brightness" panel. It changes *only* ``brightness``,
live and in camera.json, without the full camera rebuild a
``ConfigManager.save`` of the ``camera`` domain triggers (see its docstring
for why), and tells :meth:`subscribe_brightness` observers so the Camera
page's form and the active machine model follow it. :meth:`set_strobe` is
its twin for ``led_strobe`` (the panel's per-camera strobe switches).
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from core.camera import DEFAULT_VIEW_FPS, CameraHealth, CameraManager, CameraSettings
from core.led.protocol import MAX_BRIGHTNESS, MIN_BRIGHTNESS
from core.logging import get_logger
from core.utilities import ConfigManager
from core.utilities.enums import ConnectionState, LogSource
from core.utilities.exceptions import CameraError, ConfigurationError, LedError
from services.database_service import DatabaseService
from services.led_service import LedService

logger = get_logger(LogSource.CAMERA)

#: Called with ``{camera index: brightness}`` after :meth:`CameraService.set_brightness`.
BrightnessCallback = Callable[[dict[int, int]], None]
StrobeCallback = Callable[[dict[int, bool]], None]


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
        self._brightness_callbacks: list[BrightnessCallback] = []
        self._strobe_callbacks: list[StrobeCallback] = []

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

    def set_brightness(self, levels: Mapping[int, int]) -> None:
        """Set the light brightness of one or more cameras, and nothing else.

        The LED Controller page's per-camera / "Set All" controls. The same
        ``brightness`` field as the Camera page's "Light Brightness", so it
        is what the inspection cycle's strobe lights each camera at, and
        what a steady (non-strobe) camera's channel is held at. For each
        camera, in this order:

        1. camera.json's ``brightness`` key is rewritten - that key only. The
           file is saved **without** notifying the ``camera`` subscribers,
           because the composition root's one rebuilds every camera from
           the file and reconnects them: seconds of blocked GUI and dropped
           GigE links to store a light level, and - with a machine model
           applied - it would reset the running cameras to the file's
           baseline and so undo the model's ROI/exposure;
        2. the running camera adopts it (``CameraManager.set_brightness``,
           no device push), so the next trigger cycle already uses it;
        3. it is pushed to the camera's LED channel exactly as a Camera page
           Save would (:meth:`_push_brightness`: skipped for strobe cameras
           and unwired ones, best-effort);
        4. :meth:`subscribe_brightness` observers are told, once, with every
           level set - the composition root syncs the active machine model
           and repaints the Camera page from that.

        All-or-nothing on validation: an unknown camera or an out-of-range
        level raises before anything is written.

        Raises:
            ConfigurationError: unknown camera index, brightness outside
                0-255, or camera.json could not be written.
        """
        wanted = {int(index): int(value) for index, value in levels.items()}
        if not wanted:
            return
        for index, value in wanted.items():
            if not MIN_BRIGHTNESS <= value <= MAX_BRIGHTNESS:
                raise ConfigurationError(
                    f"Camera {index}: brightness must be {MIN_BRIGHTNESS}-{MAX_BRIGHTNESS}, got {value}"
                )

        document = self._config.load("camera")
        entries = {
            int(entry.get("index", -1)): entry for entry in document.get("cameras", [])
        }
        unknown = sorted(set(wanted) - set(entries))
        if unknown:
            raise ConfigurationError(f"No camera with index {unknown[0]} is configured")
        for index, value in wanted.items():
            entries[index]["brightness"] = value
        self._config.save("camera", document, notify=False)

        for index in sorted(wanted):
            persisted = CameraSettings.from_config(entries[index])
            live = self._manager.set_brightness(index, wanted[index])
            self._mirror_to_database(persisted)
            # led_channel/led_strobe are rig facts no machine model overrides,
            # so the live and persisted entries agree on them; prefer live.
            self._push_brightness(live if live is not None else persisted)
            logger.info("Camera %d light brightness set to %d", index, wanted[index])

        for callback in list(self._brightness_callbacks):
            try:
                callback(dict(wanted))
            except Exception:  # observers must never break the caller
                logger.exception("Camera brightness callback raised")

    def subscribe_brightness(self, callback: BrightnessCallback) -> None:
        """Register a callback fired after every :meth:`set_brightness`.

        Qt-free, like ``ConfigManager.subscribe`` - the composition root
        bridges it to ``AppState.camera_brightness_changed``."""
        self._brightness_callbacks.append(callback)

    def set_strobe(self, states: Mapping[int, bool]) -> None:
        """Switch LED strobe mode on or off for one or more cameras, and
        nothing else - the LED Controller page's per-camera strobe switches
        and "Enable All"/"Disable All".

        The same ``led_strobe`` field as the Camera page's "Strobe"
        checkbox, written the same rebuild-free way as :meth:`set_brightness`
        (camera.json's ``led_strobe`` key only, saved without notifying the
        ``camera`` subscribers; then adopted live by the running camera, so
        the next trigger cycle already uses it). The channel is then moved to
        its new resting state at once, best-effort:

        - strobe **on**  -> the channel is turned off (it now lights only
          during a capture);
        - strobe **off** -> the channel is held at the camera's brightness,
          exactly as a Camera page Save would (:meth:`_push_brightness`).

        A camera with no LED channel is still switched (the setting is
        stored) but nothing is sent. ``led_strobe`` is a rig fact outside
        ``_TUNABLE_CAMERA_FIELDS``, so no machine model is touched.
        :meth:`subscribe_strobe` observers are told once, with every change.

        Raises:
            ConfigurationError: unknown camera index, or camera.json could
                not be written. Nothing is written on an unknown index.
        """
        wanted = {int(index): bool(value) for index, value in states.items()}
        if not wanted:
            return
        document = self._config.load("camera")
        entries = {
            int(entry.get("index", -1)): entry for entry in document.get("cameras", [])
        }
        unknown = sorted(set(wanted) - set(entries))
        if unknown:
            raise ConfigurationError(f"No camera with index {unknown[0]} is configured")
        for index, value in wanted.items():
            entries[index]["led_strobe"] = value
        self._config.save("camera", document, notify=False)

        for index in sorted(wanted):
            persisted = CameraSettings.from_config(entries[index])
            live = self._manager.set_strobe(index, wanted[index])
            self._mirror_to_database(persisted)
            settings = live if live is not None else persisted
            if settings.led_strobe:
                self.light_off(index)
            else:
                self._push_brightness(settings)
            logger.info(
                "Camera %d LED strobe %s", index, "enabled" if wanted[index] else "disabled"
            )

        for callback in list(self._strobe_callbacks):
            try:
                callback(dict(wanted))
            except Exception:  # observers must never break the caller
                logger.exception("Camera strobe callback raised")

    def subscribe_strobe(self, callback: StrobeCallback) -> None:
        """Register a callback fired after every :meth:`set_strobe` - the
        composition root bridges it to ``AppState.camera_strobe_changed``."""
        self._strobe_callbacks.append(callback)

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

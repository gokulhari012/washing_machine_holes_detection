"""Live-preview acquisition thread, one per camera.

Grabs frames at a throttled rate and publishes them through
``AppState.preview_frame``. The rate is that camera's own ``fps``
(camera.json, editable on the Camera Configuration page) — the same field
that paces every other continuous view of it. Whether these threads run at
all is a separate station-wide switch, ``ui.live_preview_fps``.

Grabs go through the CameraManager so health statistics stay accurate, and
the camera's internal capture lock serialises preview grabs with inspection
grabs on the same device.

Fault behaviour: on a capture error the worker backs off (2 s) instead of
hammering a dead camera; the manager has already recorded the failure and
notified the health/state observers.
"""

from __future__ import annotations

import threading
import time

from PySide6.QtCore import QThread

from core.camera import DEFAULT_VIEW_FPS, CameraManager
from core.logging import get_logger
from core.utilities.enums import LogSource
from core.utilities.exceptions import CameraError
from models.app_state import AppState

logger = get_logger(LogSource.CAMERA)

ERROR_BACKOFF_S = 2.0
DISCONNECTED_IDLE_S = 0.5


class AcquisitionWorker(QThread):
    """Continuous preview grabber for a single camera index."""

    def __init__(
        self,
        camera_manager: CameraManager,
        camera_index: int,
        app_state: AppState,
        fps: float = DEFAULT_VIEW_FPS,
    ) -> None:
        super().__init__()
        self.setObjectName(f"AcquisitionWorker-{camera_index}")
        self._manager = camera_manager
        self._index = camera_index
        self._app_state = app_state
        # No 1 fps floor: a 20 MP GigE camera is legitimately configured at
        # a fraction of a frame per second, and clamping that up would make
        # the preview thread hammer a camera the operator asked to sip from.
        self._frame_interval_s = 1.0 / max(0.01, fps)
        self._stop_event = threading.Event()

    @property
    def camera_index(self) -> int:
        return self._index

    def stop(self, timeout_ms: int = 3000) -> None:
        self._stop_event.set()
        if not self.wait(timeout_ms):
            logger.error(
                "Acquisition worker %d did not stop within %d ms",
                self._index,
                timeout_ms,
            )

    def run(self) -> None:
        logger.info("Acquisition worker %d started", self._index)
        while not self._stop_event.is_set():
            tick_started = time.monotonic()
            camera = self._manager.get(self._index)

            if not (camera.settings.enabled and camera.connected):
                self._stop_event.wait(DISCONNECTED_IDLE_S)
                continue

            try:
                frame = self._manager.capture(self._index)
            except CameraError:
                self._stop_event.wait(ERROR_BACKOFF_S)
                continue

            self._app_state.publish_preview(self._index, frame)

            elapsed = time.monotonic() - tick_started
            remaining = self._frame_interval_s - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

        logger.info("Acquisition worker %d stopped", self._index)


def create_acquisition_workers(
    camera_manager: CameraManager, app_state: AppState, preview_enabled_fps: float
) -> list[AcquisitionWorker]:
    """One worker per configured camera (composition-root helper).

    ``preview_enabled_fps`` is ``app_config.ui.live_preview_fps``, and it is
    read as a **switch**, not a rate: ``<= 0`` means "no live video" — no
    preview threads exist at all, cameras are touched only when an inspection
    triggers, and the dashboard shows the pictures each cycle takes instead of
    a stream. When preview is on, each worker runs at its own camera's
    ``settings.fps`` (the Camera page's "Frame Rate"), so the station-wide
    value never overrides what a camera was tuned to.
    """
    if preview_enabled_fps <= 0:
        logger.info("Live preview disabled (ui.live_preview_fps <= 0)")
        return []
    return [
        AcquisitionWorker(
            camera_manager, index, app_state, camera_manager.get(index).settings.fps
        )
        for index in sorted(camera_manager.cameras)
    ]

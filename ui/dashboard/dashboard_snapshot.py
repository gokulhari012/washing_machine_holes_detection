"""Save a screenshot of the Dashboard after every full inspection cycle.

Switched by ``app_config.storage.save_dashboard_screenshot`` (Settings page,
Storage & Backup; off by default), read on every cycle so a Save takes effect
from the next part. Only a **full** cycle (the global PLC trigger, the
dashboard's Simulate Trigger, the toolbar's all-cameras button) is captured;
a single-camera cycle (``partial``) is not, since it does not finish a
machine.

Threading: the picture is grabbed on the GUI thread (``QWidget.grab`` and
``QPixmap`` are GUI-only), a short moment after the cycle is published so
the panels have repainted with its results, and converted to a ``QImage``,
which *is* safe to use from another thread. The PNG encode and write then
run on a short-lived daemon thread, so a slow disk never stutters the UI.
A failure is logged, never raised: a missing screenshot must not disturb the
station.

Files land beside that cycle's camera images, in
``<image_directory>/<YYYY-MM-DD>/<HHMMSS>_<machine>_dashboard_<RESULT>.png``,
with the cycle's own start time, so they sort next to the per-camera PNGs.

``storage.auto_align_dashboard_screenshot`` (off by default) resets every
camera picture to its default view — whole frame fitted and centred, no zoom,
no pan — immediately before the grab, so an operator who left a panel zoomed
in does not end up with a cropped picture in the archive. The reset is done
at grab time rather than when the cycle finishes, so a frame repainted during
the settle delay is aligned too, and it sticks: the panels stay fitted
afterwards, exactly as after a double-click.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QWidget

from core.logging import get_logger
from core.utilities import ConfigManager
from core.utilities.enums import LogSource

logger = get_logger(LogSource.UI)

#: Delay between the cycle being published and the grab, so the camera
#: panels and tiles have been repainted with that cycle's results first.
SETTLE_MS = 250


class DashboardSnapshotter:
    """Grabs *widget* after each full cycle when the setting is on."""

    def __init__(self, widget: QWidget, config_manager: ConfigManager) -> None:
        self._widget = widget
        self._config = config_manager

    def on_inspection_finished(self, cycle) -> None:
        """Slot for ``InspectionWorker.inspection_finished`` (GUI thread)."""
        if getattr(cycle, "partial", False):
            return
        storage = self._config.load("app_config").get("storage", {})
        if not storage.get("save_dashboard_screenshot", False):
            return
        path = self.path_for(cycle, storage)
        align = bool(storage.get("auto_align_dashboard_screenshot", False))
        QTimer.singleShot(SETTLE_MS, lambda: self._grab(path, align))

    @staticmethod
    def path_for(cycle, storage_cfg: dict) -> Path:
        day_dir = Path(storage_cfg.get("image_directory", "images")) / (
            f"{cycle.started_at:%Y-%m-%d}"
        )
        result = getattr(cycle.overall_result, "value", cycle.overall_result)
        return day_dir / (
            f"{cycle.started_at:%H%M%S}_{cycle.machine_number}_dashboard_{result}.png"
        )

    def _grab(self, path: Path, align: bool = False) -> None:
        try:
            if align:
                reset = getattr(self._widget, "reset_image_views", None)
                if reset is not None:
                    reset()
            image = self._widget.grab().toImage()
        except Exception:  # a screenshot must never break the GUI thread
            logger.exception("Dashboard screenshot grab failed")
            return
        threading.Thread(
            target=self._write, args=(image, path), name="dashboard-snapshot", daemon=True
        ).start()

    @staticmethod
    def _write(image: QImage, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if image.save(str(path), "PNG"):
                logger.info("Dashboard screenshot saved: %s", path)
            else:
                logger.warning("Dashboard screenshot save failed: %s", path)
        except OSError as exc:
            logger.warning("Dashboard screenshot save failed (%s): %s", path, exc)

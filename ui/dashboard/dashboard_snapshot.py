"""Save a screenshot of the Dashboard after every full inspection cycle.

Switched by ``app_config.storage.save_dashboard_screenshot`` (Settings page,
Storage & Backup; off by default), read on every cycle so a Save takes effect
from the next part. Only a **full** cycle (the global PLC trigger, the
dashboard's Simulate Trigger, the toolbar's all-cameras button) is captured;
a single-camera cycle (``partial``) is not, since it does not finish a
machine.

Threading: the picture is grabbed on the GUI thread (``QWidget.grab`` and
``QPixmap`` are GUI-only), a short moment after the cycle is published so
the panels have repainted with its results, and copied straight into a plain
numpy array there. **No Qt object ever leaves the GUI thread.** The PNG
encode and write run on one long-lived writer thread fed by a bounded queue,
using OpenCV only — so a slow disk never stutters the UI.

That split is deliberate. The station was closing itself — no traceback,
the window simply disappeared — when triggered repeatedly with this setting
on, and ran fine with it off. The writer then was a fresh
``threading.Thread`` per cycle calling ``QImage.save``, which builds a
``QImageWriter`` and a ``QFile`` (QObjects) on a thread Qt never created,
and emits log signals from it; that native, per-cycle Qt work off the GUI
thread is the suspected cause (it could not be reproduced off the station;
``logs/crash.log`` will name the thread if it ever happens again). Keep the
writer Qt-free and single.

One writer also means writes never overlap: two cycles stamped in the same
second used to race on the same file name. Now the second one gets a
``_2`` suffix instead (:func:`_unique_path`), and every file is written to
a temporary name and renamed into place, so a half-written PNG is never left
under the real name. If the writer falls :data:`MAX_PENDING` screenshots
behind, the newest is dropped with a warning — a missing screenshot must
not disturb the station, and neither must an unbounded backlog of frames.
A failure is logged, never raised.

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

import os
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
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

#: Screenshots allowed to wait for the writer before new ones are dropped.
MAX_PENDING = 8


def qimage_to_bgr(image: QImage) -> np.ndarray:
    """Copy *image* into an owned (H, W, 3) BGR array, safe on any thread."""
    image = image.convertToFormat(QImage.Format.Format_BGR888)
    width, height, stride = image.width(), image.height(), image.bytesPerLine()
    rows = np.frombuffer(image.constBits(), dtype=np.uint8, count=stride * height)
    return rows.reshape(height, stride)[:, : width * 3].reshape(height, width, 3).copy()


def _unique_path(path: Path) -> Path:
    """*path*, or ``<stem>_2``, ``_3``... if a file of that name exists."""
    candidate, n = path, 1
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
    return candidate


class DashboardSnapshotter:
    """Grabs *widget* after each full cycle when the setting is on."""

    def __init__(self, widget: QWidget, config_manager: ConfigManager) -> None:
        self._widget = widget
        self._config = config_manager
        self._queue: queue.Queue[tuple[np.ndarray, Path]] = queue.Queue(MAX_PENDING)
        self._writer: threading.Thread | None = None

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
            frame = qimage_to_bgr(self._widget.grab().toImage())
        except Exception:  # a screenshot must never break the GUI thread
            logger.exception("Dashboard screenshot grab failed")
            return
        try:
            self._queue.put_nowait((frame, path))
        except queue.Full:
            logger.warning("Dashboard screenshot dropped (writer is behind): %s", path)
            return
        if self._writer is None or not self._writer.is_alive():
            self._writer = threading.Thread(
                target=self._run_writer, name="dashboard-snapshot", daemon=True
            )
            self._writer.start()

    def _run_writer(self) -> None:
        while True:
            frame, path = self._queue.get()
            try:
                self._write(frame, path)
            except Exception:  # never let one bad write kill the writer
                logger.exception("Dashboard screenshot save failed: %s", path)

    @staticmethod
    def _write(frame: np.ndarray, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".png", frame)
            if not ok:
                logger.warning("Dashboard screenshot encode failed: %s", path)
                return
            target = _unique_path(path)
            temp = target.with_name(target.name + ".tmp")
            temp.write_bytes(encoded.tobytes())
            os.replace(temp, target)
            logger.info("Dashboard screenshot saved: %s", target)
        except OSError as exc:
            logger.warning("Dashboard screenshot save failed (%s): %s", path, exc)

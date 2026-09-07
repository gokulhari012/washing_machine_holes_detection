"""Off-thread checkerboard scanning for the Calibration page's Auto Calibrate.

Auto Calibrate has to grab a frame and search it for a checkerboard several
times a second. Both halves of that are far too slow to sit inside a GUI-thread
timer, which is how this used to run:

- a grab holds the camera's capture lock for as long as the device takes — up
  to ``basler.grab_timeout_ms`` (5 s in this station's ``camera.json``), and it
  contends with the inspection pipeline for the same device;
- ``cv2.findChessboardCorners`` measured **0.9-1.2 s** on the 4024x3036 cameras
  and **1.8-1.9 s** on the 5496x3672 Basler — per call, per frame.

Driven from a 250 ms ``QTimer`` on the Qt main thread, each tick therefore
overran its own interval by roughly an order of magnitude, the event loop never
regained control, and the window froze outright.

This worker moves both halves onto its own thread and adds the one optimisation
that actually matters: **no corner search ever runs at full resolution.** Both
the screening pass (the "is a board in view?" question asked every tick) and the
kept-view pass search a copy downscaled to :data:`SCREEN_MAX_DIM` — ~0.15 s
regardless of sensor size — because locating a board is a coarse question that
gains nothing from 20 MP.

Accuracy is not traded away for that: ``find_checkerboard`` scales the corners
found on the small copy back up and runs the sub-pixel refinement against the
**full-resolution** frame, so the correspondences fed to ``calibrate_lens`` and
the homography are in the real image's pixel basis. What the downscale removes
is the coarse search's cost, not the calibration's resolution.

A dropped frame does not end the session: a capture failure is retried on the
next tick and only reported through ``failed`` once
:data:`MAX_CONSECUTIVE_CAPTURE_FAILURES` grabs in a row have failed, so a
GigE hiccup mid-session no longer discards the views already captured.

Results come back as queued signals carrying the **full-resolution** frame, so
the page's picker keeps its image-pixel coordinate basis and the frame handed to
``calibrate_lens`` is the same one the single-threaded version used. The worker
holds no session state beyond the view count it needs to know when to stop; the
page owns the accumulated detections.

Layering: this worker takes a plain ``capture`` callable rather than importing
``CameraService``, keeping the downward dependency rule intact (``workers/``
sits below ``services/``).
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.calibration import CameraCalibration
from core.logging import get_logger
from core.utilities.enums import LogSource
from core.utilities.exceptions import VisionSystemError

logger = get_logger(LogSource.VISION)

# Longest edge any corner *search* is allowed to work on — screening and kept
# views alike. 1000 px keeps a findChessboardCorners call at ~0.15 s on every
# camera in this station while staying far above the resolution needed to
# locate a board; the sub-pixel refinement still runs at full resolution.
SCREEN_MAX_DIM = 1000

# A scan runs for minutes, so it must survive the odd dropped frame: a GigE
# camera on a shared NIC hands back an incomplete buffer now and then (the
# driver already re-triggers a couple of times before it reports one), and
# ending the session on a single one throws away every view captured so far.
# Only a camera that fails this many times *in a row* is genuinely gone.
MAX_CONSECUTIVE_CAPTURE_FAILURES = 5

_FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


class CheckerboardScanWorker(QThread):
    """Scans a camera for a checkerboard until stopped or ``max_views`` are in."""

    #: (full-resolution frame, corners in full-res pixels or ``None``)
    scanned = Signal(object, object)
    #: (CheckerboardDetection, total views captured so far)
    view_captured = Signal(object, int)
    #: human-readable progress line for the page's status label
    status = Signal(str)
    #: capture failed — the session cannot continue
    failed = Signal(str)
    #: ``max_views`` reached; the page should compute now
    quota_reached = Signal()

    def __init__(
        self,
        capture: Callable[[], np.ndarray],
        columns: int,
        rows: int,
        square_size_mm: float,
        min_gap_s: float,
        max_views: int,
        tick_s: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("CheckerboardScanWorker")
        self._capture = capture
        self._columns = columns
        self._rows = rows
        self._square_size_mm = square_size_mm
        self._min_gap_s = min_gap_s
        self._max_views = max_views
        self._tick_s = tick_s
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        """Ask the loop to exit.

        Deliberately does **not** wait: a grab in flight can hold the thread for
        seconds, and blocking the GUI thread on that is the very failure this
        worker exists to remove. Callers watch :meth:`QThread.finished` instead.
        """
        self._stop_event.set()

    # ----------------------------------------------------------------- loop
    def run(self) -> None:  # noqa: D102 — see class docstring
        views_captured = 0
        consecutive_failures = 0
        last_capture = 0.0  # monotonic; 0 means "capture the first sighting"

        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                frame = self._capture()
            except VisionSystemError as exc:
                consecutive_failures += 1
                logger.warning(
                    "Auto Calibrate capture failed (%d/%d): %s",
                    consecutive_failures,
                    MAX_CONSECUTIVE_CAPTURE_FAILURES,
                    exc,
                )
                if consecutive_failures >= MAX_CONSECUTIVE_CAPTURE_FAILURES:
                    self.failed.emit(str(exc))
                    return
                self.status.emit(
                    f"{views_captured}/{self._max_views} views captured — "
                    f"frame dropped ({consecutive_failures}/"
                    f"{MAX_CONSECUTIVE_CAPTURE_FAILURES}), retrying"
                )
                self._sleep_remainder(started)
                continue
            except Exception:  # pragma: no cover — programming error, keep the UI alive
                logger.exception("Auto Calibrate capture raised unexpectedly")
                self.failed.emit("unexpected capture error (see logs)")
                return
            consecutive_failures = 0

            if self._stop_event.is_set():
                return

            found, corners = self._screen(frame)
            self.scanned.emit(frame, corners if found else None)

            progress = f"{views_captured}/{self._max_views} views captured"
            if not found:
                self.status.emit(f"{progress} — checkerboard not visible")
                self._sleep_remainder(started)
                continue

            remaining = self._min_gap_s - (time.monotonic() - last_capture)
            if remaining > 0:
                self.status.emit(
                    f"{progress} — board visible, next capture in {remaining:.0f} s"
                )
                self._sleep_remainder(started)
                continue

            self.status.emit(f"{progress} — board visible, refining corners...")
            try:
                # Searched on the same downscaled basis as the screening pass —
                # the coarse "where is the board" question does not need the
                # sensor's full resolution. find_checkerboard scales the corners
                # back up and refines them sub-pixel against the full-resolution
                # frame, so the correspondences this view contributes are in the
                # actual image's coordinate basis, as the calibration requires.
                detection = CameraCalibration.find_checkerboard(
                    frame,
                    self._columns,
                    self._rows,
                    self._square_size_mm,
                    detect_max_dim=SCREEN_MAX_DIM,
                )
            except VisionSystemError:
                # Screening said yes, the precise pass disagreed — a borderline
                # view. Not an error; just try again on the next frame.
                self._sleep_remainder(started)
                continue

            if self._stop_event.is_set():
                return

            views_captured += 1
            last_capture = time.monotonic()
            self.view_captured.emit(detection, views_captured)
            self.status.emit(
                f"{views_captured}/{self._max_views} views captured — "
                "move/tilt the board for the next one"
            )
            if views_captured >= self._max_views:
                self.quota_reached.emit()
                return
            self._sleep_remainder(started)

    # ------------------------------------------------------------- internal
    def _screen(self, frame: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Cheap "is a board in view?" test on a downscaled copy.

        Returns corners rescaled to full-resolution pixels so the page can draw
        them over the frame it was given.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        longest = max(gray.shape[:2])
        scale = SCREEN_MAX_DIM / longest if longest > SCREEN_MAX_DIM else 1.0
        probe = (
            cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else gray
        )
        found, corners = cv2.findChessboardCorners(
            probe, (self._columns, self._rows), flags=_FIND_FLAGS | cv2.CALIB_CB_FAST_CHECK
        )
        if not found or corners is None:
            return False, None
        return True, (corners / scale if scale < 1.0 else corners)

    def _sleep_remainder(self, started: float) -> None:
        """Pace the loop to ``tick_s`` without ever sleeping past a stop request."""
        remaining = self._tick_s - (time.monotonic() - started)
        if remaining > 0:
            self._stop_event.wait(remaining)

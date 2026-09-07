"""Auto Calibrate's off-thread checkerboard scan.

Regression cover for the freeze that made Auto Calibrate unusable: the scan
used to run ``test_capture`` plus a **full-resolution** ``findChessboardCorners``
inside a 250 ms ``QTimer`` on the Qt main thread. On this station's 12-20 MP
cameras that one call measures 0.9-1.9 s on a realistic frame, so every tick
overran its interval several times over and the event loop never regained
control.

Two properties keep that from coming back, and both are asserted here:

- **no** corner search — screening or kept view — runs on a copy larger than
  :data:`SCREEN_MAX_DIM`, so scanning stays cheap whatever the sensor size;
- the kept view's corners are nonetheless refined sub-pixel against the
  full-resolution frame and reported in full-resolution pixels, so the
  calibration is fitted in the actual image's coordinate basis.

Following ``test_plc_poll_worker``: ``run()`` is driven on a plain Python
thread and stopped through the same stop event, and signals are connected
``DirectConnection`` — with no Qt event loop spinning, queued delivery would
never arrive.
"""

import threading
import time

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt

from core.utilities.exceptions import CameraCaptureError
from workers.checkerboard_worker import SCREEN_MAX_DIM, CheckerboardScanWorker

COLUMNS, ROWS = 8, 5
FULL_W, FULL_H = 4024, 3036


def render_board(columns=COLUMNS, rows=ROWS, square_px=180, width=None, height=None):
    """A checkerboard OpenCV actually detects: quiet white zone around it.

    ``columns``/``rows`` are *inner* corners, so the drawn grid is one square
    larger each way — the same convention ``find_checkerboard`` uses.
    """
    quiet = 2 * square_px
    board_w, board_h = (columns + 1) * square_px, (rows + 1) * square_px
    img = np.full((board_h + 2 * quiet, board_w + 2 * quiet), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(columns + 1):
            if (r + c) % 2 == 0:
                y0, x0 = quiet + r * square_px, quiet + c * square_px
                img[y0 : y0 + square_px, x0 : x0 + square_px] = 0
    if width and height:
        canvas = np.full((height, width), 255, np.uint8)
        y_off, x_off = (height - img.shape[0]) // 2, (width - img.shape[1]) // 2
        canvas[y_off : y_off + img.shape[0], x_off : x_off + img.shape[1]] = img
        img = canvas
    return img


@pytest.fixture()
def full_frame() -> np.ndarray:
    return render_board(width=FULL_W, height=FULL_H)


def make_worker(capture, *, min_gap_s=0.0, max_views=10, tick_s=0.0, square_mm=25.0):
    return CheckerboardScanWorker(
        capture=capture,
        columns=COLUMNS,
        rows=ROWS,
        square_size_mm=square_mm,
        min_gap_s=min_gap_s,
        max_views=max_views,
        tick_s=tick_s,
    )


def drive(worker: CheckerboardScanWorker, timeout: float = 60.0) -> threading.Thread:
    """Run the real loop body on a plain thread and join it."""
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "worker loop did not finish"
    return thread


def test_no_corner_search_runs_at_full_resolution(full_frame, monkeypatch):
    """Neither the per-tick screen nor the kept view may search the full frame.

    This is the freeze itself: at 4024x3036 that call costs about a second,
    which a 250 ms cadence cannot absorb. Both passes therefore search a copy
    bounded by ``SCREEN_MAX_DIM``; accuracy is preserved by the sub-pixel
    refinement, asserted below to still run against the full frame.
    """
    search_shapes: list[tuple[int, int]] = []
    refine_shapes: list[tuple[int, int]] = []
    real_find = cv2.findChessboardCorners
    real_subpix = cv2.cornerSubPix

    def recording_find(image, pattern_size, flags=0, **kwargs):
        search_shapes.append(image.shape[:2])
        return real_find(image, pattern_size, flags=flags, **kwargs)

    def recording_subpix(image, corners, *args, **kwargs):
        refine_shapes.append(image.shape[:2])
        return real_subpix(image, corners, *args, **kwargs)

    monkeypatch.setattr(cv2, "findChessboardCorners", recording_find)
    monkeypatch.setattr(cv2, "cornerSubPix", recording_subpix)

    worker = make_worker(lambda: full_frame, max_views=1)
    drive(worker)

    assert search_shapes, "no board search ran"
    for height, width in search_shapes:
        assert max(height, width) <= SCREEN_MAX_DIM, (
            f"a board search ran at {width}x{height} — full-resolution search on "
            "the GUI cadence is what froze the page"
        )
    assert (FULL_H, FULL_W) in refine_shapes, (
        "the kept view's corners must be refined against the full-resolution "
        "frame, or the calibration is fitted in the downscaled basis"
    )


def test_kept_view_reports_full_resolution_corners(full_frame):
    """Screening is downscaled, but the calibration data must not be."""
    captured: list = []
    worker = make_worker(lambda: full_frame, max_views=1)
    worker.view_captured.connect(
        lambda detection, count: captured.append(detection),
        Qt.ConnectionType.DirectConnection,
    )
    drive(worker)

    assert len(captured) == 1
    detection = captured[0]
    assert len(detection.pixel_points) == COLUMNS * ROWS
    longest = max(max(x, y) for x, y in detection.pixel_points)
    assert longest > SCREEN_MAX_DIM, (
        "corners look like they came from the downscaled screening copy"
    )
    # 180 px squares at 25 mm each.
    assert detection.pixels_per_mm_x == pytest.approx(180 / 25.0, rel=0.02)


def test_scanned_carries_the_full_frame_and_corners(full_frame):
    """The page relies on this frame for its picker's pixel coordinates."""
    events: list = []
    worker = make_worker(lambda: full_frame, max_views=1)
    worker.scanned.connect(
        lambda frame, corners: events.append((frame, corners)),
        Qt.ConnectionType.DirectConnection,
    )
    drive(worker)

    frame, corners = events[0]
    assert frame.shape[:2] == (FULL_H, FULL_W)
    assert corners is not None
    # Rescaled back out of the screening copy, into full-resolution pixels.
    assert float(corners[..., 0].max()) > SCREEN_MAX_DIM


def test_blank_frame_reports_not_visible_and_keeps_scanning():
    blank = np.full((600, 800), 255, np.uint8)
    statuses: list[str] = []
    calls = {"n": 0}

    def capture():
        calls["n"] += 1
        return blank

    worker = make_worker(capture)
    worker.status.connect(statuses.append, Qt.ConnectionType.DirectConnection)
    captured: list = []
    worker.view_captured.connect(
        lambda d, c: captured.append(d), Qt.ConnectionType.DirectConnection
    )

    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    time.sleep(0.3)
    worker.request_stop()
    thread.join(10)

    assert not thread.is_alive()
    assert not captured, "a blank frame must not be kept as a view"
    assert calls["n"] > 1, "the loop should keep scanning while nothing is found"
    assert any("not visible" in s for s in statuses)


def test_min_gap_holds_off_the_next_view(full_frame):
    """A board sitting still must not be captured over and over."""
    captured: list = []
    worker = make_worker(full_frame.copy, min_gap_s=30.0, max_views=5)
    worker.view_captured.connect(
        lambda d, c: captured.append(d), Qt.ConnectionType.DirectConnection
    )

    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    time.sleep(0.6)
    worker.request_stop()
    thread.join(10)

    assert not thread.is_alive()
    assert len(captured) == 1, "second view taken before the minimum gap elapsed"


def test_quota_reached_ends_the_session(full_frame):
    reached: list[bool] = []
    captured: list = []
    worker = make_worker(lambda: full_frame, max_views=2)
    worker.quota_reached.connect(
        lambda: reached.append(True), Qt.ConnectionType.DirectConnection
    )
    worker.view_captured.connect(
        lambda d, c: captured.append(c), Qt.ConnectionType.DirectConnection
    )
    drive(worker)

    assert reached == [True]
    assert captured == [1, 2]


def test_capture_failure_ends_the_session():
    reasons: list[str] = []
    worker = make_worker(
        lambda: (_ for _ in ()).throw(CameraCaptureError("camera 3: no frame within 5000 ms"))
    )
    worker.failed.connect(reasons.append, Qt.ConnectionType.DirectConnection)
    drive(worker, timeout=10)

    assert len(reasons) == 1
    assert "no frame" in reasons[0]


def test_request_stop_returns_without_waiting_out_a_slow_grab():
    """``request_stop`` must not block; the caller is the GUI thread."""
    release = threading.Event()

    def slow_capture():
        release.wait(5.0)
        return np.full((600, 800), 255, np.uint8)

    worker = make_worker(slow_capture)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    time.sleep(0.1)

    started = time.monotonic()
    worker.request_stop()
    assert time.monotonic() - started < 0.1, "request_stop blocked the caller"

    release.set()
    thread.join(10)
    assert not thread.is_alive()


def test_stop_during_the_tick_pause_is_prompt():
    """A long tick interval must not delay the stop by its full length."""
    worker = make_worker(lambda: np.full((400, 600), 255, np.uint8), tick_s=30.0)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    time.sleep(0.3)

    started = time.monotonic()
    worker.request_stop()
    thread.join(10)
    elapsed = time.monotonic() - started

    assert not thread.is_alive()
    assert elapsed < 2.0, f"stop waited out the tick interval ({elapsed:.1f}s)"



def test_a_single_dropped_frame_does_not_end_the_session(full_frame):
    """A GigE hiccup mid-scan must not discard the views already captured."""
    calls = {"n": 0}

    def flaky_capture():
        calls["n"] += 1
        if calls["n"] == 1:  # e.g. grab error 0xe1000014, incomplete buffer
            raise CameraCaptureError("camera 3: grab error 0xe1000014 (incomplete)")
        return full_frame

    reasons: list[str] = []
    views: list[int] = []
    worker = make_worker(flaky_capture, max_views=1)
    worker.failed.connect(reasons.append, Qt.ConnectionType.DirectConnection)
    worker.view_captured.connect(
        lambda _d, count: views.append(count), Qt.ConnectionType.DirectConnection
    )
    drive(worker, timeout=30)

    assert reasons == []  # the drop was retried, not reported
    assert views == [1]  # and the scan went on to capture its view

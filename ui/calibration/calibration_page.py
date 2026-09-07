"""Calibration page: pixel→mm scale, reference point, perspective correction.

Workflow (per camera):
0. **Auto Calibrate** (optional, recommended) — click "Start Auto Calibrate"
   to open a live preview and hold/move a checkerboard in view. A view is
   captured automatically whenever the board is visible **and** at least
   :data:`AUTO_CALIBRATE_MIN_GAP_S` seconds have passed since the last
   capture — move or tilt the board between captures so the up-to-
   :data:`AUTO_CALIBRATE_MAX_VIEWS` views are different enough poses for
   :meth:`CameraCalibration.calibrate_lens` to separate lens distortion from
   perspective (a single photo can't). The session ends automatically at the
   view cap, or early via "Stop & Compute" once at least
   :data:`AUTO_CALIBRATE_MIN_VIEWS` views are in. The result fills in the
   scale (Step 1) and homography (Step 3), with pixels undistorted through
   the fitted lens model first. Remove the board and redo Step 2 afterwards —
   the reference point must be captured through the fresh calibration.

   The scan runs on :class:`CheckerboardScanWorker`, never on this thread: a
   grab plus a full-resolution ``findChessboardCorners`` costs 1-2 s on this
   station's 12-20 MP cameras, and driving that from a GUI-thread timer froze
   the page. The worker searches for the board on a downscaled copy and
   refines the corners it finds sub-pixel against the full-resolution frame,
   so what reaches this page is always in the actual image's pixel basis. Its
   scan cadence is the selected camera's own ``fps`` (camera.json / Camera
   page) — the same rate every other continuous view of that camera runs at.
   This page only consumes the worker's signals, and it never waits
   on that thread — stopping is a request, with the button re-enabled when the
   thread's ``finished`` arrives.
1. **Scale** — three interchangeable ways to fill in pixels-per-mm, in
   increasing order of effort/accuracy: type it directly, enter a known
   two-point pixel/mm distance ("Compute"), or place a ruler in frame and
   click 5-10 points spaced a fixed distance apart ("Pick Ruler Points" +
   "Compute PPMM from Points") — the average of the clicked segments' pixel
   lengths averages out clicking jitter better than a single two-point read.
   All three are skippable if Auto Calibrate already filled this in.
2. **Reference point** — "Detect Hole → Set Reference" runs the vision engine
   on the captured frame and stores the hole position (in mm, through the
   current scale/homography) as the nominal position.
3. Optional **perspective**: enter 4 pixel↔mm point pairs and "Compute
   Homography" (RMS shown; replaces plain scaling in the hot path) — or let
   Auto Calibrate fill this in from the checkerboard instead of typing points.
4. **Save Calibration** persists as the camera's active calibration.
5. **Live Test** captures + detects + evaluates through the saved model.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.calibration import CalibrationManager, CameraCalibration, CheckerboardDetection
from core.camera import frame_interval_ms
from core.utilities.exceptions import VisionSystemError
from core.vision import VisionEngine, draw_detection_overlay
from services.camera_service import CameraService
from ui.widgets import PointPicker
from workers import CheckerboardScanWorker

# Live-preview auto-calibrate session tuning (see module docstring, step 0).
# The scan tick is not tuned here: it is the selected camera's configured
# frame rate, so one Camera-tab setting drives every continuous view of it.
AUTO_CALIBRATE_MIN_GAP_S = 10.0
AUTO_CALIBRATE_MAX_VIEWS = 10
AUTO_CALIBRATE_MIN_VIEWS = 3  # cv2.calibrateCamera needs several distinct poses

# Hand-clicked ruler scale (Step 1): recommended point count, see scale_from_points.
RULER_MIN_POINTS = 5
RULER_MAX_POINTS = 10


def _dspin(maximum: float = 100000.0, decimals: int = 2) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-maximum, maximum)
    spin.setDecimals(decimals)
    return spin


def _spin(minimum: int, maximum: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    return spin


class CalibrationPage(QWidget):
    """Guided scale/reference/homography calibration with live test."""

    def __init__(
        self,
        camera_service: CameraService,
        calibration_manager: CalibrationManager,
        vision_engine: VisionEngine,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cameras = camera_service
        self._manager = calibration_manager
        self._engine = vision_engine
        self._frame: np.ndarray | None = None

        # Auto Calibrate session state (see module docstring, step 0).
        # The scan itself runs on CheckerboardScanWorker, never here: a grab plus
        # a full-resolution findChessboardCorners costs 1-2 s on this station's
        # 12-20 MP cameras, which froze the event loop when it ran in a timer on
        # this thread. The page is now a pure consumer of the worker's signals.
        self._auto_session_active = False
        self._auto_views: list[CheckerboardDetection] = []
        self._auto_worker: CheckerboardScanWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        title = QLabel("Calibration")
        title.setProperty("class", "pageTitle")
        root.addWidget(title)

        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, stretch=1)

        # -------------------------------------------------------- left column
        left = QVBoxLayout()

        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera"))
        self._camera = QComboBox()
        for cfg in self._cameras.get_configs():
            self._camera.addItem(f"{cfg['index']}: {cfg.get('name', '')}", cfg["index"])
        self._camera.currentIndexChanged.connect(self._load_existing)
        capture_btn = QPushButton("Capture Frame")
        capture_btn.clicked.connect(self._on_capture)
        camera_row.addWidget(self._camera, stretch=1)
        camera_row.addWidget(capture_btn)
        left.addLayout(camera_row)

        auto_box = QGroupBox("Auto Calibrate (checkerboard) — fills in Steps 1 && 3")
        auto_form = QFormLayout(auto_box)
        board_row = QHBoxLayout()
        self._board_columns = _spin(2, 50)
        self._board_columns.setValue(8)
        self._board_rows = _spin(2, 50)
        self._board_rows.setValue(5)
        board_row.addWidget(self._board_columns)
        board_row.addWidget(QLabel("x"))
        board_row.addWidget(self._board_rows)
        board_row.addStretch()
        board_w = QWidget()
        board_w.setLayout(board_row)
        board_w.setToolTip(
            "Inner corners, not squares — one less than the square count each "
            "way (a 9x6-square board has 8x5 inner corners)"
        )
        self._board_square_mm = _dspin(500.0, 2)
        self._board_square_mm.setValue(25.0)
        self._auto_calibrate_btn = QPushButton("Start Auto Calibrate")
        self._auto_calibrate_btn.setProperty("class", "primary")
        self._auto_calibrate_btn.clicked.connect(self._on_auto_calibrate_clicked)
        auto_form.addRow("Inner Corners (cols x rows)", board_w)
        auto_form.addRow("Square Size (mm)", self._board_square_mm)
        auto_form.addRow(self._auto_calibrate_btn)
        self._auto_status = QLabel(
            f"Opens a live preview; hold a checkerboard in view and it captures "
            f"a view every {AUTO_CALIBRATE_MIN_GAP_S:.0f}+ s (move/tilt the board "
            f"between captures) for up to {AUTO_CALIBRATE_MAX_VIEWS} views, then "
            f"fits lens distortion + perspective from all of them together."
        )
        self._auto_status.setWordWrap(True)
        self._auto_status.setProperty("class", "dim")
        auto_form.addRow(self._auto_status)
        left.addWidget(auto_box)

        scale_box = QGroupBox("Step 1 — Pixel to mm scale")
        scale_form = QFormLayout(scale_box)
        self._ppmm_x = _dspin(10000, 4)
        self._ppmm_y = _dspin(10000, 4)
        known_row = QHBoxLayout()
        self._known_px = _dspin(100000, 1)
        self._known_mm = _dspin(100000, 2)
        compute_scale = QPushButton("Compute")
        compute_scale.clicked.connect(self._on_compute_scale)
        known_row.addWidget(QLabel("px"))
        known_row.addWidget(self._known_px)
        known_row.addWidget(QLabel("mm"))
        known_row.addWidget(self._known_mm)
        known_row.addWidget(compute_scale)
        known_w = QWidget()
        known_w.setLayout(known_row)
        scale_form.addRow("Pixels per mm X", self._ppmm_x)
        scale_form.addRow("Pixels per mm Y", self._ppmm_y)
        scale_form.addRow("Known distance", known_w)

        ruler_row = QHBoxLayout()
        self._ruler_pick_btn = QPushButton("Pick Ruler Points")
        self._ruler_pick_btn.setCheckable(True)
        self._ruler_pick_btn.toggled.connect(self._on_ruler_pick_toggled)
        self._ruler_spacing_mm = _dspin(1000.0, 3)
        self._ruler_spacing_mm.setValue(1.0)
        self._ruler_points_label = QLabel("Points: 0")
        self._ruler_points_label.setProperty("class", "dim")
        ruler_compute_btn = QPushButton("Compute PPMM from Points")
        ruler_compute_btn.clicked.connect(self._on_compute_ruler_scale)
        ruler_clear_btn = QPushButton("Clear Points")
        ruler_clear_btn.clicked.connect(self._on_clear_ruler_points)
        ruler_row.addWidget(self._ruler_pick_btn)
        ruler_row.addWidget(QLabel("mm/point"))
        ruler_row.addWidget(self._ruler_spacing_mm)
        ruler_row.addWidget(self._ruler_points_label)
        ruler_row.addWidget(ruler_compute_btn)
        ruler_row.addWidget(ruler_clear_btn)
        ruler_row.addStretch()
        ruler_w = QWidget()
        ruler_w.setLayout(ruler_row)
        scale_form.addRow("Ruler (click points)", ruler_w)
        ruler_hint = QLabel(
            f"Place a ruler in frame, enable picking, and left-click {RULER_MIN_POINTS}-"
            f"{RULER_MAX_POINTS} points each exactly one division apart (right-click "
            f"undoes the last one) — the average spacing becomes pixels-per-mm."
        )
        ruler_hint.setWordWrap(True)
        ruler_hint.setProperty("class", "dim")
        scale_form.addRow(ruler_hint)
        left.addWidget(scale_box)

        ref_box = QGroupBox("Step 2 — Reference (nominal) position")
        ref_form = QFormLayout(ref_box)
        self._ref_x = _dspin()
        self._ref_y = _dspin()
        detect_btn = QPushButton("Detect Hole → Set Reference")
        detect_btn.clicked.connect(self._on_detect_reference)
        ref_form.addRow("Reference X (mm)", self._ref_x)
        ref_form.addRow("Reference Y (mm)", self._ref_y)
        ref_form.addRow(detect_btn)
        left.addWidget(ref_box)

        homography_box = QGroupBox("Step 3 (optional) — Perspective correction")
        hom_layout = QVBoxLayout(homography_box)
        grid = QGridLayout()
        grid.addWidget(QLabel("px X"), 0, 1)
        grid.addWidget(QLabel("px Y"), 0, 2)
        grid.addWidget(QLabel("mm X"), 0, 3)
        grid.addWidget(QLabel("mm Y"), 0, 4)
        self._hom_points: list[tuple[QDoubleSpinBox, ...]] = []
        for row in range(4):
            spins = tuple(_dspin(100000, 2) for _ in range(4))
            self._hom_points.append(spins)
            grid.addWidget(QLabel(f"P{row + 1}"), row + 1, 0)
            for column, spin in enumerate(spins, start=1):
                grid.addWidget(spin, row + 1, column)
        hom_layout.addLayout(grid)
        hom_row = QHBoxLayout()
        compute_hom = QPushButton("Compute Homography")
        compute_hom.clicked.connect(self._on_compute_homography)
        clear_hom = QPushButton("Clear")
        clear_hom.clicked.connect(self._on_clear_homography)
        self._hom_status = QLabel("not set")
        self._hom_status.setProperty("class", "dim")
        hom_row.addWidget(compute_hom)
        hom_row.addWidget(clear_hom)
        hom_row.addWidget(self._hom_status)
        hom_row.addStretch()
        hom_layout.addLayout(hom_row)
        left.addWidget(homography_box)
        self._homography: np.ndarray | None = None
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None
        self._rms = 0.0

        action_row = QHBoxLayout()
        save_btn = QPushButton("Save Calibration")
        save_btn.setProperty("class", "primary")
        save_btn.clicked.connect(self._on_save)
        live_btn = QPushButton("Live Test")
        live_btn.clicked.connect(self._on_live_test)
        action_row.addWidget(save_btn)
        action_row.addWidget(live_btn)
        left.addLayout(action_row)
        self._status = QLabel("—")
        self._status.setProperty("class", "dim")
        left.addWidget(self._status)
        left.addStretch()
        body.addLayout(left)

        # -------------------------------------------------------- right view
        # The whole page lives in MainWindow's QScrollArea, so the body row is
        # as tall as the (very long) left column. Letting the view stretch to
        # that height parked the fitted image around the column's midpoint —
        # i.e. off-screen below the fold. Cap the height and top-align it so the
        # picture sits beside the first controls, where it is actually visible.
        self._view = PointPicker()
        self._view.setMinimumSize(480, 380)
        self._view.setMaximumHeight(620)
        self._view.points_changed.connect(self._on_ruler_points_changed)
        right = QVBoxLayout()
        right.addWidget(self._view)
        right.addStretch()
        body.addLayout(right, stretch=1)

        self._load_existing()

    # ------------------------------------------------------------- helpers
    def _camera_index(self) -> int | None:
        data = self._camera.currentData()
        return int(data) if data is not None else None

    def _load_existing(self) -> None:
        index = self._camera_index()
        if index is None:
            return
        if self._auto_session_active:
            self._cancel_auto_calibrate_session("camera changed")
        calibration = self._manager.get(index)
        if calibration is None:
            self._status.setText("No active calibration — identity fallback (1 px = 1 mm)")
            self._homography = None
            self._camera_matrix = None
            self._dist_coeffs = None
            self._hom_status.setText("not set")
            return
        self._ppmm_x.setValue(calibration.pixels_per_mm_x)
        self._ppmm_y.setValue(calibration.pixels_per_mm_y)
        self._ref_x.setValue(calibration.ref_point_mm[0])
        self._ref_y.setValue(calibration.ref_point_mm[1])
        self._homography = calibration.homography
        self._camera_matrix = calibration.camera_matrix
        self._dist_coeffs = calibration.dist_coeffs
        self._rms = calibration.rms_error
        lens_note = " + lens distortion" if calibration.camera_matrix is not None else ""
        self._hom_status.setText(
            f"active (rms {calibration.rms_error:.3f} mm{lens_note})"
            if calibration.homography is not None
            else "not set"
        )
        self._status.setText("Active calibration loaded")

    # -------------------------------------------------------------- actions
    def _on_capture(self) -> None:
        index = self._camera_index()
        if index is None:
            return
        if self._auto_session_active:
            self._cancel_auto_calibrate_session("manual capture requested")
        try:
            self._frame = self._cameras.test_capture(index)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Capture", str(exc))
            return
        self._view.set_frame(self._frame)
        self._status.setText("Frame captured")

    def _on_auto_calibrate_clicked(self) -> None:
        if self._auto_session_active:
            self._finish_auto_calibrate_session()
        else:
            self._start_auto_calibrate_session()

    def _start_auto_calibrate_session(self) -> None:
        index = self._camera_index()
        if index is None:
            return
        if self._auto_worker is not None:
            # The previous session's thread is still winding down — a grab in
            # flight can hold it for seconds. A second scanner on the same
            # camera would only fight the first for the capture lock.
            self._auto_status.setText(
                "Previous session is still stopping — try again in a moment."
            )
            return

        self._auto_views = []
        self._auto_session_active = True
        self._set_board_inputs_enabled(False)
        self._auto_calibrate_btn.setText(f"Stop && Compute (0/{AUTO_CALIBRATE_MAX_VIEWS})")
        self._auto_status.setText(
            "Live preview running — show the checkerboard; a view is captured "
            f"automatically once it's been visible for {AUTO_CALIBRATE_MIN_GAP_S:.0f}+ s "
            "since the last capture. Move/tilt the board between captures."
        )

        # Board geometry is snapshotted for the whole session: calibrate_lens
        # requires every view to come from the same board, so letting the spin
        # boxes change mid-session could only build a set it has to reject.
        worker = CheckerboardScanWorker(
            capture=lambda: self._cameras.test_capture(index),
            columns=self._board_columns.value(),
            rows=self._board_rows.value(),
            square_size_mm=self._board_square_mm.value(),
            min_gap_s=AUTO_CALIBRATE_MIN_GAP_S,
            max_views=AUTO_CALIBRATE_MAX_VIEWS,
            tick_s=frame_interval_ms(self._cameras.camera_fps(index)) / 1000.0,
            parent=self,
        )
        worker.scanned.connect(self._on_auto_scanned)
        worker.view_captured.connect(self._on_auto_view_captured)
        worker.status.connect(self._on_auto_status)
        worker.failed.connect(self._on_auto_failed)
        worker.quota_reached.connect(self._finish_auto_calibrate_session)
        worker.finished.connect(self._on_auto_worker_finished)
        self._auto_worker = worker
        worker.start()

    def _set_board_inputs_enabled(self, enabled: bool) -> None:
        self._board_columns.setEnabled(enabled)
        self._board_rows.setEnabled(enabled)
        self._board_square_mm.setEnabled(enabled)

    # ------------------------------------------------ worker signal handlers
    def _on_auto_scanned(self, frame: np.ndarray, corners: object) -> None:
        if not self._auto_session_active:
            return
        # Keep the full-resolution frame: Step 2's detection, the ruler picker's
        # image-pixel coordinates and calibrate_lens's image_size all read it.
        self._frame = frame
        overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
        if corners is not None:
            cv2.drawChessboardCorners(
                overlay,
                (self._board_columns.value(), self._board_rows.value()),
                np.asarray(corners, dtype=np.float32),
                True,
            )
        self._view.set_frame(overlay)

    def _on_auto_view_captured(self, detection: object, count: int) -> None:
        if not self._auto_session_active:
            return
        self._auto_views.append(detection)
        self._auto_calibrate_btn.setText(
            f"Stop && Compute ({count}/{AUTO_CALIBRATE_MAX_VIEWS})"
        )

    def _on_auto_status(self, text: str) -> None:
        if self._auto_session_active:
            self._auto_status.setText(text)

    def _on_auto_failed(self, reason: str) -> None:
        self._cancel_auto_calibrate_session(f"capture failed: {reason}")

    def _on_auto_worker_finished(self) -> None:
        worker = self._auto_worker
        self._auto_worker = None
        if worker is not None:
            worker.deleteLater()
        self._auto_calibrate_btn.setEnabled(True)

    def _stop_auto_worker(self) -> None:
        """Ask the scan thread to stop; never wait for it on this thread.

        A grab in flight can hold the worker for seconds (``grab_timeout_ms``),
        and blocking the GUI thread on that would reintroduce exactly the freeze
        this worker exists to remove. The button stays disabled until the
        thread's ``finished`` signal arrives.
        """
        self._auto_session_active = False
        self._set_board_inputs_enabled(True)
        if self._auto_worker is not None:
            self._auto_worker.request_stop()
            if self._auto_worker.isRunning():
                self._auto_calibrate_btn.setEnabled(False)

    def _cancel_auto_calibrate_session(self, reason: str) -> None:
        self._stop_auto_worker()
        self._auto_calibrate_btn.setText("Start Auto Calibrate")
        self._auto_status.setText(f"Auto Calibrate stopped: {reason}")

    def _finish_auto_calibrate_session(self) -> None:
        # Views have already been accumulated from the worker's signals, so the
        # fit can run the moment the stop is requested — no waiting on the
        # scan thread, which may still be finishing a grab.
        self._stop_auto_worker()
        self._auto_calibrate_btn.setText("Start Auto Calibrate")
        views = self._auto_views
        if len(views) < AUTO_CALIBRATE_MIN_VIEWS:
            self._auto_status.setText(
                f"Stopped with only {len(views)} view(s) captured — need at least "
                f"{AUTO_CALIBRATE_MIN_VIEWS} at different poses; nothing computed."
            )
            return
        if self._frame is None:
            self._auto_status.setText("Stopped before any frame was captured; nothing computed.")
            return

        height, width = self._frame.shape[:2]
        try:
            lens = CameraCalibration.calibrate_lens(views, (width, height))
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Auto Calibrate", str(exc))
            return

        last = views[-1]
        undistorted_pixels = CameraCalibration.undistort_points(
            last.pixel_points, lens.camera_matrix, lens.dist_coeffs
        )
        try:
            self._homography, self._rms = CameraCalibration.compute_homography(
                undistorted_pixels, last.mm_points
            )
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Auto Calibrate", str(exc))
            return

        self._camera_matrix = lens.camera_matrix
        self._dist_coeffs = lens.dist_coeffs

        columns, rows = self._board_columns.value(), self._board_rows.value()
        grid = np.asarray(undistorted_pixels, dtype=np.float64).reshape(rows, columns, 2)
        square_mm = self._board_square_mm.value()
        self._ppmm_x.setValue(
            float(np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2).mean()) / square_mm
        )
        self._ppmm_y.setValue(
            float(np.linalg.norm(grid[1:, :, :] - grid[:-1, :, :], axis=2).mean()) / square_mm
        )

        self._hom_status.setText(
            f"auto-calibrated (lens rms {lens.overall_rms_px:.3f} px over {len(views)} "
            f"views, homography rms {self._rms:.3f} mm)"
        )
        self._auto_status.setText(
            f"Done — {len(views)} views, lens rms {lens.overall_rms_px:.3f} px."
        )
        self._status.setText(
            f"Auto-calibrated from {len(views)} checkerboard views "
            f"(lens rms {lens.overall_rms_px:.3f} px, homography rms {self._rms:.3f} mm). "
            f"Remove the board, place the part, and redo Step 2 (reference) before saving."
        )

    def _on_ruler_pick_toggled(self, checked: bool) -> None:
        if checked and self._frame is None:
            QMessageBox.information(self, "Ruler", "Capture a frame first.")
            self._ruler_pick_btn.setChecked(False)
            return
        self._view.set_pick_mode(checked)

    def _on_ruler_points_changed(self, points: list) -> None:
        self._ruler_points_label.setText(f"Points: {len(points)}")

    def _on_clear_ruler_points(self) -> None:
        self._view.clear_points()

    def _on_compute_ruler_scale(self) -> None:
        points = self._view.points()
        if len(points) < RULER_MIN_POINTS:
            QMessageBox.warning(
                self, "Ruler",
                f"Place at least {RULER_MIN_POINTS} points, each one division "
                f"apart, then Compute.",
            )
            return
        if len(points) > RULER_MAX_POINTS:
            QMessageBox.warning(self, "Ruler", f"Use at most {RULER_MAX_POINTS} points.")
            return
        spacing_mm = self._ruler_spacing_mm.value()
        try:
            ppmm = CameraCalibration.scale_from_points(points, spacing_mm)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Ruler", str(exc))
            return
        self._ppmm_x.setValue(ppmm)
        self._ppmm_y.setValue(ppmm)
        self._status.setText(
            f"Pixels-per-mm set to {ppmm:.3f} from {len(points)} ruler points "
            f"({spacing_mm:g} mm apart)"
        )

    def _on_compute_scale(self) -> None:
        try:
            ppmm = CameraCalibration.scale_from_distance(
                self._known_px.value(), self._known_mm.value()
            )
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Compute Scale", str(exc))
            return
        self._ppmm_x.setValue(ppmm)
        self._ppmm_y.setValue(ppmm)

    def _on_detect_reference(self) -> None:
        index = self._camera_index()
        if index is None:
            return
        if self._frame is None:
            QMessageBox.information(self, "Reference", "Capture a frame first.")
            return
        try:
            result = self._engine.detect(self._frame, index)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Reference", str(exc))
            return
        best = result.best
        if best is None:
            QMessageBox.warning(self, "Reference", "No hole found on the captured frame.")
            return
        x_mm, y_mm = self._current_model().pixel_to_mm(best.x_px, best.y_px)
        self._ref_x.setValue(x_mm)
        self._ref_y.setValue(y_mm)
        self._view.set_frame(draw_detection_overlay(self._frame, result))
        self._status.setText(
            f"Reference set from detection: ({x_mm:.2f}, {y_mm:.2f}) mm "
            f"(conf {best.confidence:.2f})"
        )

    def _on_compute_homography(self) -> None:
        pixel_points = [(s[0].value(), s[1].value()) for s in self._hom_points]
        mm_points = [(s[2].value(), s[3].value()) for s in self._hom_points]
        try:
            self._homography, self._rms = CameraCalibration.compute_homography(
                pixel_points, mm_points
            )
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Homography", str(exc))
            return
        self._hom_status.setText(f"computed (rms {self._rms:.3f} mm)")

    def _on_clear_homography(self) -> None:
        self._homography = None
        self._camera_matrix = None
        self._dist_coeffs = None
        self._rms = 0.0
        self._hom_status.setText("not set")

    def _current_model(self) -> CameraCalibration:
        return CameraCalibration(
            camera_index=self._camera_index() or 0,
            pixels_per_mm_x=self._ppmm_x.value() or 1.0,
            pixels_per_mm_y=self._ppmm_y.value() or 1.0,
            homography=self._homography,
            camera_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeffs,
            ref_point_mm=(self._ref_x.value(), self._ref_y.value()),
            rms_error=self._rms,
        )

    def _on_save(self) -> None:
        index = self._camera_index()
        if index is None:
            return
        if self._homography is None and (
            self._ppmm_x.value() <= 0 or self._ppmm_y.value() <= 0
        ):
            QMessageBox.warning(self, "Save", "Pixels-per-mm must be positive.")
            return
        try:
            self._manager.save(self._current_model())
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Save Calibration", str(exc))
            return
        self._status.setText(f"Calibration saved for camera {index}")

    # --------------------------------------------------------------- events
    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        if self._auto_session_active:
            self._cancel_auto_calibrate_session("page hidden")  # never hammer test_capture off-screen

    def _on_live_test(self) -> None:
        index = self._camera_index()
        if index is None:
            return
        try:
            frame = self._cameras.test_capture(index)
            result = self._engine.detect(frame, index)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Live Test", str(exc))
            return
        self._view.set_frame(draw_detection_overlay(frame, result))
        best = result.best
        if best is None:
            self._status.setText("Live test: no hole found")
            return
        height, width = frame.shape[:2]
        x_mm, y_mm, deviation = self._manager.evaluate(
            index, best.x_px, best.y_px, width, height
        )
        deviation_text = f"{deviation:.2f} mm" if deviation is not None else "n/a (uncalibrated)"
        self._status.setText(
            f"Live test: ({x_mm:.2f}, {y_mm:.2f}) mm — deviation {deviation_text}"
        )

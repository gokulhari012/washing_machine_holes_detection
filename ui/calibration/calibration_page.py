"""Calibration page: pixel→mm scale, reference point, perspective correction.

Workflow (per camera):
0. **Auto Calibrate** (optional, recommended) — place a checkerboard on/near
   the inspection plane, capture, and "Auto Calibrate" detects its corners
   and fills in the scale (Step 1) and homography (Step 3) from dozens of
   correspondences in one shot. Remove the board and redo Step 2 afterwards —
   the reference point must be captured through the fresh homography.
1. **Scale** — enter a known distance in px and mm ("Compute"), or type
   pixels-per-mm directly. Skippable if Auto Calibrate already filled it in.
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

from core.calibration import CalibrationManager, CameraCalibration
from core.utilities.exceptions import VisionSystemError
from core.vision import VisionEngine, draw_detection_overlay
from services.camera_service import CameraService
from ui.widgets import ImageView


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
        auto_calibrate_btn = QPushButton("Capture && Auto Calibrate")
        auto_calibrate_btn.setProperty("class", "primary")
        auto_calibrate_btn.clicked.connect(self._on_auto_calibrate)
        auto_form.addRow("Inner Corners (cols x rows)", board_w)
        auto_form.addRow("Square Size (mm)", self._board_square_mm)
        auto_form.addRow(auto_calibrate_btn)
        hint = QLabel(
            "Place a checkerboard flat on/near the inspection plane, filling "
            "as much of the frame as practical, then run this. Afterwards "
            "remove it, place the real part, and redo Step 2."
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "dim")
        auto_form.addRow(hint)
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
        self._view = ImageView()
        self._view.setMinimumSize(480, 380)
        body.addWidget(self._view, stretch=1)

        self._load_existing()

    # ------------------------------------------------------------- helpers
    def _camera_index(self) -> int | None:
        data = self._camera.currentData()
        return int(data) if data is not None else None

    def _load_existing(self) -> None:
        index = self._camera_index()
        if index is None:
            return
        calibration = self._manager.get(index)
        if calibration is None:
            self._status.setText("No active calibration — identity fallback (1 px = 1 mm)")
            self._homography = None
            self._hom_status.setText("not set")
            return
        self._ppmm_x.setValue(calibration.pixels_per_mm_x)
        self._ppmm_y.setValue(calibration.pixels_per_mm_y)
        self._ref_x.setValue(calibration.ref_point_mm[0])
        self._ref_y.setValue(calibration.ref_point_mm[1])
        self._homography = calibration.homography
        self._rms = calibration.rms_error
        self._hom_status.setText(
            f"active (rms {calibration.rms_error:.3f} mm)"
            if calibration.homography is not None
            else "not set"
        )
        self._status.setText("Active calibration loaded")

    # -------------------------------------------------------------- actions
    def _on_capture(self) -> None:
        index = self._camera_index()
        if index is None:
            return
        try:
            self._frame = self._cameras.test_capture(index)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Capture", str(exc))
            return
        self._view.set_frame(self._frame)
        self._status.setText("Frame captured")

    def _on_auto_calibrate(self) -> None:
        index = self._camera_index()
        if index is None:
            return
        try:
            self._frame = self._cameras.test_capture(index)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Auto Calibrate", str(exc))
            return

        columns = self._board_columns.value()
        rows = self._board_rows.value()
        try:
            detection = CameraCalibration.find_checkerboard(
                self._frame, columns, rows, self._board_square_mm.value()
            )
            self._homography, self._rms = CameraCalibration.compute_homography(
                detection.pixel_points, detection.mm_points
            )
        except VisionSystemError as exc:
            self._view.set_frame(self._frame)
            QMessageBox.warning(self, "Auto Calibrate", str(exc))
            return

        self._ppmm_x.setValue(detection.pixels_per_mm_x)
        self._ppmm_y.setValue(detection.pixels_per_mm_y)
        self._hom_status.setText(
            f"auto-calibrated (rms {self._rms:.3f} mm, {len(detection.pixel_points)} points)"
        )

        overlay = (
            cv2.cvtColor(self._frame, cv2.COLOR_GRAY2BGR)
            if self._frame.ndim == 2 else self._frame.copy()
        )
        cv2.drawChessboardCorners(overlay, (columns, rows), detection.corners_px, True)
        self._view.set_frame(overlay)
        self._status.setText(
            f"Auto-calibrated from {len(detection.pixel_points)} checkerboard corners "
            f"(rms {self._rms:.3f} mm). Remove the board, place the part, and redo "
            f"Step 2 (reference) before saving."
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
        if self._frame is None:
            QMessageBox.information(self, "Reference", "Capture a frame first.")
            return
        try:
            result = self._engine.detect(self._frame)
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
        self._rms = 0.0
        self._hom_status.setText("not set")

    def _current_model(self) -> CameraCalibration:
        return CameraCalibration(
            camera_index=self._camera_index() or 0,
            pixels_per_mm_x=self._ppmm_x.value() or 1.0,
            pixels_per_mm_y=self._ppmm_y.value() or 1.0,
            homography=self._homography,
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

    def _on_live_test(self) -> None:
        index = self._camera_index()
        if index is None:
            return
        try:
            frame = self._cameras.test_capture(index)
            result = self._engine.detect(frame)
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

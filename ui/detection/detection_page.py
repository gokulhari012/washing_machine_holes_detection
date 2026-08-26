"""Detection Settings page.

Edits detection.json: common judgement thresholds, the active strategy, and
per-strategy parameter blocks (only the active block is shown). "Save &
Apply" hot-swaps the running VisionEngine; "Restore Defaults" reloads the
shipped configuration. "Test" captures a frame from the chosen camera, runs
the engine, and shows the annotated result with timing.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.utilities import ConfigManager
from core.utilities.enums import DetectorType
from core.utilities.exceptions import VisionSystemError
from core.vision import (
    DarkHoleDetector,
    DetectionResult,
    Hole,
    OpenCVHoleDetector,
    VisionEngine,
    draw_debug_overlay,
    draw_detection_overlay,
)
from services.camera_service import CameraService
from ui.widgets import RoiEditor

# Strategies the auto sweep can grid-search, and the detector class it builds
# each trial candidate from directly (bypassing the shared VisionEngine, so a
# sweep never mutates the live/production configuration).
_SWEEP_DETECTORS: dict[str, type] = {
    "opencv": OpenCVHoleDetector,
    "dark_hole": DarkHoleDetector,
}


def _dspin(minimum: float, maximum: float, step: float, decimals: int = 2) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    return spin


def _spin(minimum: int, maximum: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    return spin


class DetectionPage(QWidget):
    """All algorithm parameters, strategy switching, live test."""

    def __init__(
        self,
        config_manager: ConfigManager,
        vision_engine: VisionEngine,
        camera_service: CameraService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config_manager
        self._engine = vision_engine
        self._cameras = camera_service

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        title = QLabel("Detection Settings")
        title.setProperty("class", "pageTitle")
        root.addWidget(title)

        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, stretch=1)

        # ------------------------------------------------------- left: form
        left = QVBoxLayout()

        common_box = QGroupBox("Judgement (common)")
        common = QFormLayout(common_box)
        self._confidence = _dspin(0.0, 1.0, 0.05)
        self._expected = _spin(1, 16)
        self._tolerance = _dspin(0.0, 500.0, 0.1, 1)
        self._tolerance.setSuffix(" mm")
        self._tolerance.setToolTip("0 disables the position tolerance check")
        common.addRow("Confidence Threshold", self._confidence)
        common.addRow("Expected Hole Count", self._expected)
        common.addRow("Position Tolerance", self._tolerance)
        left.addWidget(common_box)

        strategy_row = QHBoxLayout()
        strategy_row.addWidget(QLabel("Active Detector"))
        self._strategy = QComboBox()
        self._strategy.addItems([d.value for d in DetectorType])
        self._strategy.currentIndexChanged.connect(
            lambda index: self._stack.setCurrentIndex(index)
        )
        self._strategy.currentTextChanged.connect(self._update_sweep_availability)
        strategy_row.addWidget(self._strategy, stretch=1)
        left.addLayout(strategy_row)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_opencv_form())
        self._stack.addWidget(self._build_template_form())
        self._stack.addWidget(self._build_yolo_form())
        self._stack.addWidget(self._build_dark_hole_form())
        left.addWidget(self._stack)

        # (label, param key, trial values, form spin box) per strategy the
        # sweep supports — the spin box is what "Apply" writes the winner
        # into and what the base (non-swept) parameters are read from.
        self._sweep_specs: dict[str, tuple[tuple, tuple]] = {
            "opencv": (
                ("Threshold", "detection_threshold", list(range(20, 201, 15)), self._cv_threshold),
                ("Blur", "blur_kernel_size", [3, 5, 7], self._cv_blur),
            ),
            "dark_hole": (
                ("Min Contrast", "min_contrast", list(range(5, 61, 5)), self._dh_min_contrast),
                ("Blur", "blur_kernel_size", [1, 3, 5], self._dh_blur),
            ),
        }

        buttons = QHBoxLayout()
        save_btn = QPushButton("Save && Apply")
        save_btn.setProperty("class", "primary")
        save_btn.clicked.connect(self._on_save)
        defaults_btn = QPushButton("Restore Defaults")
        defaults_btn.clicked.connect(self._on_defaults)
        buttons.addWidget(save_btn)
        buttons.addWidget(defaults_btn)
        left.addLayout(buttons)
        left.addStretch()
        body.addLayout(left)

        # ------------------------------------------------------ right: test
        right = QVBoxLayout()
        test_row = QHBoxLayout()
        test_row.addWidget(QLabel("Camera"))
        self._test_camera = QComboBox()
        for cfg in self._cameras.get_configs():
            self._test_camera.addItem(f"{cfg['index']}: {cfg.get('name', '')}", cfg["index"])
        test_btn = QPushButton("Test on Camera")
        test_btn.clicked.connect(self._on_test)
        test_row.addWidget(self._test_camera)
        test_row.addWidget(test_btn)
        test_row.addWidget(QLabel("View"))
        self._view_mode = QComboBox()
        self._view_mode.addItems(["Result", "Debug (edges/contours)"])
        self._view_mode.setToolTip(
            "Debug: everything the active strategy's threshold mask / edge map "
            "currently sees, not just the holes it accepted"
        )
        self._view_mode.currentIndexChanged.connect(self._render_preview)
        test_row.addWidget(self._view_mode)
        test_row.addStretch()
        right.addLayout(test_row)

        self._view = RoiEditor()
        self._view.setMinimumSize(480, 360)
        right.addWidget(self._view, stretch=1)
        self._result_label = QLabel("—")
        self._result_label.setProperty("class", "dim")
        right.addWidget(self._result_label)
        right.addWidget(self._build_sweep_box())
        body.addLayout(right, stretch=1)

        self._last_frame = None  # np.ndarray | None — set by a successful Test
        self._last_result = None  # DetectionResult | None
        self._sweep_rows: list[tuple[int, int, Hole]] = []

        self._load()
        self._update_sweep_availability(self._strategy.currentText())

    # ------------------------------------------------------- strategy forms
    def _build_opencv_form(self) -> QWidget:
        box = QGroupBox("OpenCV Parameters")
        form = QFormLayout(box)
        self._cv_threshold = _spin(0, 255)
        self._cv_adaptive = QCheckBox("Adaptive threshold")
        self._cv_blur = _spin(1, 31)
        self._cv_edge_low = _spin(0, 500)
        self._cv_edge_high = _spin(0, 500)
        self._cv_morph_op = QComboBox()
        self._cv_morph_op.addItems(["close", "open", "none"])
        self._cv_morph_kernel = _spin(1, 31)
        self._cv_morph_iter = _spin(1, 10)
        self._cv_min_diameter = _spin(1, 4000)
        self._cv_max_diameter = _spin(1, 4000)
        self._cv_circularity = _dspin(0.0, 1.0, 0.05)
        self._cv_aspect_ratio = _dspin(0.0, 1.0, 0.05)
        self._cv_aspect_ratio.setToolTip(
            "Minor/major axis of the fitted ellipse — rejects scratches and "
            "shadow streaks a round-hole gate alone would miss"
        )
        self._cv_contour_mode = QComboBox()
        self._cv_contour_mode.addItems(["external", "list", "tree"])
        form.addRow("Detection Threshold", self._cv_threshold)
        form.addRow("", self._cv_adaptive)
        form.addRow("Blur Kernel", self._cv_blur)
        form.addRow("Edge Threshold Low", self._cv_edge_low)
        form.addRow("Edge Threshold High", self._cv_edge_high)
        form.addRow("Morphology", self._cv_morph_op)
        form.addRow("Morph Kernel", self._cv_morph_kernel)
        form.addRow("Morph Iterations", self._cv_morph_iter)
        form.addRow("Min Hole Diameter (px)", self._cv_min_diameter)
        form.addRow("Max Hole Diameter (px)", self._cv_max_diameter)
        form.addRow("Min Circularity", self._cv_circularity)
        form.addRow("Min Aspect Ratio", self._cv_aspect_ratio)
        form.addRow("Contour Mode", self._cv_contour_mode)
        return box

    def _build_template_form(self) -> QWidget:
        box = QGroupBox("Template Matching Parameters")
        form = QFormLayout(box)
        path_row = QHBoxLayout()
        self._tm_path = QLineEdit()
        browse = QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(lambda: self._browse(self._tm_path, "Images (*.png *.jpg *.bmp)"))
        path_row.addWidget(self._tm_path)
        path_row.addWidget(browse)
        path_w = QWidget()
        path_w.setLayout(path_row)
        self._tm_threshold = _dspin(0.0, 1.0, 0.05)
        self._tm_method = QComboBox()
        self._tm_method.addItems(["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"])
        form.addRow("Template Image", path_w)
        form.addRow("Match Threshold", self._tm_threshold)
        form.addRow("Method", self._tm_method)
        return box

    def _build_yolo_form(self) -> QWidget:
        box = QGroupBox("YOLO Parameters")
        form = QFormLayout(box)
        path_row = QHBoxLayout()
        self._yolo_path = QLineEdit()
        browse = QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(lambda: self._browse(self._yolo_path, "Models (*.pt *.onnx)"))
        path_row.addWidget(self._yolo_path)
        path_row.addWidget(browse)
        path_w = QWidget()
        path_w.setLayout(path_row)
        self._yolo_conf = _dspin(0.0, 1.0, 0.05)
        self._yolo_iou = _dspin(0.0, 1.0, 0.05)
        self._yolo_class = _spin(0, 999)
        form.addRow("Model Weights", path_w)
        form.addRow("Confidence", self._yolo_conf)
        form.addRow("IoU Threshold", self._yolo_iou)
        form.addRow("Class ID", self._yolo_class)
        return box

    def _build_dark_hole_form(self) -> QWidget:
        box = QGroupBox("Dark Hole Parameters")
        form = QFormLayout(box)
        self._dh_channel = QComboBox()
        self._dh_channel.addItems(["auto", "gray", "red", "green", "blue"])
        self._dh_channel.setToolTip(
            "Which channel carries the signal. 'auto' picks the widest-spread "
            "channel — the right choice under a red or IR ring light"
        )
        self._dh_blur = _spin(1, 31)
        self._dh_min_contrast = _spin(1, 255)
        self._dh_min_contrast.setToolTip(
            "How many grey levels darker than its surroundings a bore must be"
        )
        self._dh_otsu = QCheckBox("Also raise the threshold automatically (Otsu)")
        self._dh_morph = _spin(1, 31)
        self._dh_min_diameter = _spin(1, 4000)
        self._dh_max_diameter = _spin(1, 4000)
        self._dh_fill = _dspin(0.05, 1.0, 0.05)
        self._dh_fill.setToolTip(
            "Smallest visible share of the bore that still counts — 0.35 accepts "
            "a bore whose rim is two thirds hidden"
        )
        self._dh_fit_error = _dspin(0.05, 1.0, 0.05)
        self._dh_fit_error.setToolTip(
            "How far the rim may stray from a circle (fraction of the radius)"
        )
        form.addRow("Channel", self._dh_channel)
        form.addRow("Blur Kernel", self._dh_blur)
        form.addRow("Min Contrast", self._dh_min_contrast)
        form.addRow("", self._dh_otsu)
        form.addRow("Morph Kernel", self._dh_morph)
        form.addRow("Min Hole Diameter (px)", self._dh_min_diameter)
        form.addRow("Max Hole Diameter (px)", self._dh_max_diameter)
        form.addRow("Min Visible Fraction", self._dh_fill)
        form.addRow("Max Fit Error", self._dh_fit_error)
        return box

    def _build_sweep_box(self) -> QWidget:
        box = QGroupBox("Auto Sweep")
        layout = QVBoxLayout(box)

        hint = QLabel(
            "Grid-searches the active strategy's most sensitive parameters "
            "against the last 'Test on Camera' frame. Draw an ROI around the "
            "hole first to score only candidates found inside it — otherwise "
            "the single best candidate anywhere in the frame is scored, which "
            "can be fooled by texture on a hole-free part."
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "dim")
        layout.addWidget(hint)

        roi_row = QHBoxLayout()
        roi_mode = QCheckBox("Draw ROI (mark the hole)")
        roi_mode.toggled.connect(self._view.set_roi_mode)
        clear_roi_btn = QPushButton("Clear ROI")
        clear_roi_btn.clicked.connect(self._view.clear_roi)
        roi_row.addWidget(roi_mode)
        roi_row.addWidget(clear_roi_btn)
        roi_row.addStretch()
        layout.addLayout(roi_row)

        run_row = QHBoxLayout()
        self._sweep_btn = QPushButton("Run Auto Sweep")
        self._sweep_btn.clicked.connect(self._on_run_sweep)
        apply_btn = QPushButton("Apply Best")
        apply_btn.clicked.connect(self._on_apply_best_sweep)
        run_row.addWidget(self._sweep_btn)
        run_row.addWidget(apply_btn)
        run_row.addStretch()
        layout.addLayout(run_row)

        self._sweep_status = QLabel("Run 'Test on Camera' first.")
        self._sweep_status.setWordWrap(True)
        self._sweep_status.setProperty("class", "dim")
        layout.addWidget(self._sweep_status)

        self._sweep_table = QTableWidget(0, 5)
        self._sweep_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._sweep_table.horizontalHeader().setStretchLastSection(True)
        self._sweep_table.setMaximumHeight(200)
        self._sweep_table.cellClicked.connect(self._on_sweep_row_clicked)
        layout.addWidget(self._sweep_table)

        return box

    def _browse(self, target: QLineEdit, name_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", name_filter)
        if path:
            target.setText(path)

    # ------------------------------------------------------------ load/save
    def _load(self) -> None:
        cfg = self._config.load("detection")
        common = cfg.get("common", {})
        self._confidence.setValue(float(common.get("confidence_threshold", 0.6)))
        self._expected.setValue(int(common.get("expected_hole_count", 1)))
        self._tolerance.setValue(float(common.get("position_tolerance_mm", 0.0)))
        self._strategy.setCurrentText(cfg.get("active_detector", "opencv"))

        opencv = cfg.get("opencv", {})
        self._cv_threshold.setValue(int(opencv.get("detection_threshold", 60)))
        self._cv_adaptive.setChecked(bool(opencv.get("adaptive_threshold", False)))
        self._cv_blur.setValue(int(opencv.get("blur_kernel_size", 5)))
        self._cv_edge_low.setValue(int(opencv.get("edge_threshold_low", 50)))
        self._cv_edge_high.setValue(int(opencv.get("edge_threshold_high", 150)))
        self._cv_morph_op.setCurrentText(opencv.get("morphology_operation", "close"))
        self._cv_morph_kernel.setValue(int(opencv.get("morphology_kernel_size", 5)))
        self._cv_morph_iter.setValue(int(opencv.get("morphology_iterations", 1)))
        self._cv_min_diameter.setValue(int(opencv.get("min_hole_diameter_px", 20)))
        self._cv_max_diameter.setValue(int(opencv.get("max_hole_diameter_px", 200)))
        self._cv_circularity.setValue(float(opencv.get("min_circularity", 0.7)))
        self._cv_aspect_ratio.setValue(float(opencv.get("min_aspect_ratio", 0.35)))
        self._cv_contour_mode.setCurrentText(opencv.get("contour_retrieval_mode", "external"))

        template = cfg.get("template_matching", {})
        self._tm_path.setText(template.get("template_path", ""))
        self._tm_threshold.setValue(float(template.get("match_threshold", 0.8)))
        self._tm_method.setCurrentText(template.get("method", "TM_CCOEFF_NORMED"))

        dark = cfg.get("dark_hole", {})
        self._dh_channel.setCurrentText(str(dark.get("channel", "auto")))
        self._dh_blur.setValue(int(dark.get("blur_kernel_size", 3)))
        self._dh_min_contrast.setValue(int(dark.get("min_contrast", 18)))
        self._dh_otsu.setChecked(bool(dark.get("use_otsu", True)))
        self._dh_morph.setValue(int(dark.get("morphology_kernel_size", 3)))
        self._dh_min_diameter.setValue(int(dark.get("min_hole_diameter_px", 15)))
        self._dh_max_diameter.setValue(int(dark.get("max_hole_diameter_px", 120)))
        self._dh_fill.setValue(float(dark.get("min_fill_ratio", 0.35)))
        self._dh_fit_error.setValue(float(dark.get("max_fit_error", 0.25)))

        yolo = cfg.get("yolo", {})
        self._yolo_path.setText(yolo.get("model_path", ""))
        self._yolo_conf.setValue(float(yolo.get("confidence", 0.5)))
        self._yolo_iou.setValue(float(yolo.get("iou_threshold", 0.45)))
        self._yolo_class.setValue(int(yolo.get("class_id", 0)))

    def _collect(self) -> dict:
        return {
            "active_detector": self._strategy.currentText(),
            "common": {
                "confidence_threshold": self._confidence.value(),
                "expected_hole_count": self._expected.value(),
                "position_tolerance_mm": self._tolerance.value(),
            },
            "opencv": {
                "detection_threshold": self._cv_threshold.value(),
                "adaptive_threshold": self._cv_adaptive.isChecked(),
                "blur_kernel_size": self._cv_blur.value(),
                "edge_threshold_low": self._cv_edge_low.value(),
                "edge_threshold_high": self._cv_edge_high.value(),
                "morphology_operation": self._cv_morph_op.currentText(),
                "morphology_kernel_size": self._cv_morph_kernel.value(),
                "morphology_iterations": self._cv_morph_iter.value(),
                "min_hole_diameter_px": self._cv_min_diameter.value(),
                "max_hole_diameter_px": self._cv_max_diameter.value(),
                "min_circularity": self._cv_circularity.value(),
                "min_aspect_ratio": self._cv_aspect_ratio.value(),
                "contour_retrieval_mode": self._cv_contour_mode.currentText(),
            },
            "template_matching": {
                "template_path": self._tm_path.text().strip(),
                "match_threshold": self._tm_threshold.value(),
                "method": self._tm_method.currentText(),
            },
            "dark_hole": {
                "channel": self._dh_channel.currentText(),
                "blur_kernel_size": self._dh_blur.value(),
                "min_contrast": self._dh_min_contrast.value(),
                "use_otsu": self._dh_otsu.isChecked(),
                "morphology_kernel_size": self._dh_morph.value(),
                "min_hole_diameter_px": self._dh_min_diameter.value(),
                "max_hole_diameter_px": self._dh_max_diameter.value(),
                "min_fill_ratio": self._dh_fill.value(),
                "max_fit_error": self._dh_fit_error.value(),
            },
            "yolo": {
                "model_path": self._yolo_path.text().strip(),
                "confidence": self._yolo_conf.value(),
                "iou_threshold": self._yolo_iou.value(),
                "class_id": self._yolo_class.value(),
            },
        }

    # -------------------------------------------------------------- actions
    def _on_save(self) -> None:
        cfg = self._collect()
        try:
            self._engine.apply_config(cfg)  # validate + hot-swap first
            self._config.save("detection", cfg)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Save & Apply", str(exc))
            return
        QMessageBox.information(self, "Save & Apply", "Detection parameters applied.")

    def _on_defaults(self) -> None:
        try:
            cfg = self._config.restore_defaults("detection")
            self._engine.apply_config(cfg)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Restore Defaults", str(exc))
            return
        self._load()

    def _on_test(self) -> None:
        camera_index = self._test_camera.currentData()
        if camera_index is None:
            return
        try:
            frame = self._cameras.test_capture(int(camera_index))
            self._engine.apply_config(self._collect())  # test what's on screen
            result = self._engine.detect(frame)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Test", str(exc))
            return
        self._last_frame = frame
        self._last_result = result
        self._render_preview()
        best = result.best
        if best is not None:
            self._result_label.setText(
                f"{len(result.holes)} hole(s) — best ({best.x_px:.1f}, {best.y_px:.1f}) px, "
                f"Ø {best.diameter_px:.1f} px, conf {best.confidence:.2f} "
                f"— {result.processing_ms:.1f} ms"
            )
        else:
            self._result_label.setText(f"No hole found — {result.processing_ms:.1f} ms")

    def _render_preview(self) -> None:
        """Redraw the last captured frame in whichever mode 'View' is set to.

        Both modes reuse the frame/result from the last "Test on Camera" —
        switching the view combo never itself pushes the on-screen parameters
        to the (production-shared) engine; only "Test" and "Save & Apply" do.
        """
        if self._last_frame is None:
            return
        if self._view_mode.currentIndex() == 1:  # Debug (edges/contours)
            try:
                stages = self._engine.debug_stages(self._last_frame)
            except VisionSystemError as exc:
                QMessageBox.warning(self, "Debug View", str(exc))
                return
            self._view.set_frame(draw_debug_overlay(self._last_frame, stages))
        else:
            self._view.set_frame(draw_detection_overlay(self._last_frame, self._last_result))

    # ---------------------------------------------------------- auto sweep
    def _update_sweep_availability(self, strategy: str) -> None:
        supported = strategy in self._sweep_specs
        self._sweep_btn.setEnabled(supported)
        self._sweep_btn.setToolTip(
            "" if supported else f"Auto sweep is not available for the '{strategy}' strategy"
        )

    def _on_run_sweep(self) -> None:
        if self._last_frame is None:
            QMessageBox.information(self, "Auto Sweep", "Run 'Test on Camera' first.")
            return
        strategy = self._strategy.currentText()
        spec = self._sweep_specs.get(strategy)
        if spec is None:
            QMessageBox.information(
                self, "Auto Sweep", f"Auto sweep is not available for the '{strategy}' strategy."
            )
            return

        (primary_label, primary_key, primary_values, _pw), \
            (secondary_label, secondary_key, secondary_values, _sw) = spec
        detector_cls = _SWEEP_DETECTORS[strategy]
        base_params = self._collect().get(strategy, {})
        roi = self._view.current_roi()

        rows: list[tuple[int, int, Hole]] = []
        for primary in primary_values:
            for secondary in secondary_values:
                params = dict(base_params, **{primary_key: primary, secondary_key: secondary})
                try:
                    result = detector_cls(params).detect(self._last_frame)
                except VisionSystemError:
                    continue  # this combination is not a valid configuration — skip it
                hole = self._best_in_roi(result.holes, roi)
                if hole is not None:
                    rows.append((primary, secondary, hole))

        tried = len(primary_values) * len(secondary_values)
        self._sweep_rows = sorted(rows, key=lambda item: item[2].confidence, reverse=True)[:20]
        self._fill_sweep_table(primary_label, secondary_label)

        if not self._sweep_rows:
            where = " in the ROI" if roi[2] > 0 and roi[3] > 0 else ""
            self._sweep_status.setText(
                f"No combination found a hole{where} ({tried} tried) — widen the "
                f"diameter gates, or adjust/clear the ROI."
            )
            return

        best_primary, best_secondary, best_hole = self._sweep_rows[0]
        self._sweep_status.setText(
            f"Best: {primary_label}={best_primary}, {secondary_label}={best_secondary} — "
            f"confidence {best_hole.confidence:.2f}, Ø {best_hole.diameter_px:.1f} px "
            f"({len(self._sweep_rows)} of {tried} combos found it — click a row to preview, "
            f"'Apply Best' to load it into the form)"
        )

    @staticmethod
    def _best_in_roi(holes: list[Hole], roi: tuple[int, int, int, int]) -> Hole | None:
        """Highest-confidence hole whose centre falls inside *roi*.

        ``holes`` is already sorted best-first, so the first match found is
        the best one; no ROI (``w``/``h`` <= 0) falls back to the single best
        candidate anywhere in the frame.
        """
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            return holes[0] if holes else None
        for hole in holes:
            if x <= hole.x_px <= x + w and y <= hole.y_px <= y + h:
                return hole
        return None

    def _fill_sweep_table(self, primary_label: str, secondary_label: str) -> None:
        self._sweep_table.setHorizontalHeaderLabels(
            [primary_label, secondary_label, "Confidence", "Ø px", "Circ."]
        )
        self._sweep_table.setRowCount(len(self._sweep_rows))
        for row_index, (primary, secondary, hole) in enumerate(self._sweep_rows):
            values = [
                str(primary), str(secondary),
                f"{hole.confidence:.2f}", f"{hole.diameter_px:.1f}", f"{hole.circularity:.2f}",
            ]
            for col, value in enumerate(values):
                self._sweep_table.setItem(row_index, col, QTableWidgetItem(value))

    def _on_sweep_row_clicked(self, row: int, _column: int) -> None:
        """Preview that combination's result, without touching the live engine."""
        if self._last_frame is None or row >= len(self._sweep_rows):
            return
        _primary, _secondary, hole = self._sweep_rows[row]
        self._view.set_frame(
            draw_detection_overlay(self._last_frame, DetectionResult(holes=[hole]))
        )

    def _on_apply_best_sweep(self) -> None:
        if not self._sweep_rows:
            return
        self._apply_sweep_row(self._sweep_rows[0])

    def _apply_sweep_row(self, row: tuple[int, int, Hole]) -> None:
        primary, secondary, _hole = row
        spec = self._sweep_specs.get(self._strategy.currentText())
        if spec is None:
            return
        (primary_label, _pk, _pv, primary_widget), \
            (secondary_label, _sk, _sv, secondary_widget) = spec
        primary_widget.setValue(primary)
        secondary_widget.setValue(secondary)
        self._sweep_status.setText(
            f"Applied {primary_label}={primary}, {secondary_label}={secondary} to the form "
            f"— review and press 'Save && Apply' to make it live."
        )

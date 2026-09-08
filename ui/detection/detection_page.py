"""Detection Settings page — per camera.

Every camera has its own independent detection configuration: its own active
strategy, that strategy's parameters, and its own common judgement
thresholds (confidence, expected hole count, position tolerance). The
"Camera" selector at the top chooses which camera's block is being edited
*and* which camera "Test on Camera" captures from — switching it reloads the
form from that camera's *live* block and discards any unsaved edits on the
form, without touching the other cameras' live detectors.

The form always mirrors what the engine is running, never detection.json:
a machine-model switch (``MachineModelService.apply_profile``) hot-swaps
every camera's strategy without persisting, so the file holds the previous
model's parameters while the detectors run the new ones. The page reloads
itself on ``AppState.active_machine_model_changed`` for the same reason -
nothing else would tell it the parameters underneath it just changed.

"Save & Apply" hot-swaps only the selected camera's running detector
(``VisionEngine.apply_camera_config``) and persists only that camera's block
into detection.json's ``cameras`` map; "Restore Defaults" resets only the
selected camera back to its shipped block. "Test" captures a frame from the
selected camera, runs the engine, and shows the annotated result with timing.

**Auto Sweep** grid-searches (almost) every gating parameter of the active
strategy — not just two — against the last "Test on Camera" frame. Drawing an
ROI around the hole first is now required, not optional: it both scopes which
candidate counts as "found" (only one whose centre falls inside it) and lets
the sweep derive sensible min/max diameter gates from the box's own size, so
those two parameters don't need to be searched at all. The box also bounds
*where* each trial detector actually runs — every candidate is evaluated
against a small crop around the ROI (see ``_crop_around_roi``), not the full
frame, which is what keeps a several-thousand-combination grid (see
``_SWEEP_LEVELS``) finishing in seconds rather than minutes. "Apply Best"
writes every one of the winning combination's parameters into the form, not
just two spin boxes.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtWidgets import (
    QApplication,
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
    QProgressBar,
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
from core.utilities.exceptions import ConfigurationError, VisionSystemError
from core.vision import (
    DarkHoleDetector,
    DetectionResult,
    Hole,
    OpenCVHoleDetector,
    VisionEngine,
    draw_debug_overlay,
    draw_detection_overlay,
)
from models.app_state import AppState
from services.camera_service import CameraService
from ui.widgets import RoiEditor

# Strategies the auto sweep can grid-search, and the detector class it builds
# each trial candidate from directly (bypassing the shared VisionEngine, so a
# sweep never mutates the live/production configuration).
_SWEEP_DETECTORS: dict[str, type] = {
    "opencv": OpenCVHoleDetector,
    "dark_hole": DarkHoleDetector,
}

# Auto Sweep discretization: every value each gating parameter is tried at.
# min_hole_diameter_px/max_hole_diameter_px are deliberately absent — they are
# derived from the drawn ROI instead (see _diameter_bounds_from_roi), and
# opencv's edge_threshold_low/high are absent because they only weight
# confidence (20%) rather than gating acceptance, so sweeping them would add a
# lot of combinations for very little benefit; both stay at the form's current
# value. Levels are a deliberately bounded, representative spread — a true
# "every real-valued setting" search is infinite — chosen so the full grid
# (a few thousand combinations per strategy) finishes in seconds against the
# ROI-cropped image; use "Test on Camera" + a tighter ROI and re-run to refine
# further around a promising region.
_SWEEP_LEVELS: dict[str, dict[str, list]] = {
    "opencv": {
        "detection_threshold": [20, 45, 70, 95, 120, 150, 190],
        "blur_kernel_size": [1, 3, 5, 7],
        "morphology_operation": ["close", "open", "none"],
        "morphology_kernel_size": [3, 5],
        "morphology_iterations": [1, 2],
        "min_circularity": [0.3, 0.5, 0.7],
        "min_aspect_ratio": [0.15, 0.3, 0.45],
        "contour_retrieval_mode": ["external", "list", "tree"],
    },
    "dark_hole": {
        "channel": ["auto", "gray", "red", "green", "blue"],
        "blur_kernel_size": [1, 3, 5, 7],
        "min_contrast": [5, 12, 20, 32, 48, 70],
        "use_otsu": [True, False],
        "morphology_kernel_size": [3, 5, 7],
        "min_fill_ratio": [0.2, 0.35, 0.5, 0.65],
        "max_fit_error": [0.15, 0.25, 0.4],
    },
}


def _opencv_param_grid(base: dict) -> list[dict]:
    """Every combination the opencv sweep tries, seeded with *base* (the
    form's current values, so unswept keys like edge thresholds pass through).

    ``adaptive_threshold`` is tried both ways; ``detection_threshold`` only
    varies in the non-adaptive branch, since adaptive mode ignores it
    entirely — sweeping it there would be pure waste. Likewise
    ``morphology_kernel_size``/``morphology_iterations`` only vary when
    ``morphology_operation`` is not "none", which has no kernel to vary.
    """
    levels = _SWEEP_LEVELS["opencv"]
    combos: list[dict] = []
    for adaptive in (False, True):
        thresholds = (
            levels["detection_threshold"]
            if not adaptive
            else [base.get("detection_threshold", 60)]
        )
        for threshold in thresholds:
            for blur in levels["blur_kernel_size"]:
                for morph_op in levels["morphology_operation"]:
                    is_none = morph_op == "none"
                    kernels = [levels["morphology_kernel_size"][0]] if is_none \
                        else levels["morphology_kernel_size"]
                    iterations_values = [1] if is_none else levels["morphology_iterations"]
                    for kernel in kernels:
                        for iterations in iterations_values:
                            for circularity in levels["min_circularity"]:
                                for aspect in levels["min_aspect_ratio"]:
                                    for contour_mode in levels["contour_retrieval_mode"]:
                                        combos.append({
                                            **base,
                                            "adaptive_threshold": adaptive,
                                            "detection_threshold": threshold,
                                            "blur_kernel_size": blur,
                                            "morphology_operation": morph_op,
                                            "morphology_kernel_size": kernel,
                                            "morphology_iterations": iterations,
                                            "min_circularity": circularity,
                                            "min_aspect_ratio": aspect,
                                            "contour_retrieval_mode": contour_mode,
                                        })
    return combos


def _dark_hole_param_grid(base: dict) -> list[dict]:
    """Every combination the dark_hole sweep tries, seeded with *base*."""
    levels = _SWEEP_LEVELS["dark_hole"]
    combos: list[dict] = []
    for channel in levels["channel"]:
        for blur in levels["blur_kernel_size"]:
            for min_contrast in levels["min_contrast"]:
                for use_otsu in levels["use_otsu"]:
                    for kernel in levels["morphology_kernel_size"]:
                        for min_fill in levels["min_fill_ratio"]:
                            for max_fit_error in levels["max_fit_error"]:
                                combos.append({
                                    **base,
                                    "channel": channel,
                                    "blur_kernel_size": blur,
                                    "min_contrast": min_contrast,
                                    "use_otsu": use_otsu,
                                    "morphology_kernel_size": kernel,
                                    "min_fill_ratio": min_fill,
                                    "max_fit_error": max_fit_error,
                                })
    return combos


_SWEEP_GRID_BUILDERS = {
    "opencv": _opencv_param_grid,
    "dark_hole": _dark_hole_param_grid,
}

_SWEEP_PROGRESS_TICK = 25  # UI refresh cadence — every Nth trial, not every one


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
    """Per-camera algorithm parameters, strategy switching, live test."""

    def __init__(
        self,
        config_manager: ConfigManager,
        vision_engine: VisionEngine,
        camera_service: CameraService,
        app_state: AppState,
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

        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera"))
        self._camera = QComboBox()
        for cfg in self._cameras.get_effective_configs():
            self._camera.addItem(f"{cfg['index']}: {cfg.get('name', '')}", cfg["index"])
        self._camera.currentIndexChanged.connect(self._on_camera_changed)
        app_state.active_machine_model_changed.connect(self._on_machine_model_applied)
        camera_row.addWidget(self._camera, stretch=1)
        left.addLayout(camera_row)

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

        # param key -> form widget, per sweepable strategy — lets "Apply Best"
        # write an arbitrary winning combination back onto the form generically
        # (see _set_widget_value) instead of two hardcoded spin boxes.
        self._param_widgets: dict[str, dict[str, QWidget]] = {
            "opencv": {
                "detection_threshold": self._cv_threshold,
                "adaptive_threshold": self._cv_adaptive,
                "blur_kernel_size": self._cv_blur,
                "morphology_operation": self._cv_morph_op,
                "morphology_kernel_size": self._cv_morph_kernel,
                "morphology_iterations": self._cv_morph_iter,
                "min_circularity": self._cv_circularity,
                "min_aspect_ratio": self._cv_aspect_ratio,
                "contour_retrieval_mode": self._cv_contour_mode,
                "min_hole_diameter_px": self._cv_min_diameter,
                "max_hole_diameter_px": self._cv_max_diameter,
            },
            "dark_hole": {
                "channel": self._dh_channel,
                "blur_kernel_size": self._dh_blur,
                "min_contrast": self._dh_min_contrast,
                "use_otsu": self._dh_otsu,
                "morphology_kernel_size": self._dh_morph,
                "min_fill_ratio": self._dh_fill,
                "max_fit_error": self._dh_fit_error,
                "min_hole_diameter_px": self._dh_min_diameter,
                "max_hole_diameter_px": self._dh_max_diameter,
            },
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
        test_btn = QPushButton("Test on Camera")
        test_btn.setToolTip("Captures from, and tests, whichever camera is selected above")
        test_btn.clicked.connect(self._on_test)
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
        self._last_camera_index: int | None = None  # which camera _last_frame/_last_result are for
        self._sweep_rows: list[tuple[dict, Hole]] = []  # (full param dict, hole) — best first
        self._sweep_cancel_requested = False

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
            "Draw an ROI around the hole (required) — the sweep only counts a "
            "candidate whose centre falls inside it, derives min/max diameter "
            "gates from the box's own size, and evaluates every trial against "
            "just a crop around it. It then grid-searches every other gating "
            "parameter of the active strategy against the last 'Test on "
            "Camera' frame — thousands of combinations, not just two."
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
        self._sweep_cancel_btn = QPushButton("Cancel")
        self._sweep_cancel_btn.setEnabled(False)
        self._sweep_cancel_btn.clicked.connect(self._on_cancel_sweep)
        apply_btn = QPushButton("Apply Best")
        apply_btn.clicked.connect(self._on_apply_best_sweep)
        run_row.addWidget(self._sweep_btn)
        run_row.addWidget(self._sweep_cancel_btn)
        run_row.addWidget(apply_btn)
        run_row.addStretch()
        layout.addLayout(run_row)

        self._sweep_progress = QProgressBar()
        self._sweep_progress.setVisible(False)
        layout.addWidget(self._sweep_progress)

        self._sweep_status = QLabel("Run 'Test on Camera' first.")
        self._sweep_status.setWordWrap(True)
        self._sweep_status.setProperty("class", "dim")
        layout.addWidget(self._sweep_status)

        self._sweep_table = QTableWidget(0, 0)
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
    def _camera_index(self) -> int | None:
        data = self._camera.currentData()
        return int(data) if data is not None else None

    def _load(self) -> None:
        """Fill the form from the block that camera's detector is *running*.

        Not from detection.json: ``VisionEngine.apply_config`` /
        ``apply_camera_config`` are "preview, don't persist" entry points, so
        after a machine-model switch (``MachineModelService.apply_profile``)
        the file still holds the previous model's parameters while the engine
        runs the new ones. Showing the file there would describe a detector
        nothing is using. Falls back to detection.json for a camera the
        engine has no strategy for.
        """
        camera_index = self._camera_index()
        if camera_index is None:
            return
        cfg = self._engine.camera_config(camera_index)
        if cfg is None:
            document = self._config.load("detection")
            cfg = document.get("cameras", {}).get(str(camera_index), {})
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
        camera_index = self._camera_index()
        if camera_index is None:
            return
        cfg = self._collect()
        try:
            self._engine.apply_camera_config(camera_index, cfg)  # validate + hot-swap first
            document = self._config.load("detection")
            document.setdefault("cameras", {})[str(camera_index)] = cfg
            self._config.save("detection", document)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Save & Apply", str(exc))
            return
        QMessageBox.information(
            self, "Save & Apply", f"Detection parameters applied to camera {camera_index}."
        )

    def _on_defaults(self) -> None:
        camera_index = self._camera_index()
        if camera_index is None:
            return
        try:
            defaults_document = self._config.load_defaults("detection")
            camera_defaults = defaults_document.get("cameras", {}).get(str(camera_index))
            if camera_defaults is None:
                raise ConfigurationError(f"No shipped defaults for camera {camera_index}")
            self._engine.apply_camera_config(camera_index, camera_defaults)
            document = self._config.load("detection")
            document.setdefault("cameras", {})[str(camera_index)] = camera_defaults
            self._config.save("detection", document)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Restore Defaults", str(exc))
            return
        self._load()

    def _on_test(self) -> None:
        camera_index = self._camera_index()
        if camera_index is None:
            return
        try:
            frame = self._cameras.test_capture(camera_index)
            self._engine.apply_camera_config(camera_index, self._collect())  # test what's on screen
            result = self._engine.detect(frame, camera_index)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Test", str(exc))
            return
        self._last_frame = frame
        self._last_result = result
        self._last_camera_index = camera_index
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

        Both modes reuse the frame/result/camera from the last "Test on
        Camera" — switching the view combo never itself pushes the on-screen
        parameters to the engine; only "Test" and "Save & Apply" do.
        """
        if self._last_frame is None or self._last_camera_index is None:
            return
        if self._view_mode.currentIndex() == 1:  # Debug (edges/contours)
            try:
                stages = self._engine.debug_stages(self._last_frame, self._last_camera_index)
            except VisionSystemError as exc:
                QMessageBox.warning(self, "Debug View", str(exc))
                return
            self._view.set_frame(draw_debug_overlay(self._last_frame, stages))
        else:
            self._view.set_frame(draw_detection_overlay(self._last_frame, self._last_result))

    def _on_machine_model_applied(self, name: str, plc_code: int) -> None:
        """A machine-model profile was pushed live — re-read the engine.

        ``apply_profile`` hot-swaps every camera's strategy and parameters
        without writing detection.json, so without this the form keeps
        showing the previous model's block. The camera list is rebuilt too,
        since the profile may have renamed the cameras it applied to.
        """
        self._reload_cameras()
        self._load()
        self._update_sweep_availability(self._strategy.currentText())
        self._clear_test_results()

    def _reload_cameras(self) -> None:
        """Refill the camera selector, keeping the current camera selected."""
        selected = self._camera_index()
        blocked = self._camera.blockSignals(True)
        try:
            self._camera.clear()
            for cfg in self._cameras.get_effective_configs():
                self._camera.addItem(f"{cfg['index']}: {cfg.get('name', '')}", cfg["index"])
            position = self._camera.findData(selected)
            self._camera.setCurrentIndex(position if position >= 0 else 0)
        finally:
            self._camera.blockSignals(blocked)

    def _on_camera_changed(self) -> None:
        """Reload the form for the newly selected camera; edits on the previous
        camera's form that were never applied to its detector are discarded
        (matches the Calibration page's camera switch), and any
        Test/Debug/Auto-Sweep result from the previous camera is cleared since
        it no longer matches what's on screen.
        """
        self._load()
        self._update_sweep_availability(self._strategy.currentText())
        self._clear_test_results()

    def _clear_test_results(self) -> None:
        """Drop the last Test/Debug frame and Auto-Sweep table.

        Called whenever the parameters on screen stop describing the run that
        produced them — a camera switch, or a machine-model switch that
        re-tuned the detector underneath the page.
        """
        self._last_frame = None
        self._last_result = None
        self._last_camera_index = None
        self._view.clear_frame()
        self._result_label.setText("—")
        self._sweep_rows = []
        self._sweep_table.setRowCount(0)
        self._sweep_status.setText("Run 'Test on Camera' first.")

    # ---------------------------------------------------------- auto sweep
    def _update_sweep_availability(self, strategy: str) -> None:
        supported = strategy in _SWEEP_GRID_BUILDERS
        self._sweep_btn.setEnabled(supported)
        self._sweep_btn.setToolTip(
            "" if supported else f"Auto sweep is not available for the '{strategy}' strategy"
        )

    def _on_cancel_sweep(self) -> None:
        self._sweep_cancel_requested = True

    def _on_run_sweep(self) -> None:
        if self._last_frame is None:
            QMessageBox.information(self, "Auto Sweep", "Run 'Test on Camera' first.")
            return
        strategy = self._strategy.currentText()
        grid_builder = _SWEEP_GRID_BUILDERS.get(strategy)
        if grid_builder is None:
            QMessageBox.information(
                self, "Auto Sweep", f"Auto sweep is not available for the '{strategy}' strategy."
            )
            return
        roi = self._view.current_roi()
        if roi[2] <= 0 or roi[3] <= 0:
            QMessageBox.information(
                self, "Auto Sweep",
                "Draw an ROI around the hole first (checkbox above) — the sweep "
                "needs it to know where to look and to size the diameter gates.",
            )
            return

        min_diameter, max_diameter = self._diameter_bounds_from_roi(roi)
        base_params = dict(
            self._collect().get(strategy, {}),
            min_hole_diameter_px=min_diameter,
            max_hole_diameter_px=max_diameter,
        )
        combos = grid_builder(base_params)
        cropped, offset_x, offset_y = self._crop_around_roi(self._last_frame, roi)
        detector_cls = _SWEEP_DETECTORS[strategy]

        found: list[tuple[dict, Hole]] = []
        tried = self._run_sweep_grid(combos, detector_cls, cropped, roi, offset_x, offset_y, found)

        self._sweep_rows = sorted(found, key=lambda item: item[1].confidence, reverse=True)[:20]
        self._fill_sweep_table(strategy)

        cancelled = self._sweep_cancel_requested
        note = " (stopped early)" if cancelled else ""
        if not self._sweep_rows:
            self._sweep_status.setText(
                f"No combination found a hole in the ROI ({tried} of {len(combos)} tried{note}) "
                f"— widen the ROI, check it still marks the hole, or verify the part is "
                f"actually visible in this frame."
            )
            return

        _best_params, best_hole = self._sweep_rows[0]
        self._sweep_status.setText(
            f"Best: confidence {best_hole.confidence:.2f}, Ø {best_hole.diameter_px:.1f} px "
            f"(diameter gate {min_diameter:.0f}-{max_diameter:.0f} px from the ROI; "
            f"{len(self._sweep_rows)} of {tried}/{len(combos)} tried found it{note} — click a "
            f"row to preview, 'Apply Best' to load all its parameters into the form)"
        )

    def _run_sweep_grid(
        self,
        combos: list[dict],
        detector_cls: type,
        cropped: np.ndarray,
        roi: tuple[int, int, int, int],
        offset_x: int,
        offset_y: int,
        found: list[tuple[dict, Hole]],
    ) -> int:
        """Run every combination in *combos* against *cropped*, appending
        (params, hole) to *found* for each that lands a candidate inside
        *roi*. Returns how many were actually tried (may be less than
        ``len(combos)`` if cancelled). Keeps the UI responsive and cancellable
        by yielding to the event loop every ``_SWEEP_PROGRESS_TICK`` trials —
        this only runs from the UI thread, on a small ROI crop, so a plain
        loop with periodic ``processEvents()`` is enough; it doesn't warrant
        promoting to a full worker thread for a diagnostic tuning tool.
        """
        self._sweep_cancel_requested = False
        self._sweep_btn.setEnabled(False)
        self._sweep_cancel_btn.setEnabled(True)
        self._sweep_progress.setMaximum(len(combos))
        self._sweep_progress.setValue(0)
        self._sweep_progress.setVisible(True)
        tried = 0
        try:
            for tried, params in enumerate(combos, start=1):
                if self._sweep_cancel_requested:
                    break
                try:
                    result = detector_cls(params).detect(cropped)
                except VisionSystemError:
                    pass  # this combination is not a valid configuration — skip it
                else:
                    holes = [
                        replace(hole, x_px=hole.x_px + offset_x, y_px=hole.y_px + offset_y)
                        for hole in result.holes
                    ]
                    hole = self._best_in_roi(holes, roi)
                    if hole is not None:
                        found.append((params, hole))
                if tried % _SWEEP_PROGRESS_TICK == 0 or tried == len(combos):
                    self._sweep_progress.setValue(tried)
                    self._sweep_status.setText(f"Trying {tried}/{len(combos)} combinations...")
                    QApplication.processEvents()
        finally:
            self._sweep_btn.setEnabled(True)
            self._sweep_cancel_btn.setEnabled(False)
            self._sweep_progress.setVisible(False)
        return tried

    @staticmethod
    def _diameter_bounds_from_roi(roi: tuple[int, int, int, int]) -> tuple[float, float]:
        """Derive min/max hole-diameter gates from the drawn ROI's own size,
        so the sweep doesn't need to grid-search them at all — a generous
        margin either side covers an imprecisely drawn box."""
        _x, _y, width, height = roi
        short_side, long_side = sorted((width, height))
        return max(4.0, short_side * 0.5), max(8.0, long_side * 1.5)

    @staticmethod
    def _crop_around_roi(
        frame: np.ndarray, roi: tuple[int, int, int, int], margin_factor: float = 1.5
    ) -> tuple[np.ndarray, int, int]:
        """Crop *frame* to the ROI plus a margin (context for background/
        annulus sampling), so each of the sweep's thousands of detector calls
        runs against a small image instead of the full frame.

        Returns:
            ``(cropped, offset_x, offset_y)`` — add the offset back to any
            pixel coordinate a detector reports on the crop to recover its
            position in *frame*.
        """
        x, y, width, height = roi
        pad_x, pad_y = int(width * margin_factor), int(height * margin_factor)
        frame_height, frame_width = frame.shape[:2]
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(frame_width, x + width + pad_x), min(frame_height, y + height + pad_y)
        return frame[y0:y1, x0:x1], x0, y0

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

    def _fill_sweep_table(self, strategy: str) -> None:
        columns = list(_SWEEP_LEVELS[strategy])
        headers = columns + ["Confidence", "Ø px", "Circ."]
        self._sweep_table.setColumnCount(len(headers))
        self._sweep_table.setHorizontalHeaderLabels(headers)
        self._sweep_table.setRowCount(len(self._sweep_rows))
        for row_index, (params, hole) in enumerate(self._sweep_rows):
            values = [str(params.get(key)) for key in columns]
            values += [f"{hole.confidence:.2f}", f"{hole.diameter_px:.1f}", f"{hole.circularity:.2f}"]
            for col, value in enumerate(values):
                self._sweep_table.setItem(row_index, col, QTableWidgetItem(value))

    def _on_sweep_row_clicked(self, row: int, _column: int) -> None:
        """Preview that combination's result, without touching the live engine."""
        if self._last_frame is None or row >= len(self._sweep_rows):
            return
        _params, hole = self._sweep_rows[row]
        self._view.set_frame(
            draw_detection_overlay(self._last_frame, DetectionResult(holes=[hole]))
        )

    def _on_apply_best_sweep(self) -> None:
        if not self._sweep_rows:
            return
        self._apply_sweep_row(self._sweep_rows[0])

    def _apply_sweep_row(self, row: tuple[dict, Hole]) -> None:
        params, _hole = row
        widgets = self._param_widgets.get(self._strategy.currentText(), {})
        for key, widget in widgets.items():
            if key in params:
                self._set_widget_value(widget, params[key])
        self._sweep_status.setText(
            "Applied the winning combination to the form — review and press "
            "'Save && Apply' to make it live."
        )

    @staticmethod
    def _set_widget_value(widget: QWidget, value: object) -> None:
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(round(value)))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QComboBox):
            widget.setCurrentText(str(value))
        else:
            raise TypeError(f"Unsupported widget type for sweep apply: {type(widget)}")

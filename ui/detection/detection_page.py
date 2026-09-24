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
If that camera's saved ``led_strobe`` is on (Camera page), "Test" brackets
its capture with ``CameraService.light_on``/``light_off`` on that camera's
configured channel, the same as the Camera page's own Test Camera — there is
no strobe checkbox here, since this page has no unsaved camera-settings form
to read one from; it reads the camera's *live-effective* config instead
(``CameraService.effective_config``), so a machine-model switch is honoured
too. A non-strobe camera is untouched, exactly as before.

Under the result line, "Test" also shows the **judgement** a real cycle would
reach on that frame — GOOD/NG with the hole's deviation from the calibrated
reference point against "Position Tolerance", using exactly the pipeline's
rules (``InspectionService._inspect_one``) and its hole choice
(``services.inspection_service.select_hole``). The verdict re-evaluates the
moment "Expected Hole Count" or "Position Tolerance" changes, without a new
capture, so a tolerance can be dialled in against a known part. The deviation
exists only for a **calibrated** camera (it is measured from the calibration's
reference point, not from the image centre); on an uncalibrated camera the
pipeline skips the tolerance check entirely, and the page says so rather than
reporting a GOOD it never checked.

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

**Dataset & Training** appears on the ``yolo`` form only, because it is the
one strategy that cannot do anything until a trained ``.pt`` exists. It opens
:class:`~ui.detection.yolo_training_dialog.YoloTrainingDialog` — the bundled
``YoloLabel.exe`` for labelling, then a training run off the GUI thread — and
a run that produced weights loads them straight into "Model Weights" here, so
the model is one "Save & Apply" from being what the station runs. The
dialog's folder/model choices persist in ``app_config.json``'s
``yolo_training`` block.

``opencv``, ``dark_hole`` and ``template_matching`` are sweepable. For
``template_matching`` the ROI does more than bound the search: the diameter
gates it implies are divided by the template's own size to give the **scale
range**, so the sweep searches the sizes the drawn box could actually
contain rather than a ladder of guessed factors (``_scales_from_roi``), and
each trial searches a single scale so the winning row names it. ``yolo`` is
deliberately not sweepable — see ``_UNSWEEPABLE_REASON``, which is also what
its greyed-out button says. One detector instance is built per sweep and
reconfigured per trial, so template_matching re-reads its template file once
for the whole grid rather than once per combination.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
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

from core.calibration import CalibrationManager
from core.logging import get_logger
from core.utilities import ConfigManager
from core.utilities.enums import DetectorType, LogSource
from core.utilities.exceptions import ConfigurationError, VisionSystemError
from core.vision import (
    DarkHoleDetector,
    DetectionResult,
    Hole,
    HoleDetector,
    OpenCVHoleDetector,
    TemplateMatchingDetector,
    VisionEngine,
    draw_debug_overlay,
    draw_detection_overlay,
    normalize_image,
)
from core.vision.template_matching_detector import parse_scales, template_mean_side
from models.app_state import AppState
from services.camera_service import CameraService
from services.inspection_service import select_hole
from services.yolo_training_service import YoloTrainingService
from ui.detection.yolo_training_dialog import YoloTrainingDialog
from ui.widgets import RoiEditor

logger = get_logger(LogSource.UI)

#: app_config.json block the training dialog's folder/model choices persist in,
#: so an operator does not retype two paths every session.
_TRAINING_SETTINGS_KEY = "yolo_training"

# Strategies the auto sweep can grid-search, and the detector class it builds
# each trial candidate from directly (bypassing the shared VisionEngine, so a
# sweep never mutates the live/production configuration).
_SWEEP_DETECTORS: dict[str, type] = {
    "opencv": OpenCVHoleDetector,
    "dark_hole": DarkHoleDetector,
    "template_matching": TemplateMatchingDetector,
}

# ``yolo`` is deliberately absent, and the disabled button says why: a trained
# model's only sweepable knobs are its two post-processing thresholds, and the
# sweep evaluates every trial against a small ROI crop (see _crop_around_roi).
# A detector trained on full frames, run against a 150 px crop that Ultralytics
# then letterboxes to its own imgsz, answers a different question from the one
# the production cycle asks — so a sweep here would not be slow, it would be
# misleading. Tune a model with "Test on Camera" and its confidence readout.
_UNSWEEPABLE_REASON = {
    "yolo": (
        "Auto sweep is not available for 'yolo': a trained model has only its "
        "two thresholds to search, and the sweep runs each trial on a small ROI "
        "crop, which is not what the model saw in training — the result would "
        "mislead rather than tune. Use 'Test on Camera' and read the confidence."
    ),
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
    "template_matching": {
        "method": ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"],
        # Descending on purpose. A threshold no candidate clears simply
        # contributes no row, and every threshold a candidate *does* clear
        # scores it identically — so with the sort being stable, the first row
        # for a given method/scale is the **strictest threshold that still
        # found the hole**, which is the number an operator wants.
        "match_threshold": [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5],
        # Derived per run from the drawn ROI, not a fixed ladder — see
        # _scales_from_roi. Listed with an empty level set so the results
        # table still shows the winning scale, which is the whole point of
        # sweeping this strategy.
        "scales": [],
    },
}

#: How many scales _scales_from_roi spreads across the ROI-implied range.
_SWEEP_SCALE_STEPS = 9


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


def _scales_from_roi(base: dict) -> list[float]:
    """The scale range worth searching, from the ROI-derived diameter gates.

    A match's diameter is the template's own mean side times the scale it
    matched at, so the ROI that already bounds the hole's diameter bounds the
    scale directly: no ladder of guessed factors, and no combination spent on
    a scale that could not produce a hole the drawn box would accept. Spread
    geometrically, because scale is multiplicative — a fixed +0.1 step is
    coarse at 0.3x and needlessly fine at 3x.

    Raises:
        DetectionError: no template configured, or it cannot be read.
    """
    side = template_mean_side(str(base.get("template_path", "") or ""))
    min_diameter = float(base.get("min_hole_diameter_px", 0) or 0)
    max_diameter = float(base.get("max_hole_diameter_px", 0) or 0)
    low = max(0.05, min_diameter / side)
    high = max(low * 1.2, max_diameter / side)
    ratio = (high / low) ** (1.0 / (_SWEEP_SCALE_STEPS - 1))
    return sorted({round(low * ratio**step, 3) for step in range(_SWEEP_SCALE_STEPS)})


def _template_matching_param_grid(base: dict) -> list[dict]:
    """Every combination the template_matching sweep tries, seeded with *base*.

    Each trial searches exactly **one** scale, so the winning row names the
    scale that fitted rather than a set that happened to contain it; "Apply
    Best" then writes that single scale into the form's Scales field, which
    the operator can widen by hand around it.
    """
    levels = _SWEEP_LEVELS["template_matching"]
    combos: list[dict] = []
    for method in levels["method"]:
        for scale in _scales_from_roi(base):
            for threshold in levels["match_threshold"]:
                combos.append({
                    **base,
                    "method": method,
                    "scales": [scale],
                    "match_threshold": threshold,
                })
    return combos


_SWEEP_GRID_BUILDERS = {
    "opencv": _opencv_param_grid,
    "dark_hole": _dark_hole_param_grid,
    "template_matching": _template_matching_param_grid,
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


def _format_scales(value: object) -> str:
    """Render a ``scales`` value for the form's line edit.

    Trailing zeros are trimmed (``1.0`` -> ``1``, ``1.05`` -> ``1.05``) so a
    hand-typed list survives a load/save round trip looking the way it was
    typed rather than growing decimals each time.
    """
    return ", ".join(f"{scale:g}" for scale in parse_scales(value))


class DetectionPage(QWidget):
    """Per-camera algorithm parameters, strategy switching, live test."""

    def __init__(
        self,
        config_manager: ConfigManager,
        vision_engine: VisionEngine,
        camera_service: CameraService,
        app_state: AppState,
        calibration: CalibrationManager | None = None,
        training: YoloTrainingService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config_manager
        self._engine = vision_engine
        self._cameras = camera_service
        # Optional so a page built without one (tests) still works — the
        # verdict then reports the tolerance as unchecked, as for an
        # uncalibrated camera.
        self._calibration = calibration
        # Also optional: without it the yolo form's "Dataset & Training"
        # button is disabled and says so, but every parameter still edits.
        self._training = training

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
        self._tolerance.setToolTip(
            "Max distance (mm) the hole may sit from the calibrated reference "
            "point before the camera reports NG. 0 disables the check; it is "
            "also skipped for an uncalibrated camera."
        )
        self._normalize = QCheckBox("Normalize image before detection")
        self._normalize.setToolTip(
            "Contrast-stretches the frame to the full 0-255 range before the "
            "active strategy runs. Off by default: a strategy's own "
            "thresholds (e.g. detection/contrast) are tuned against the "
            "frame's current contrast, so turning this on may need them "
            "re-tuned."
        )
        common.addRow("Confidence Threshold", self._confidence)
        common.addRow("Expected Hole Count", self._expected)
        common.addRow("Position Tolerance", self._tolerance)
        # Re-judge the last Test against the new threshold, no new capture.
        self._expected.valueChanged.connect(self._refresh_verdict)
        self._tolerance.valueChanged.connect(self._refresh_verdict)
        common.addRow("", self._normalize)
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
            "template_matching": {
                "method": self._tm_method,
                "match_threshold": self._tm_threshold,
                "scales": self._tm_scales,
                "min_hole_diameter_px": self._tm_min_diameter,
                "max_hole_diameter_px": self._tm_max_diameter,
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
        self._verdict_label = QLabel("")
        self._verdict_label.setWordWrap(True)
        right.addWidget(self._verdict_label)
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
        self._tm_method.setToolTip(
            "TM_CCOEFF_NORMED subtracts the mean and is the one to use. "
            "TM_CCORR_NORMED does not, so on a bright low-contrast plate it "
            "scores near 1.0 almost everywhere"
        )
        self._tm_scales = QLineEdit()
        self._tm_scales.setPlaceholderText("1.0")
        self._tm_scales.setToolTip(
            "Comma-separated sizes the template is searched at, relative to "
            "how it was cropped — '0.9, 1.0, 1.1' tolerates about +/-10% of "
            "standoff or part-height variation. One entry is fastest; each "
            "extra one is another full pass over the frame. Run Auto Sweep "
            "with an ROI drawn round the hole to find the right value."
        )
        self._tm_max_matches = _spin(1, 50)
        self._tm_max_matches.setToolTip("Most holes this strategy will report per frame")
        self._tm_min_diameter = _spin(0, 4000)
        self._tm_max_diameter = _spin(0, 4000)
        size_hint = (
            "Gate on the matched size in px; 0 on either field disables it. "
            "Only meaningful with several scales in play — it is what stops a "
            "much-magnified match on a background feature counting as a hole."
        )
        self._tm_min_diameter.setToolTip(size_hint)
        self._tm_max_diameter.setToolTip(size_hint)
        form.addRow("Template Image", path_w)
        form.addRow("Match Threshold", self._tm_threshold)
        form.addRow("Method", self._tm_method)
        form.addRow("Scales", self._tm_scales)
        form.addRow("Max Matches", self._tm_max_matches)
        form.addRow("Min Hole Diameter (px)", self._tm_min_diameter)
        form.addRow("Max Hole Diameter (px)", self._tm_max_diameter)
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
        self._yolo_conf.setToolTip(
            "The model's own per-box gate. 'Confidence Threshold' above "
            "filters again afterwards, so the effective floor is whichever "
            "of the two is higher"
        )
        self._yolo_iou = _dspin(0.0, 1.0, 0.05)
        self._yolo_class = _spin(-1, 999)
        self._yolo_class.setSpecialValueText("any class")  # shown at -1
        self._yolo_class.setToolTip(
            "Which trained class counts as a hole. Set it to 'any class' (-1) "
            "for a single-class model — a model whose one class is not id 0 "
            "otherwise detects nothing at all"
        )
        self._yolo_device = QComboBox()
        self._yolo_device.setEditable(True)
        self._yolo_device.addItems(["", "cpu", "0", "cuda:0"])
        self._yolo_device.setToolTip(
            "Leave blank to let Ultralytics choose. This station has no CUDA "
            "unless one was fitted, in which case '0' selects the first GPU"
        )
        self._yolo_imgsz = _spin(0, 4096)
        self._yolo_imgsz.setSpecialValueText("model default")  # shown at 0
        self._yolo_imgsz.setToolTip(
            "Inference size in px. Worth setting only to match how the model "
            "was trained; anything else costs accuracy"
        )
        self._yolo_max_det = _spin(0, 1000)
        self._yolo_max_det.setSpecialValueText("model default")  # shown at 0
        self._yolo_min_diameter = _spin(0, 4000)
        self._yolo_max_diameter = _spin(0, 4000)
        size_hint = (
            "Gate on the box's mean side in px; 0 on either field disables it. "
            "A trained model usually needs no size gate — this is for the case "
            "where it also fires on a similar feature at a very different scale."
        )
        self._yolo_min_diameter.setToolTip(size_hint)
        self._yolo_max_diameter.setToolTip(size_hint)
        form.addRow("Model Weights", path_w)
        form.addRow("Confidence", self._yolo_conf)
        form.addRow("IoU Threshold", self._yolo_iou)
        form.addRow("Class ID", self._yolo_class)
        form.addRow("Device", self._yolo_device)
        form.addRow("Inference Size (px)", self._yolo_imgsz)
        form.addRow("Max Detections", self._yolo_max_det)
        form.addRow("Min Hole Diameter (px)", self._yolo_min_diameter)
        form.addRow("Max Hole Diameter (px)", self._yolo_max_diameter)

        # yolo is the one strategy with nothing to run until a model exists,
        # so the tooling that produces one is offered right where it is
        # selected rather than at a command prompt.
        self._train_btn = QPushButton("Dataset && Training…")
        self._train_btn.setToolTip(
            "Label captured frames and train a model, without leaving the app"
        )
        self._train_btn.clicked.connect(self._on_open_training)
        if self._training is None:
            self._train_btn.setEnabled(False)
            self._train_btn.setToolTip(
                "Training tooling was not wired into this page (it is optional, "
                "so a page built without it still edits parameters normally)"
            )
        form.addRow("", self._train_btn)
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
        self._normalize.setChecked(bool(common.get("normalize_image", False)))
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
        self._tm_scales.setText(_format_scales(template.get("scales")))
        self._tm_max_matches.setValue(int(template.get("max_matches", 10)))
        self._tm_min_diameter.setValue(int(template.get("min_hole_diameter_px", 0)))
        self._tm_max_diameter.setValue(int(template.get("max_hole_diameter_px", 0)))

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
        self._yolo_device.setCurrentText(str(yolo.get("device", "") or ""))
        self._yolo_imgsz.setValue(int(yolo.get("imgsz", 0) or 0))
        self._yolo_max_det.setValue(int(yolo.get("max_detections", 0) or 0))
        self._yolo_min_diameter.setValue(int(yolo.get("min_hole_diameter_px", 0)))
        self._yolo_max_diameter.setValue(int(yolo.get("max_hole_diameter_px", 0)))

    def _collect(self) -> dict:
        return {
            "active_detector": self._strategy.currentText(),
            "common": {
                "confidence_threshold": self._confidence.value(),
                "expected_hole_count": self._expected.value(),
                "position_tolerance_mm": self._tolerance.value(),
                "normalize_image": self._normalize.isChecked(),
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
                # Normalised here, not on the way out of the detector, so
                # detection.json always holds a clean list whatever was typed.
                "scales": parse_scales(self._tm_scales.text()),
                "max_matches": self._tm_max_matches.value(),
                "min_hole_diameter_px": self._tm_min_diameter.value(),
                "max_hole_diameter_px": self._tm_max_diameter.value(),
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
                "device": self._yolo_device.currentText().strip(),
                "imgsz": self._yolo_imgsz.value(),
                "max_detections": self._yolo_max_det.value(),
                "min_hole_diameter_px": self._yolo_min_diameter.value(),
                "max_hole_diameter_px": self._yolo_max_diameter.value(),
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
        cfg = self._cameras.effective_config(camera_index) or {}
        strobe = bool(cfg.get("led_strobe", False))
        if strobe:
            self._cameras.light_on(camera_index)
        try:
            frame = self._cameras.test_capture(camera_index)
            self._engine.apply_camera_config(camera_index, self._collect())  # test what's on screen
            result = self._engine.detect(frame, camera_index)
            if self._calibration is not None:
                select_hole(self._calibration, camera_index, result)  # judge what a cycle would
        except VisionSystemError as exc:
            # Turn the light off before the blocking warning dialog, not
            # after — it must not sit lit for however long the operator
            # takes to dismiss it (same reasoning as the Camera page).
            if strobe:
                self._cameras.light_off(camera_index)
            QMessageBox.warning(self, "Test", str(exc))
            return
        if strobe:
            self._cameras.light_off(camera_index)
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
        self._refresh_verdict()

    def _on_open_training(self) -> None:
        """Open the labelling/training dialog for the selected camera.

        Modeless: a run is minutes to hours, and the station has to stay
        usable — an inspection cycle included — for all of it. The dialog's
        folder and model choices are persisted to app_config.json when it
        closes, and a run that produced weights offers them straight to the
        Model Weights field, which is the point of hosting this here.
        """
        if self._training is None:
            return
        settings = self._config.get_value("app_config", _TRAINING_SETTINGS_KEY) or {}
        dialog = YoloTrainingDialog(self._training, dict(settings), self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        dialog.finished.connect(lambda _result: self._on_training_closed(dialog))
        dialog.show()

    def _on_training_closed(self, dialog: YoloTrainingDialog) -> None:
        """Persist the dialog's choices, and adopt the weights it produced."""
        try:
            document = self._config.load("app_config")
            # Merged, never replaced: the block also holds keys the dialog
            # has no field for — 'labeling_tool', which only main.py reads —
            # and assigning over it would delete a relocated station's path
            # the first time anyone opened this dialog.
            block = dict(document.get(_TRAINING_SETTINGS_KEY) or {})
            block.update(dialog.settings())
            document[_TRAINING_SETTINGS_KEY] = block
            self._config.save("app_config", document)
        except VisionSystemError as exc:
            # Never raised back at the operator: failing to remember a folder
            # path must not look like the training itself failed.
            logger.warning("Could not persist training settings: %s", exc)

        weights = dialog.trained_weights
        if weights and dialog.result() == QDialog.DialogCode.Accepted:
            self._strategy.setCurrentText(DetectorType.YOLO.value)
            self._yolo_path.setText(weights)
            QMessageBox.information(
                self, "Trained Model",
                "The trained weights are in the Model Weights field.\n\n"
                "Press 'Save && Apply' to make this camera use them, then "
                "'Test on Camera' to check it before running production.",
            )
        dialog.deleteLater()

    def _refresh_verdict(self) -> None:
        """Show the GOOD/NG a real cycle would reach on the last Test frame.

        Mirrors ``InspectionService._inspect_one`` rule for rule, reading the
        *on-screen* expected count and tolerance (Test applies the form to the
        engine, so that is what the last detection ran under anyway).
        """
        if self._last_result is None or self._last_frame is None:
            self._set_verdict("", "")
            return
        result = self._last_result
        camera_index = self._last_camera_index
        expected = self._expected.value()
        best = result.best
        if best is None or len(result.holes) < expected:
            self._set_verdict(
                "NG",
                f"NG — {len(result.holes)} of {expected} expected hole(s) found; "
                f"position tolerance not reached.",
            )
            return

        calibrated = self._calibration is not None and self._calibration.has(camera_index)
        if not calibrated:
            self._set_verdict(
                "GOOD",
                f"GOOD — position tolerance NOT checked: camera {camera_index} is "
                f"not calibrated, so there is no reference point to measure "
                f"deviation from (a real cycle skips the check too). Calibrate "
                f"it on the Calibration page to enable the tolerance.",
            )
            return

        height, width = self._last_frame.shape[:2]
        x_mm, y_mm, deviation = self._calibration.evaluate(
            camera_index, best.x_px, best.y_px, width, height
        )
        tolerance = self._tolerance.value()
        position = f"position X {x_mm:+.2f} / Y {y_mm:+.2f} mm"
        if tolerance <= 0:
            self._set_verdict(
                "GOOD",
                f"GOOD — deviation {deviation:.2f} mm from reference; tolerance "
                f"check disabled (0 mm). {position}",
            )
        elif deviation > tolerance:
            self._set_verdict(
                "NG",
                f"NG — deviation {deviation:.2f} mm exceeds the {tolerance:.2f} mm "
                f"tolerance by {deviation - tolerance:.2f} mm. {position}",
            )
        else:
            self._set_verdict(
                "GOOD",
                f"GOOD — deviation {deviation:.2f} mm within the {tolerance:.2f} mm "
                f"tolerance ({tolerance - deviation:.2f} mm margin). {position}",
            )

    def _set_verdict(self, result: str, text: str) -> None:
        """``result`` drives the theme's ``QLabel[result=...]`` colour."""
        label = self._verdict_label
        label.style().unpolish(label)
        label.setProperty("result", result)
        label.style().polish(label)
        label.setText(text)

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
        self._set_verdict("", "")
        self._sweep_rows = []
        self._sweep_table.setRowCount(0)
        self._sweep_status.setText("Run 'Test on Camera' first.")

    # ---------------------------------------------------------- auto sweep
    def _update_sweep_availability(self, strategy: str) -> None:
        supported = strategy in _SWEEP_GRID_BUILDERS
        self._sweep_btn.setEnabled(supported)
        self._sweep_btn.setToolTip("" if supported else self._unsweepable_reason(strategy))

    @staticmethod
    def _unsweepable_reason(strategy: str) -> str:
        """Why this strategy's sweep button is greyed out — a bare "not
        available" leaves an operator wondering whether it is a bug."""
        return _UNSWEEPABLE_REASON.get(
            strategy, f"Auto sweep is not available for the '{strategy}' strategy"
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
            QMessageBox.information(self, "Auto Sweep", self._unsweepable_reason(strategy))
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
        try:
            # template_matching derives its scale range here, which needs the
            # template itself — an unset or unreadable one has to be reported
            # as the configuration problem it is, not as "found nothing".
            combos = grid_builder(base_params)
            detector = _SWEEP_DETECTORS[strategy](base_params)
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Auto Sweep", str(exc))
            return
        cropped, offset_x, offset_y = self._crop_around_roi(self._last_frame, roi)
        if self._normalize.isChecked():  # match what the live/tested detector actually sees
            cropped = normalize_image(cropped)

        found: list[tuple[dict, Hole]] = []
        tried = self._run_sweep_grid(combos, detector, cropped, roi, offset_x, offset_y, found)

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
        detector: HoleDetector,
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

        *detector* is **one** instance, reconfigured per trial rather than
        rebuilt: every combination carries the full parameter set, so
        ``configure`` fully determines each trial, and a strategy that does
        real work on construction (template_matching re-reads and rescales
        its template file) then pays that cost once for the whole grid
        instead of once per combination.
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
                    detector.configure(params)
                    result = detector.detect(cropped)
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
            values = [self._format_cell(params.get(key)) for key in columns]
            values += [f"{hole.confidence:.2f}", f"{hole.diameter_px:.1f}", f"{hole.circularity:.2f}"]
            for col, value in enumerate(values):
                self._sweep_table.setItem(row_index, col, QTableWidgetItem(value))

    @staticmethod
    def _format_cell(value: object) -> str:
        """One results-table cell. A list parameter (template_matching's
        ``scales``) renders as its bare values rather than ``[1.05]``."""
        if isinstance(value, (list, tuple)):
            return ", ".join(f"{item:g}" if isinstance(item, float) else str(item)
                             for item in value)
        return str(value)

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
        elif isinstance(widget, QLineEdit):
            # Only template_matching's "scales" reaches here, and it arrives
            # as the single-entry list each trial searched.
            widget.setText(
                _format_scales(value) if isinstance(value, (list, tuple)) else str(value)
            )
        else:
            raise TypeError(f"Unsupported widget type for sweep apply: {type(widget)}")

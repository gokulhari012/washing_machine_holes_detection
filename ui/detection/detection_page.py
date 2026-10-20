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

**template_opencv** (template matching, then OpenCV inside each match) has
only two parameters of its own; its two stages run the camera's
``template_matching`` and ``opencv`` blocks. Selecting it therefore shows
three forms — its own, then those two retitled as Stage 1 / Stage 2 — rather
than one, which is why the forms are shown by an explicit strategy -> forms
map (``_FORMS_SHOWN``) instead of a QStackedWidget indexed by the combo's
position. Its Auto Sweep tunes one stage at a time, chosen from a "Stage"
combo shown beside "Run Auto Sweep" only for this strategy (``_SWEEP_STAGES``):
Stage 1 runs the ``template_matching`` grid (ROI around the matched feature),
Stage 2 the ``opencv`` grid — but each Stage 2 trial runs only inside Stage
1's match, found once with the form's current values, because that is the
crop a real cycle hands OpenCV. "Apply Best" writes into that stage's form.

**blob_analysis** is the lab's center-hole detector (threshold, open/close,
keep the largest blob that is round enough) and **template_blob_analysis**
is the same two-stage arrangement as template_opencv with blob analysis as
Stage 2 — it shows its own small form over the ``template_matching`` and
``blob_analysis`` forms, and sweeps one stage at a time exactly as above.
Both stages' trial images follow the same rules, so the two two-stage
strategies share one map (``_TWO_STAGE_DETECTORS``) for everything that has
to know which detector a strategy's Stage 1 regions come from. The
blob_analysis sweep derives its ``min_area_px`` from the ROI the same way the
diameter gates are derived, rather than searching it.

**caliper_based_circle_fitting** (radial calipers find the rim, a robust
circle fit gives the centre) and **template_caliper_based_circle_fitting**
(the same as Stage 2 of a template match) follow exactly the same pattern as
blob analysis and its template version. Its sweep searches only what
*measures* the rim — the seed threshold, blur, search band, edge polarity,
selection and strength — and leaves the three acceptance criteria (inlier
tolerance, min inlier ratio, max RMS) at the form's values: they define what
a good fit is, and since the hole's confidence is its inlier ratio, a sweep
allowed to loosen the tolerance would always pick the loosest one.

The **Judgement** box also carries two per-camera switches the engine and the
pipeline act on: **Downscale** (detect on a frame shrunk by the factor; see
``VisionEngine``'s module docstring — results and every pixel parameter stay
in full-resolution pixels, so nothing here needs re-tuning) and **On NG ->
Required another run** (re-capture and re-check an NG up to N more times; see
``InspectionService``). "Test on Camera" honours downscale, because it runs
the engine; it does *not* re-run an NG — it is one capture by design — and
says in the verdict what a real cycle would do instead.

``opencv``, ``dark_hole``, ``blob_analysis``, ``caliper_based_circle_fitting`` and
``template_matching`` are sweepable. For
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

import math
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
    BlobAnalysisDetector,
    CaliperCircleDetector,
    DarkHoleDetector,
    DetectionResult,
    Hole,
    HoleDetector,
    OpenCVHoleDetector,
    TemplateBlobAnalysisDetector,
    TemplateCaliperCircleDetector,
    TemplateMatchingDetector,
    TemplateOpenCVDetector,
    TemplateRegionDetector,
    VisionEngine,
    draw_debug_overlay,
    draw_detection_overlay,
    normalize_image,
)
from core.vision.template_matching_detector import parse_scales, template_mean_side
from core.vision.vision_engine import (
    DEFAULT_DOWNSCALE_FACTOR,
    MAX_DOWNSCALE_FACTOR,
    MAX_NG_RERUNS,
)
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
    "blob_analysis": BlobAnalysisDetector,
    "caliper_based_circle_fitting": CaliperCircleDetector,
}

#: The "template first, then the hole inside it" strategies, and the detector
#: each one's Stage 1 regions come from — Stage 2's sweep trials run inside
#: those regions (``DetectionPage._template_region_views``), and "no hole"
#: on one of these may mean the template itself was not found.
_TWO_STAGE_DETECTORS: dict[str, type[TemplateRegionDetector]] = {
    DetectorType.TEMPLATE_OPENCV.value: TemplateOpenCVDetector,
    DetectorType.TEMPLATE_BLOB_ANALYSIS.value: TemplateBlobAnalysisDetector,
    DetectorType.TEMPLATE_CALIPER_BASED_CIRCLE_FITTING.value: TemplateCaliperCircleDetector,
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

#: Stage 1 of every two-stage strategy — the same template_matching block.
_TEMPLATE_SWEEP_STAGE = (
    DetectorType.TEMPLATE_MATCHING.value,
    "Stage 1 — Template Matching",
    "Stage 1: draw the ROI around the feature the template should "
    "match (the whole matched area, not just the hole) — the sweep "
    "scores where the template lands and sizes its scale range from "
    "the box.",
)


def _hole_sweep_stage(strategy: DetectorType, label: str) -> tuple[str, str, str]:
    """Stage 2 of a two-stage strategy: *strategy*'s own grid, in the match."""
    return (
        strategy.value,
        f"Stage 2 — {label}",
        f"Stage 2: draw the ROI around the hole. Stage 1 runs once with the "
        f"form's current values, and every {label} trial searches only inside "
        f"its match — so Stage 1 must already find the template.",
    )


#: Multi-stage strategies sweep **one stage at a time**, picked from the
#: "Stage" combo beside "Run Auto Sweep": ``(stage strategy, combo label,
#: what the ROI must mark)``. A joint grid would be the product of both
#: stages' grids — far too big — and each stage is its own strategy's block
#: anyway, so it reuses that strategy's grid, results table and "Apply Best".
#: Each two-stage strategy's Stage 2 trials run inside Stage 1's matches,
#: exactly as a real cycle does (see ``DetectionPage._template_region_views``).
_SWEEP_STAGES: dict[str, tuple[tuple[str, str, str], ...]] = {
    DetectorType.TEMPLATE_OPENCV.value: (
        _TEMPLATE_SWEEP_STAGE,
        _hole_sweep_stage(DetectorType.OPENCV, "OpenCV"),
    ),
    DetectorType.TEMPLATE_BLOB_ANALYSIS.value: (
        _TEMPLATE_SWEEP_STAGE,
        _hole_sweep_stage(DetectorType.BLOB_ANALYSIS, "Blob Analysis"),
    ),
    DetectorType.TEMPLATE_CALIPER_BASED_CIRCLE_FITTING.value: (
        _TEMPLATE_SWEEP_STAGE,
        _hole_sweep_stage(DetectorType.CALIPER_BASED_CIRCLE_FITTING, "Caliper Circle Fitting"),
    ),
}

#: Which parameter forms each strategy shows, top to bottom; a strategy not
#: listed shows just its own. A two-stage strategy runs the camera's own
#: template_matching block and its hole stage's block as its two stages, so
#: it shows (and Save writes) those two forms under its own small one.
_FORMS_SHOWN: dict[str, tuple[str, ...]] = {
    DetectorType.TEMPLATE_OPENCV.value: (
        DetectorType.TEMPLATE_OPENCV.value,
        DetectorType.TEMPLATE_MATCHING.value,
        DetectorType.OPENCV.value,
    ),
    DetectorType.TEMPLATE_BLOB_ANALYSIS.value: (
        DetectorType.TEMPLATE_BLOB_ANALYSIS.value,
        DetectorType.TEMPLATE_MATCHING.value,
        DetectorType.BLOB_ANALYSIS.value,
    ),
    DetectorType.TEMPLATE_CALIPER_BASED_CIRCLE_FITTING.value: (
        DetectorType.TEMPLATE_CALIPER_BASED_CIRCLE_FITTING.value,
        DetectorType.TEMPLATE_MATCHING.value,
        DetectorType.CALIPER_BASED_CIRCLE_FITTING.value,
    ),
}

#: Titles the stage forms carry while a multi-form strategy is selected, so
#: it is obvious which stage each block of parameters drives.
_TEMPLATE_STAGE_TITLE = "Stage 1 — Template Matching (finds the region)"
_STAGE_TITLES: dict[str, dict[str, str]] = {
    DetectorType.TEMPLATE_OPENCV.value: {
        DetectorType.TEMPLATE_MATCHING.value: _TEMPLATE_STAGE_TITLE,
        DetectorType.OPENCV.value: "Stage 2 — OpenCV (finds the hole inside the region)",
    },
    DetectorType.TEMPLATE_BLOB_ANALYSIS.value: {
        DetectorType.TEMPLATE_MATCHING.value: _TEMPLATE_STAGE_TITLE,
        DetectorType.BLOB_ANALYSIS.value:
            "Stage 2 — Blob Analysis (finds the hole inside the region)",
    },
    DetectorType.TEMPLATE_CALIPER_BASED_CIRCLE_FITTING.value: {
        DetectorType.TEMPLATE_MATCHING.value: _TEMPLATE_STAGE_TITLE,
        DetectorType.CALIPER_BASED_CIRCLE_FITTING.value:
            "Stage 2 — Caliper Circle Fitting (measures the hole inside the region)",
    },
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
    "blob_analysis": {
        "threshold": [20, 30, 40, 50, 60, 75, 90, 110, 130, 150, 175, 200],
        "blur_kernel_size": [1, 3, 5, 7, 9, 11],
        # 1 = no clean-up at all (open/close with a 1x1 kernel is a no-op)
        "morphology_kernel_size": [1, 3, 5, 7, 9],
        # Descending, for the same reason as template_matching's
        # match_threshold: the hole's confidence *is* its circularity, so
        # every floor it clears scores it identically and the stable sort
        # keeps the strictest floor that still found it.
        "min_circularity": [0.9, 0.85, 0.8, 0.7, 0.6, 0.5],
    },
    # Only what *measures* the rim. inlier_threshold_px, min_inlier_ratio and
    # max_rms_px stay at the form's values: they define what a good fit is,
    # and the hole's confidence is its inlier ratio — a sweep free to widen
    # the tolerance would always pick the widest. num_calipers and
    # caliper_width_px stay too: more calipers only sample the rim more finely.
    "caliper_based_circle_fitting": {
        "coarse_threshold": [0, 40, 70, 100, 130, 160],  # 0 = Otsu, per image
        "blur_kernel_size": [1, 3, 5, 9],
        "search_band": [0.3, 0.6],
        "edge_polarity": ["dark_to_light", "either"],
        "edge_selection": ["strongest", "first"],
        # Descending, as for blob_analysis's min_circularity: every floor a rim
        # clears with the same inliers scores the same, and the stable sort
        # keeps the strictest one.
        "min_edge_strength": [40, 20, 10, 5],
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


def _blob_analysis_param_grid(base: dict) -> list[dict]:
    """Every combination the blob_analysis sweep tries, seeded with *base*.

    ``min_area_px`` is not searched: it is set to the area of a circle at the
    ROI-derived minimum diameter, so it gates exactly where that diameter gate
    already does. Left at the form's value instead, a Min Area typed for a
    bigger hole would reject every trial, and the sweep would report "found
    nothing" with no hint why.
    """
    levels = _SWEEP_LEVELS["blob_analysis"]
    min_diameter = float(base.get("min_hole_diameter_px", 0) or 0)
    seeded = {**base, "min_area_px": int(math.pi * (min_diameter / 2.0) ** 2)}
    combos: list[dict] = []
    for threshold in levels["threshold"]:
        for blur in levels["blur_kernel_size"]:
            for kernel in levels["morphology_kernel_size"]:
                for circularity in levels["min_circularity"]:
                    combos.append({
                        **seeded,
                        "threshold": threshold,
                        "blur_kernel_size": blur,
                        "morphology_kernel_size": kernel,
                        "min_circularity": circularity,
                    })
    return combos


def _caliper_param_grid(base: dict) -> list[dict]:
    """Every combination the caliper sweep tries, seeded with *base* — whose
    diameter gates come from the ROI and also bound the seed blob, so a big
    dark background region cannot be the seed of any trial."""
    levels = _SWEEP_LEVELS["caliper_based_circle_fitting"]
    combos: list[dict] = []
    for seed_threshold in levels["coarse_threshold"]:
        for blur in levels["blur_kernel_size"]:
            for band in levels["search_band"]:
                for polarity in levels["edge_polarity"]:
                    for selection in levels["edge_selection"]:
                        for strength in levels["min_edge_strength"]:
                            combos.append({
                                **base,
                                "coarse_threshold": seed_threshold,
                                "blur_kernel_size": blur,
                                "search_band": band,
                                "edge_polarity": polarity,
                                "edge_selection": selection,
                                "min_edge_strength": strength,
                            })
    return combos


_SWEEP_GRID_BUILDERS = {
    "opencv": _opencv_param_grid,
    "dark_hole": _dark_hole_param_grid,
    "template_matching": _template_matching_param_grid,
    "blob_analysis": _blob_analysis_param_grid,
    "caliper_based_circle_fitting": _caliper_param_grid,
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


def _inline(*widgets: QWidget) -> QWidget:
    """Several widgets side by side in one form row, left-aligned."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch()
    return row


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
        # Downscale: the engine shrinks the frame before the strategy runs and
        # maps every result back, converting each pixel parameter as it goes,
        # so switching it on needs nothing on this page re-tuned.
        self._downscale = QCheckBox("Enabled")
        self._downscale.setToolTip(
            "Detect on a smaller copy of the frame — faster on a high-resolution "
            "camera. Positions, diameters and every pixel setting on this page "
            "stay in full-resolution pixels, so nothing needs re-tuning; only "
            "the precision drops, by about the factor in pixels."
        )
        self._downscale_factor = _dspin(1.0, MAX_DOWNSCALE_FACTOR, 0.5, 1)
        self._downscale_factor.setPrefix("÷ ")
        self._downscale_factor.setToolTip(
            "Each side of the frame is divided by this before detection: 2 = half "
            "the width and half the height (a quarter of the pixels)."
        )
        self._downscale_factor.setEnabled(False)  # follows the checkbox
        self._downscale.toggled.connect(self._downscale_factor.setEnabled)

        # "Required another run": the pipeline re-captures and re-checks an NG.
        self._rerun = QCheckBox("Required another run")
        self._rerun.setToolTip(
            "When this camera's result is NG, take a fresh picture and check "
            "again, up to the number of times set here, before reporting NG. "
            "The first GOOD wins; a failed re-capture keeps the NG. Each run is "
            "a full capture, so it lengthens the cycle the PLC waits on."
        )
        self._rerun_count = _spin(1, MAX_NG_RERUNS)
        self._rerun_count.setSuffix(" time(s)")
        self._rerun_count.setToolTip("How many more times an NG is re-captured and re-checked")
        self._rerun_count.setEnabled(False)  # follows the checkbox
        self._rerun.toggled.connect(self._rerun_count.setEnabled)

        common.addRow("Confidence Threshold", self._confidence)
        common.addRow("Expected Hole Count", self._expected)
        common.addRow("Position Tolerance", self._tolerance)
        # Re-judge the last Test against the new threshold, no new capture.
        self._expected.valueChanged.connect(self._refresh_verdict)
        self._tolerance.valueChanged.connect(self._refresh_verdict)
        self._rerun.toggled.connect(self._refresh_verdict)
        self._rerun_count.valueChanged.connect(self._refresh_verdict)
        common.addRow("", self._normalize)
        common.addRow("Downscale", _inline(self._downscale, self._downscale_factor))
        common.addRow("On NG", _inline(self._rerun, self._rerun_count))
        left.addWidget(common_box)

        strategy_row = QHBoxLayout()
        strategy_row.addWidget(QLabel("Active Detector"))
        self._strategy = QComboBox()
        self._strategy.addItems([d.value for d in DetectorType])
        for strategy, tooltip in (
            (
                DetectorType.TEMPLATE_OPENCV,
                "Template matching and OpenCV detection: find the template "
                "first, then the hole with OpenCV inside it",
            ),
            (
                DetectorType.BLOB_ANALYSIS,
                "Blob analysis: threshold the frame and take the largest dark "
                "blob that is round enough as the hole",
            ),
            (
                DetectorType.TEMPLATE_BLOB_ANALYSIS,
                "Template matching and blob analysis: find the template first, "
                "then the largest round dark blob inside it",
            ),
            (
                DetectorType.CALIPER_BASED_CIRCLE_FITTING,
                "Caliper-based circle fitting: radial calipers find the rim "
                "edge, a robust circle fit gives the centre — steady against "
                "a partly hidden rim, burrs and reflections",
            ),
            (
                DetectorType.TEMPLATE_CALIPER_BASED_CIRCLE_FITTING,
                "Template matching and caliper circle fitting: find the template "
                "first, then measure the hole's rim with calipers inside it",
            ),
        ):
            self._strategy.setItemData(
                self._strategy.findText(strategy.value), tooltip, Qt.ItemDataRole.ToolTipRole
            )
        strategy_row.addWidget(self._strategy, stretch=1)
        left.addLayout(strategy_row)

        # One form per strategy; _show_forms decides which are visible.
        self._forms: dict[str, QGroupBox] = {
            DetectorType.OPENCV.value: self._build_opencv_form(),
            DetectorType.TEMPLATE_MATCHING.value: self._build_template_form(),
            DetectorType.YOLO.value: self._build_yolo_form(),
            DetectorType.DARK_HOLE.value: self._build_dark_hole_form(),
            DetectorType.TEMPLATE_OPENCV.value: self._build_template_opencv_form(),
            DetectorType.BLOB_ANALYSIS.value: self._build_blob_form(),
            DetectorType.TEMPLATE_BLOB_ANALYSIS.value: self._build_template_blob_form(),
            DetectorType.CALIPER_BASED_CIRCLE_FITTING.value: self._build_caliper_form(),
            DetectorType.TEMPLATE_CALIPER_BASED_CIRCLE_FITTING.value:
                self._build_template_caliper_form(),
        }
        self._form_titles = {name: form.title() for name, form in self._forms.items()}
        # Each combined form sits above its two stage forms (template matching
        # first, then the hole stage) — visible ones keep this order.
        for name in (
            DetectorType.TEMPLATE_OPENCV.value,
            DetectorType.TEMPLATE_BLOB_ANALYSIS.value,
            DetectorType.TEMPLATE_CALIPER_BASED_CIRCLE_FITTING.value,
            DetectorType.TEMPLATE_MATCHING.value,
            DetectorType.OPENCV.value,
            DetectorType.BLOB_ANALYSIS.value,
            DetectorType.CALIPER_BASED_CIRCLE_FITTING.value,
            DetectorType.YOLO.value,
            DetectorType.DARK_HOLE.value,
        ):
            left.addWidget(self._forms[name])
        self._strategy.currentTextChanged.connect(self._show_forms)
        self._strategy.currentTextChanged.connect(self._update_sweep_availability)

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
            "blob_analysis": {
                "threshold": self._ba_threshold,
                "blur_kernel_size": self._ba_blur,
                "morphology_kernel_size": self._ba_morph,
                "min_area_px": self._ba_min_area,
                "min_circularity": self._ba_circularity,
                "min_hole_diameter_px": self._ba_min_diameter,
                "max_hole_diameter_px": self._ba_max_diameter,
            },
            "caliper_based_circle_fitting": {
                "coarse_threshold": self._cf_seed_threshold,
                "blur_kernel_size": self._cf_blur,
                "search_band": self._cf_band,
                "edge_polarity": self._cf_polarity,
                "edge_selection": self._cf_selection,
                "min_edge_strength": self._cf_strength,
                "min_hole_diameter_px": self._cf_min_diameter,
                "max_hole_diameter_px": self._cf_max_diameter,
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
        # Which strategy's block _sweep_rows hold — a stage of template_opencv
        # is not the Active Detector, and "Apply Best" must hit that stage's form.
        self._sweep_rows_strategy: str | None = None
        self._sweep_cancel_requested = False

        self._load()
        # _load only emits currentTextChanged if the strategy actually changed.
        self._show_forms(self._strategy.currentText())
        self._update_sweep_availability(self._strategy.currentText())

    # ------------------------------------------------------- strategy forms
    def _show_forms(self, strategy: str) -> None:
        """Show the parameter forms *strategy* runs (``_FORMS_SHOWN``), and
        title the stage forms of a multi-stage strategy as its stages."""
        shown = _FORMS_SHOWN.get(strategy, (strategy,))
        stage_titles = _STAGE_TITLES.get(strategy, {})
        for name, form in self._forms.items():
            form.setTitle(stage_titles.get(name, self._form_titles[name]))
            form.setVisible(name in shown)

    def _visible_forms(self) -> list[str]:
        """Names of the forms currently shown — what a test asserts on, since
        ``isVisible`` is False for every widget of a page never shown."""
        return [name for name, form in self._forms.items() if not form.isHidden()]

    def _build_template_opencv_form(self) -> QGroupBox:
        box, self._to_regions, self._to_padding = self._build_regions_form(
            "Template Matching + OpenCV Parameters",
            "Stage 1 finds the template in the frame; stage 2 looks for the hole "
            "with OpenCV only inside the matched area — a hole anywhere else is "
            "ignored. Crop the template around the feature that holds the hole, "
            "not around the hole alone. Both stages use the forms below, which "
            "are the same parameters the 'template_matching' and 'opencv' "
            "strategies use; Stage 1's diameter gate applies to the matched area.",
        )
        return box

    def _build_template_blob_form(self) -> QGroupBox:
        box, self._tb_regions, self._tb_padding = self._build_regions_form(
            "Template Matching + Blob Analysis Parameters",
            "Stage 1 finds the template in the frame; stage 2 runs blob analysis "
            "only inside the matched area and takes the largest round dark blob "
            "there as the hole — a blob anywhere else is ignored, however big. "
            "Each matched region gives at most one hole. Crop the template "
            "around the feature that holds the hole, not around the hole alone. "
            "Both stages use the forms below, which are the same parameters the "
            "'template_matching' and 'blob_analysis' strategies use; Stage 1's "
            "diameter gate applies to the matched area.",
        )
        return box

    @staticmethod
    def _build_regions_form(title: str, hint_text: str) -> tuple[QGroupBox, QSpinBox, QSpinBox]:
        """A two-stage strategy's own block: how many template matches are
        searched, and how much each one is padded. Returns the box and the
        two spin boxes, which each strategy keeps under its own names."""
        box = QGroupBox(title)
        form = QFormLayout(box)
        hint = QLabel(hint_text)
        hint.setWordWrap(True)
        hint.setProperty("class", "dim")
        regions = _spin(1, 10)
        regions.setToolTip(
            "How many of the best template matches are searched for a hole — "
            "1 searches only the best one. Raise it only if the part really "
            "carries the feature more than once."
        )
        padding = _spin(0, 4000)
        padding.setSuffix(" px")
        padding.setToolTip(
            "Grows each matched area by this much on every side before the hole "
            "search — for a hole at, or just past, the template's edge."
        )
        form.addRow(hint)
        form.addRow("Regions Searched", regions)
        form.addRow("Region Padding", padding)
        return box, regions, padding

    def _build_blob_form(self) -> QGroupBox:
        box = QGroupBox("Blob Analysis Parameters")
        form = QFormLayout(box)
        hint = QLabel(
            "Every pixel darker than the threshold becomes a blob; the largest "
            "blob that passes the gates below is the hole (one hole per frame). "
            "Its confidence is its circularity. The Debug view shows the "
            "cleaned mask — every blob the gates judge."
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "dim")
        self._ba_threshold = _spin(0, 255)
        self._ba_threshold.setToolTip(
            "Dark level: pixels darker than this grey level are blob pixels. "
            "Raise it if the hole breaks up, lower it if the background joins in."
        )
        self._ba_blur = _spin(1, 31)
        self._ba_blur.setToolTip("Gaussian blur before the threshold; even values round up to odd")
        self._ba_morph = _spin(1, 31)
        self._ba_morph.setToolTip(
            "Square kernel for the open-then-close clean-up of the mask "
            "(removes specks, fills pin-holes). 1 = no clean-up; even values "
            "round up to odd."
        )
        self._ba_min_area = _spin(0, 50_000_000)
        self._ba_min_area.setSuffix(" px²")
        self._ba_min_area.setToolTip("Smallest blob area that counts, in pixels of the camera image")
        self._ba_circularity = _dspin(0.0, 1.0, 0.01)
        self._ba_circularity.setToolTip(
            "4πA/P² — 1.0 is a perfect circle. Blobs below this are not holes. "
            "'Confidence Threshold' above filters the same number again."
        )
        self._ba_min_diameter = _spin(0, 4000)
        self._ba_max_diameter = _spin(0, 4000)
        size_hint = (
            "Gate on the fitted diameter in px; 0 disables it. Auto Sweep sets "
            "both from the ROI you draw."
        )
        self._ba_min_diameter.setToolTip(size_hint)
        self._ba_max_diameter.setToolTip(size_hint)
        form.addRow(hint)
        form.addRow("Threshold (dark level)", self._ba_threshold)
        form.addRow("Blur Kernel", self._ba_blur)
        form.addRow("Morph Kernel", self._ba_morph)
        form.addRow("Min Area", self._ba_min_area)
        form.addRow("Min Circularity", self._ba_circularity)
        form.addRow("Min Hole Diameter (px)", self._ba_min_diameter)
        form.addRow("Max Hole Diameter (px)", self._ba_max_diameter)
        return box

    def _build_template_caliper_form(self) -> QGroupBox:
        box, self._tc_regions, self._tc_padding = self._build_regions_form(
            "Template Matching + Caliper Circle Fitting Parameters",
            "Stage 1 finds the template in the frame; stage 2 aims calipers at "
            "the darkest round blob inside the matched area only and fits a "
            "circle to the rim edges they find — a hole anywhere else is "
            "ignored. Each matched region gives at most one hole. Pad the "
            "region if the bore's rim runs past the template's edge. Both "
            "stages use the forms below, which are the same parameters the "
            "'template_matching' and 'caliper_based_circle_fitting' strategies "
            "use; Stage 1's diameter gate applies to the matched area.",
        )
        return box

    def _build_caliper_form(self) -> QGroupBox:
        box = QGroupBox("Caliper Circle Fitting Parameters")
        form = QFormLayout(box)
        hint = QLabel(
            "A dark blob gives a rough centre; calipers cast outward from it "
            "find the rim edge, and a robust circle fit to those edge points "
            "gives the hole (one per frame), ignoring points a burr or a "
            "reflection put off the circle. Confidence is the inlier ratio. "
            "Debug view: the seed mask, the search band, and the edge points — "
            "filled dots for inliers, rings for rejected outliers."
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "dim")
        self._cf_seed_threshold = _spin(0, 255)
        self._cf_seed_threshold.setSpecialValueText("auto (Otsu)")  # shown at 0
        self._cf_seed_threshold.setToolTip(
            "Dark level for the rough seed blob that aims the calipers. It only "
            "has to find the hole roughly — the rim itself is measured by the "
            "calipers. 0 picks a threshold per image (Otsu)."
        )
        self._cf_blur = _spin(1, 31)
        self._cf_blur.setToolTip("Gaussian blur before seeding and edge search; even rounds up to odd")
        self._cf_calipers = _spin(8, 720)
        self._cf_calipers.setToolTip("How many calipers are cast, evenly spaced round the hole")
        self._cf_width = _spin(1, 51)
        self._cf_width.setSuffix(" px")
        self._cf_width.setToolTip(
            "Each caliper averages this many parallel samples across its width "
            "— wider is steadier against noise and scratches, narrower follows "
            "a tight rim more closely"
        )
        self._cf_band = _dspin(0.05, 0.95, 0.05)
        self._cf_band.setToolTip(
            "How far each caliper reaches either side of the seed radius, as a "
            "fraction of it: 0.5 searches from half to one and a half seed radii"
        )
        self._cf_polarity = QComboBox()
        self._cf_polarity.addItems(["dark_to_light", "either"])
        self._cf_polarity.setToolTip(
            "Which edges count, going outward: 'dark_to_light' is a dark bore "
            "on brighter metal; 'either' also accepts bright-to-dark"
        )
        self._cf_selection = QComboBox()
        self._cf_selection.addItems(["strongest", "first"])
        self._cf_selection.setToolTip(
            "Which edge on each caliper: the strongest, or the first (innermost) "
            "— use 'first' when a brighter chamfer ring round the bore gives a "
            "stronger outer edge"
        )
        self._cf_strength = _spin(1, 255)
        self._cf_strength.setToolTip(
            "Weakest edge a caliper accepts, in grey levels per pixel of the edge profile"
        )
        self._cf_tolerance = _dspin(0.1, 50.0, 0.5, 1)
        self._cf_tolerance.setSuffix(" px")
        self._cf_tolerance.setToolTip(
            "An edge point within this distance of the fitted rim is an inlier. "
            "Keep it above Max Fit RMS, or the RMS check can never fail."
        )
        self._cf_min_inliers = _dspin(0.0, 1.0, 0.05)
        self._cf_min_inliers.setToolTip(
            "Share of all calipers whose edge must agree with the circle — 0.30 "
            "accepts a bore with about a third of its rim visible. 'Confidence "
            "Threshold' above filters the same number again."
        )
        self._cf_max_rms = _dspin(0.1, 50.0, 0.1, 1)
        self._cf_max_rms.setSuffix(" px")
        self._cf_max_rms.setToolTip("Largest RMS distance of the inliers from the fitted rim")
        self._cf_min_diameter = _spin(0, 4000)
        self._cf_max_diameter = _spin(0, 4000)
        size_hint = (
            "Gate on the fitted diameter in px; 0 disables it. Auto Sweep sets "
            "both from the ROI you draw."
        )
        self._cf_min_diameter.setToolTip(size_hint)
        self._cf_max_diameter.setToolTip(size_hint)
        form.addRow(hint)
        form.addRow("Seed Threshold", self._cf_seed_threshold)
        form.addRow("Blur Kernel", self._cf_blur)
        form.addRow("Calipers", self._cf_calipers)
        form.addRow("Caliper Width", self._cf_width)
        form.addRow("Search Band (± seed radius)", self._cf_band)
        form.addRow("Edge Polarity", self._cf_polarity)
        form.addRow("Edge Selection", self._cf_selection)
        form.addRow("Min Edge Strength", self._cf_strength)
        form.addRow("Inlier Tolerance", self._cf_tolerance)
        form.addRow("Min Inlier Ratio", self._cf_min_inliers)
        form.addRow("Max Fit RMS", self._cf_max_rms)
        form.addRow("Min Hole Diameter (px)", self._cf_min_diameter)
        form.addRow("Max Hole Diameter (px)", self._cf_max_diameter)
        return box

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
            "Camera' frame — thousands of combinations, not just two. "
            "The template_* strategies are swept one stage at a time — pick it "
            "under Stage."
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
        # Only shown for a multi-stage strategy (_SWEEP_STAGES); filled by
        # _update_sweep_availability with that strategy's stages.
        self._sweep_stage_label = QLabel("Stage")
        self._sweep_stage = QComboBox()
        self._sweep_stage.currentIndexChanged.connect(self._on_sweep_stage_changed)
        run_row.addWidget(self._sweep_stage_label)
        run_row.addWidget(self._sweep_stage)
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
        self._downscale.setChecked(bool(common.get("downscale_enabled", False)))
        self._downscale_factor.setValue(
            float(common.get("downscale_factor", DEFAULT_DOWNSCALE_FACTOR))
        )
        self._rerun.setChecked(bool(common.get("ng_rerun_enabled", False)))
        self._rerun_count.setValue(int(common.get("ng_rerun_count", 1)))
        self._strategy.setCurrentText(cfg.get("active_detector", "opencv"))

        combined = cfg.get("template_opencv", {})
        self._to_regions.setValue(int(combined.get("max_regions", 1)))
        self._to_padding.setValue(int(combined.get("region_padding_px", 0)))

        combined_blob = cfg.get("template_blob_analysis", {})
        self._tb_regions.setValue(int(combined_blob.get("max_regions", 1)))
        self._tb_padding.setValue(int(combined_blob.get("region_padding_px", 0)))

        blob = cfg.get("blob_analysis", {})
        self._ba_threshold.setValue(int(blob.get("threshold", 40)))
        self._ba_blur.setValue(int(blob.get("blur_kernel_size", 7)))
        self._ba_morph.setValue(int(blob.get("morphology_kernel_size", 5)))
        self._ba_min_area.setValue(int(blob.get("min_area_px", 500)))
        self._ba_circularity.setValue(float(blob.get("min_circularity", 0.8)))
        self._ba_min_diameter.setValue(int(blob.get("min_hole_diameter_px", 0)))
        self._ba_max_diameter.setValue(int(blob.get("max_hole_diameter_px", 0)))

        combined_caliper = cfg.get("template_caliper_based_circle_fitting", {})
        self._tc_regions.setValue(int(combined_caliper.get("max_regions", 1)))
        self._tc_padding.setValue(int(combined_caliper.get("region_padding_px", 0)))

        caliper = cfg.get("caliper_based_circle_fitting", {})
        self._cf_seed_threshold.setValue(int(caliper.get("coarse_threshold", 0)))
        self._cf_blur.setValue(int(caliper.get("blur_kernel_size", 5)))
        self._cf_calipers.setValue(int(caliper.get("num_calipers", 72)))
        self._cf_width.setValue(int(caliper.get("caliper_width_px", 5)))
        self._cf_band.setValue(float(caliper.get("search_band", 0.5)))
        self._cf_polarity.setCurrentText(str(caliper.get("edge_polarity", "dark_to_light")))
        self._cf_selection.setCurrentText(str(caliper.get("edge_selection", "strongest")))
        self._cf_strength.setValue(int(caliper.get("min_edge_strength", 10)))
        self._cf_tolerance.setValue(float(caliper.get("inlier_threshold_px", 3.0)))
        self._cf_min_inliers.setValue(float(caliper.get("min_inlier_ratio", 0.3)))
        self._cf_max_rms.setValue(float(caliper.get("max_rms_px", 2.0)))
        self._cf_min_diameter.setValue(int(caliper.get("min_hole_diameter_px", 0)))
        self._cf_max_diameter.setValue(int(caliper.get("max_hole_diameter_px", 0)))

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
                # The factor and count are kept while switched off, so turning
                # either back on restores the value it had.
                "downscale_enabled": self._downscale.isChecked(),
                "downscale_factor": self._downscale_factor.value(),
                "ng_rerun_enabled": self._rerun.isChecked(),
                "ng_rerun_count": self._rerun_count.value(),
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
            "template_opencv": {
                "max_regions": self._to_regions.value(),
                "region_padding_px": self._to_padding.value(),
            },
            "blob_analysis": {
                "threshold": self._ba_threshold.value(),
                "blur_kernel_size": self._ba_blur.value(),
                "morphology_kernel_size": self._ba_morph.value(),
                "min_area_px": self._ba_min_area.value(),
                "min_circularity": self._ba_circularity.value(),
                "min_hole_diameter_px": self._ba_min_diameter.value(),
                "max_hole_diameter_px": self._ba_max_diameter.value(),
            },
            "template_blob_analysis": {
                "max_regions": self._tb_regions.value(),
                "region_padding_px": self._tb_padding.value(),
            },
            "caliper_based_circle_fitting": {
                "coarse_threshold": self._cf_seed_threshold.value(),
                "blur_kernel_size": self._cf_blur.value(),
                "num_calipers": self._cf_calipers.value(),
                "caliper_width_px": self._cf_width.value(),
                "search_band": self._cf_band.value(),
                "edge_polarity": self._cf_polarity.currentText(),
                "edge_selection": self._cf_selection.currentText(),
                "min_edge_strength": self._cf_strength.value(),
                "inlier_threshold_px": self._cf_tolerance.value(),
                "min_inlier_ratio": self._cf_min_inliers.value(),
                "max_rms_px": self._cf_max_rms.value(),
                "min_hole_diameter_px": self._cf_min_diameter.value(),
                "max_hole_diameter_px": self._cf_max_diameter.value(),
            },
            "template_caliper_based_circle_fitting": {
                "max_regions": self._tc_regions.value(),
                "region_padding_px": self._tc_padding.value(),
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
        factor = self._engine.downscale_factor(camera_index)
        timing = f"{result.processing_ms:.1f} ms" + (
            f" (detected at ÷{factor:g})" if factor > 1.0 else ""
        )
        best = result.best
        if best is not None:
            self._result_label.setText(
                f"{len(result.holes)} hole(s) — best ({best.x_px:.1f}, {best.y_px:.1f}) px, "
                f"Ø {best.diameter_px:.1f} px, conf {best.confidence:.2f} "
                f"— {timing}"
            )
        else:
            self._result_label.setText(f"No hole found{self._where_searched(result)} — {timing}")
        self._refresh_verdict()

    def _where_searched(self, result: DetectionResult) -> str:
        """For a two-stage strategy, *which* stage came up empty: the
        template not being found and an empty template match need opposite
        fixes, and the overlay alone does not say so in words."""
        if result.regions:
            best = max(region.score for region in result.regions)
            return f" inside the template match (score {best:.2f})"
        if self._strategy.currentText() in _TWO_STAGE_DETECTORS:
            return " — the template itself was not found"
        return ""

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
                f"position tolerance not reached.{self._rerun_note()}",
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
                f"tolerance by {deviation - tolerance:.2f} mm. {position}"
                f"{self._rerun_note()}",
            )
        else:
            self._set_verdict(
                "GOOD",
                f"GOOD — deviation {deviation:.2f} mm within the {tolerance:.2f} mm "
                f"tolerance ({tolerance - deviation:.2f} mm margin). {position}",
            )

    def _rerun_note(self) -> str:
        """What a real cycle does next with this NG, when "Required another
        run" is on — Test itself is a single capture and never re-runs."""
        if not self._rerun.isChecked():
            return ""
        return (
            f" (A real cycle would take a new picture and check again, up to "
            f"{self._rerun_count.value()} more time(s), before reporting NG.)"
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
        self._sweep_rows_strategy = None
        self._sweep_table.setRowCount(0)
        self._sweep_status.setText("Run 'Test on Camera' first.")

    # ---------------------------------------------------------- auto sweep
    def _update_sweep_availability(self, strategy: str) -> None:
        stages = _SWEEP_STAGES.get(strategy, ())
        self._fill_sweep_stages(stages)
        self._sweep_stage_label.setVisible(bool(stages))
        self._sweep_stage.setVisible(bool(stages))
        supported = bool(stages) or strategy in _SWEEP_GRID_BUILDERS
        self._sweep_btn.setEnabled(supported)
        if not supported:
            self._sweep_btn.setToolTip(self._unsweepable_reason(strategy))
        else:
            self._on_sweep_stage_changed()

    def _fill_sweep_stages(self, stages: tuple[tuple[str, str, str], ...]) -> None:
        """Put *stages* in the Stage combo, keeping the selected stage when the
        list is unchanged (a camera reload must not jump back to Stage 1)."""
        wanted = [name for name, _label, _hint in stages]
        current = [self._sweep_stage.itemData(i) for i in range(self._sweep_stage.count())]
        if wanted == current:
            return
        blocked = self._sweep_stage.blockSignals(True)
        try:
            self._sweep_stage.clear()
            for name, label, hint in stages:
                self._sweep_stage.addItem(label, name)
                self._sweep_stage.setItemData(
                    self._sweep_stage.count() - 1, hint, Qt.ItemDataRole.ToolTipRole
                )
        finally:
            self._sweep_stage.blockSignals(blocked)

    def _on_sweep_stage_changed(self, _index: int = -1) -> None:
        """The run button and the combo say what the ROI must mark for the
        selected stage — the two stages want different boxes."""
        hint = self._sweep_stage.currentData(Qt.ItemDataRole.ToolTipRole) or ""
        if self._sweep_stage.isHidden():
            hint = ""
        self._sweep_stage.setToolTip(hint)
        self._sweep_btn.setToolTip(hint)

    def _form_title(self, strategy: str) -> str:
        """The title *strategy*'s form carries right now ("Stage 2 — ..."
        while template_opencv shows it), for the sweep's status lines."""
        form = self._forms.get(strategy)
        return form.title() if form is not None else strategy

    def _sweep_target(self, strategy: str) -> str:
        """The strategy whose parameter block the sweep tunes: the Active
        Detector itself, or the selected stage of a multi-stage one."""
        if strategy in _SWEEP_STAGES:
            return str(self._sweep_stage.currentData() or "")
        return strategy

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
        target = self._sweep_target(strategy)
        grid_builder = _SWEEP_GRID_BUILDERS.get(target)
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
            self._collect().get(target, {}),
            min_hole_diameter_px=min_diameter,
            max_hole_diameter_px=max_diameter,
        )
        try:
            # template_matching derives its scale range here, which needs the
            # template itself — an unset or unreadable one has to be reported
            # as the configuration problem it is, not as "found nothing".
            combos = grid_builder(base_params)
            detector = _SWEEP_DETECTORS[target](base_params)
            two_stage = _TWO_STAGE_DETECTORS.get(strategy)
            if two_stage is not None and target == two_stage.HOLE_STAGE:
                views = self._template_region_views(roi, two_stage)
            else:
                cropped, offset_x, offset_y = self._crop_around_roi(self._last_frame, roi)
                if self._normalize.isChecked():  # match what the live/tested detector sees
                    cropped = normalize_image(cropped)
                views = [(cropped, offset_x, offset_y)]
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Auto Sweep", str(exc))
            return
        if not views:
            return  # _template_region_views already said why

        found: list[tuple[dict, Hole]] = []
        tried = self._run_sweep_grid(combos, detector, views, roi, found)

        self._sweep_rows = sorted(found, key=lambda item: item[1].confidence, reverse=True)[:20]
        self._sweep_rows_strategy = target
        self._fill_sweep_table(target)

        cancelled = self._sweep_cancel_requested
        note = " (stopped early)" if cancelled else ""
        if not self._sweep_rows:
            stage_hint = ""
            if target != strategy and target == TemplateRegionDetector.TEMPLATE_STAGE:
                stage_hint = (
                    " For Stage 1 the ROI must surround the whole feature the "
                    "template shows — a box around the bore alone is smaller "
                    "than the template, so it can never be matched there."
                )
            self._sweep_status.setText(
                f"No combination found a hole in the ROI ({tried} of {len(combos)} tried{note}) "
                f"— widen the ROI, check it still marks the hole, or verify the part is "
                f"actually visible in this frame.{stage_hint}"
            )
            return

        _best_params, best_hole = self._sweep_rows[0]
        self._sweep_status.setText(
            f"Best: confidence {best_hole.confidence:.2f}, Ø {best_hole.diameter_px:.1f} px "
            f"(diameter gate {min_diameter:.0f}-{max_diameter:.0f} px from the ROI; "
            f"{len(self._sweep_rows)} of {tried}/{len(combos)} tried found it{note} — click a "
            f"row to preview, 'Apply Best' to load all its parameters into the "
            f"{self._form_title(target)} form)"
        )

    def _template_region_views(
        self, roi: tuple[int, int, int, int], two_stage: type[TemplateRegionDetector]
    ) -> list[tuple[np.ndarray, int, int]]:
        """A two-stage strategy's Stage 2 trial images: each Stage 1 match
        that overlaps *roi*, cropped from the (normalised, if enabled) frame.

        That is precisely what a real cycle hands the hole stage — the padded
        match, not a box around the hole — and both OpenCV and blob analysis
        threshold the whole crop they are given (blob analysis even picks the
        *largest* blob in it), so tuning them on any other crop would tune
        them for an image they never see. Stage 1 runs **once** here, with
        the form's current values (template, threshold, scales, and
        *two_stage*'s own regions searched and padding).

        Returns an empty list — after telling the operator why — when Stage 1
        finds nothing, or nothing near the ROI.

        Raises:
            VisionSystemError: Stage 1 is misconfigured (no template, ...).
        """
        frame = self._last_frame
        if self._normalize.isChecked():  # the engine normalises the whole frame
            frame = normalize_image(frame)
        locator = two_stage(two_stage.params_from_camera_config(self._collect()))
        regions = locator.detect(frame).regions
        x, y, w, h = roi
        covering = [
            region for region in regions
            if region.x < x + w and x < region.x + region.width
            and region.y < y + h and y < region.y + region.height
        ]
        if not covering:
            QMessageBox.information(
                self, "Auto Sweep",
                "Stage 1 found no template match in this frame. Sweep "
                "'Stage 1 — Template Matching' first and 'Apply Best', then "
                "sweep Stage 2." if not regions else
                "Stage 1's template match does not cover the ROI you drew — "
                "Stage 2 only ever searches inside the match. Re-draw the ROI "
                "around the hole inside the orange match box, or re-tune Stage 1.",
            )
            return []
        return [
            (frame[r.y : r.y + r.height, r.x : r.x + r.width], r.x, r.y) for r in covering
        ]

    def _run_sweep_grid(
        self,
        combos: list[dict],
        detector: HoleDetector,
        views: list[tuple[np.ndarray, int, int]],
        roi: tuple[int, int, int, int],
        found: list[tuple[dict, Hole]],
    ) -> int:
        """Run every combination in *combos* against every ``(image,
        offset_x, offset_y)`` in *views* — one crop around the ROI, or
        a two-stage strategy's Stage 2 matched regions — appending (params, hole)
        to *found* for each that lands a candidate inside *roi*. Returns how
        many were actually tried (may be less than
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
                    holes = [
                        replace(hole, x_px=hole.x_px + offset_x, y_px=hole.y_px + offset_y)
                        for image, offset_x, offset_y in views
                        for hole in detector.detect(image).holes
                    ]
                except VisionSystemError:
                    pass  # this combination is not a valid configuration — skip it
                else:
                    holes.sort(key=lambda candidate: candidate.confidence, reverse=True)
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
        target = self._sweep_rows_strategy or self._strategy.currentText()
        widgets = self._param_widgets.get(target, {})
        for key, widget in widgets.items():
            if key in params:
                self._set_widget_value(widget, params[key])
        self._sweep_status.setText(
            f"Applied the winning combination to the "
            f"{self._form_title(target)} form — review and press "
            f"'Save && Apply' to make it live."
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

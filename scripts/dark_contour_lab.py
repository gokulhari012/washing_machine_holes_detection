"""Interactive lab for tuning the dark circular/elliptical contour pipeline.

Loads every image in a folder ("test images/" by default), lets you draw an
ROI over the region you care about, tune the threshold/morphology/shape
gates with live feedback, and step through images one at a time (list
selection or arrow keys) to see whether the same settings hold up across the
whole set.

Companion to ``dark_contour_test.py``, which owns the actual detection
pipeline (``find_dark_blobs`` / ``draw_overlay``) — this module is the
interactive front end; that one is the scriptable/batch one.

Usage:
    python scripts/dark_contour_lab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from PySide6.QtCore import QRectF, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dark_contour_test import Blob, draw_overlay, find_dark_blobs  # noqa: E402
from ui.theme import COLOR_GOOD, apply_dark_theme  # noqa: E402
from ui.widgets import ImageView, RoiEditor  # noqa: E402

DEFAULT_INPUT_DIR = BASE_DIR / "test images"
DEFAULT_OUTPUT_DIR = BASE_DIR / "scripts" / "output"


def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    return spin


def _dspin(minimum: float, maximum: float, step: float, value: float, decimals: int = 2) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin


class DarkContourLab(QMainWindow):
    """Load images, draw an ROI, tune gates, inspect results one image at a time."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dark Contour Lab")
        self.resize(1440, 900)

        self._paths: list[Path] = []
        self._current_path: Path | None = None
        self._current_image: np.ndarray | None = None
        self._current_blobs: list[Blob] = []
        self._current_mask: np.ndarray | None = None

        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(80)
        self._recompute_timer.timeout.connect(self._recompute)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        root.addWidget(self._build_image_list(), stretch=0)
        root.addLayout(self._build_center(), stretch=1)
        root.addWidget(self._build_controls(), stretch=0)

        self._load_folder(DEFAULT_INPUT_DIR)

    # --------------------------------------------------------------- layout
    def _build_image_list(self) -> QWidget:
        box = QGroupBox("Images")
        box.setFixedWidth(240)
        layout = QVBoxLayout(box)
        open_btn = QPushButton("Open Folder…")
        open_btn.clicked.connect(self._on_open_folder)
        layout.addWidget(open_btn)
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_image_selected)
        layout.addWidget(self._list, stretch=1)
        return box

    def _build_center(self) -> QVBoxLayout:
        center = QVBoxLayout()

        self._roi_editor = RoiEditor()
        self._roi_editor.setMinimumSize(560, 420)
        self._roi_editor.roi_changed.connect(self._on_roi_changed)
        center.addWidget(self._roi_editor, stretch=3)

        roi_row = QHBoxLayout()
        self._roi_mode_cb = QCheckBox("Draw ROI (drag on image)")
        self._roi_mode_cb.toggled.connect(self._roi_editor.set_roi_mode)
        clear_roi_btn = QPushButton("Clear ROI")
        clear_roi_btn.clicked.connect(self._on_clear_roi)
        self._roi_label = QLabel("ROI: full image")
        self._roi_label.setProperty("class", "dim")
        roi_row.addWidget(self._roi_mode_cb)
        roi_row.addWidget(clear_roi_btn)
        roi_row.addWidget(self._roi_label, stretch=1)
        center.addLayout(roi_row)

        center.addWidget(QLabel("Threshold mask:"))
        self._mask_view = ImageView()
        self._mask_view.setMinimumHeight(160)
        self._mask_view.setMaximumHeight(220)
        center.addWidget(self._mask_view, stretch=1)

        self._status_label = QLabel("Load images to begin.")
        self._status_label.setProperty("class", "dim")
        center.addWidget(self._status_label)
        return center

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(300)
        layout = QVBoxLayout(panel)

        gates_box = QGroupBox("Threshold Controls")
        form = QFormLayout(gates_box)

        self._threshold = _spin(0, 255, 60)
        self._adaptive = QCheckBox("Adaptive threshold")
        self._blur = _spin(1, 31, 5)
        self._morph_kernel = _spin(1, 31, 5)
        self._morph_iter = _spin(1, 50, 1)
        self._min_diameter = _spin(1, 2000, 8)
        self._max_diameter = _spin(1, 4000, 120)
        self._min_circularity = _dspin(0.0, 1.0, 0.05, 0.55)
        self._min_aspect = _dspin(0.0, 1.0, 0.05, 0.35)

        for widget in (
            self._threshold, self._blur, self._morph_kernel, self._morph_iter,
            self._min_diameter, self._max_diameter, self._min_circularity, self._min_aspect,
        ):
            widget.valueChanged.connect(self._schedule_recompute)
        self._adaptive.toggled.connect(self._on_adaptive_toggled)

        form.addRow("Threshold", self._threshold)
        form.addRow("", self._adaptive)
        form.addRow("Blur Kernel", self._blur)
        form.addRow("Morph Kernel", self._morph_kernel)
        form.addRow("Morph Iterations", self._morph_iter)
        form.addRow("Min Diameter (px)", self._min_diameter)
        form.addRow("Max Diameter (px)", self._max_diameter)
        form.addRow("Min Circularity", self._min_circularity)
        form.addRow("Min Aspect Ratio", self._min_aspect)
        layout.addWidget(gates_box)

        layout.addWidget(self._build_sweep_box())

        save_btn = QPushButton("Save Overlay + Mask…")
        save_btn.clicked.connect(self._on_save)
        layout.addWidget(save_btn)

        results_box = QGroupBox("Detected Contours")
        results_layout = QVBoxLayout(results_box)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["#", "Shape", "Center (x, y)", "Ø px", "Circ."])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        results_layout.addWidget(self._table)
        layout.addWidget(results_box, stretch=1)

        return panel

    def _build_sweep_box(self) -> QWidget:
        box = QGroupBox("Threshold Sweep (in ROI)")
        layout = QVBoxLayout(box)

        range_row = QHBoxLayout()
        self._sweep_start = _spin(0, 255, 20)
        self._sweep_end = _spin(0, 255, 100)
        self._sweep_step = _spin(1, 50, 1)
        for label, widget in (("From", self._sweep_start), ("To", self._sweep_end), ("Step", self._sweep_step)):
            range_row.addWidget(QLabel(label))
            range_row.addWidget(widget)
        layout.addLayout(range_row)

        sweep_btn = QPushButton("Run Sweep")
        sweep_btn.clicked.connect(self._on_run_sweep)
        layout.addWidget(sweep_btn)

        self._sweep_status = QLabel("Draw an ROI, then run the sweep.")
        self._sweep_status.setProperty("class", "dim")
        self._sweep_status.setWordWrap(True)
        layout.addWidget(self._sweep_status)

        self._sweep_table = QTableWidget(0, 4)
        self._sweep_table.setHorizontalHeaderLabels(["Thresh", "Blobs", "Best Circ.", "Best Ø"])
        self._sweep_table.horizontalHeader().setStretchLastSection(True)
        self._sweep_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._sweep_table.setMaximumHeight(180)
        self._sweep_table.cellClicked.connect(self._on_sweep_row_clicked)
        layout.addWidget(self._sweep_table)

        return box

    # ------------------------------------------------------------- image list
    def _load_folder(self, folder: Path) -> None:
        if not folder.is_dir():
            self._status_label.setText(f"Folder not found: {folder}")
            return
        paths = (
            sorted(folder.glob("*.bmp")) + sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
        )
        self._paths = paths
        self._list.clear()
        for path in paths:
            self._list.addItem(path.name)
        if paths:
            self._list.setCurrentRow(0)
        else:
            self._status_label.setText(f"No images found in {folder}")

    def _on_open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", str(DEFAULT_INPUT_DIR))
        if folder:
            self._load_folder(Path(folder))

    def _on_image_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._paths):
            return
        path = self._paths[row]
        image = cv2.imread(str(path))
        if image is None:
            self._status_label.setText(f"Could not read {path.name}")
            return
        self._current_path = path
        self._current_image = image
        self._recompute()  # update the pixmap *before* changing the transform — avoids a stale-frame flash
        self._zoom_to_roi(*self._roi_editor.current_roi())  # keep the same ROI framed across images

    # ---------------------------------------------------------------- roi
    def _on_roi_changed(self, x: int, y: int, w: int, h: int) -> None:
        self._recompute()
        self._zoom_to_roi(x, y, w, h)

    def _on_clear_roi(self) -> None:
        self._roi_editor.clear_roi()
        self._recompute()
        self._zoom_to_roi(0, 0, 0, 0)

    def _zoom_to_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Fit the view to the ROI with a margin around it; no ROI -> fit whole image."""
        if w <= 0 or h <= 0:
            self._roi_editor.reset_view()
            return
        margin = max(w, h) * 0.25
        self._roi_editor.zoom_to_rect(QRectF(x - margin, y - margin, w + 2 * margin, h + 2 * margin))

    # ------------------------------------------------------------- sweep
    def _on_run_sweep(self) -> None:
        if self._current_image is None:
            return
        roi_x, roi_y, roi_w, roi_h = self._roi_editor.current_roi()
        if roi_w <= 0 or roi_h <= 0:
            self._sweep_status.setText("Draw an ROI first — the sweep only scans inside it.")
            return
        region = self._current_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        start, end = self._sweep_start.value(), self._sweep_end.value()
        if start > end:
            start, end = end, start
        step = max(1, self._sweep_step.value())

        common_kwargs = dict(
            blur_kernel=self._blur.value(),
            morph_kernel=self._morph_kernel.value(),
            morph_iterations=self._morph_iter.value(),
            min_diameter=self._min_diameter.value(),
            max_diameter=self._max_diameter.value(),
            min_circularity=self._min_circularity.value(),
            min_aspect_ratio=self._min_aspect.value(),
        )

        rows: list[tuple[int, int, float | None, float | None]] = []
        best: tuple[float, int, Blob] | None = None
        for t in range(start, end + 1, step):
            blobs, _ = find_dark_blobs(region, threshold=t, adaptive=False, **common_kwargs)
            top = max(blobs, key=lambda b: b.circularity) if blobs else None
            rows.append((t, len(blobs), top.circularity if top else None, top.diameter if top else None))
            if top is not None and (best is None or top.circularity > best[0]):
                best = (top.circularity, t, top)

        best_threshold = best[1] if best else None
        self._fill_sweep_table(rows, best_threshold)

        if best is None:
            self._sweep_status.setText(f"No contours found for any threshold in [{start}, {end}].")
            return

        _, best_t, blob = best
        self._sweep_status.setText(
            f"Best at threshold {best_t}: circularity {blob.circularity:.2f}, "
            f"Ø {blob.diameter:.1f}px, aspect {blob.aspect_ratio:.2f} ({blob.shape})"
        )
        self._adaptive.setChecked(False)
        self._threshold.setValue(best_t)
        self._recompute()

    def _fill_sweep_table(
        self, rows: list[tuple[int, int, float | None, float | None]], best_threshold: int | None
    ) -> None:
        self._sweep_table.setRowCount(len(rows))
        highlight = QColor(COLOR_GOOD)
        highlight.setAlpha(70)
        for row_index, (threshold, count, circularity, diameter) in enumerate(rows):
            values = [
                str(threshold),
                str(count),
                f"{circularity:.2f}" if circularity is not None else "—",
                f"{diameter:.1f}" if diameter is not None else "—",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if threshold == best_threshold:
                    item.setBackground(highlight)
                self._sweep_table.setItem(row_index, col, item)

    def _on_sweep_row_clicked(self, row: int, _column: int) -> None:
        item = self._sweep_table.item(row, 0)
        if item is None:
            return
        self._adaptive.setChecked(False)
        self._threshold.setValue(int(item.text()))
        self._recompute()

    # ------------------------------------------------------------- controls
    def _on_adaptive_toggled(self, checked: bool) -> None:
        self._threshold.setEnabled(not checked)
        self._recompute()

    def _schedule_recompute(self) -> None:
        self._recompute_timer.start()

    # ---------------------------------------------------------------- core
    def _recompute(self) -> None:
        if self._current_image is None:
            return
        image = self._current_image
        roi_x, roi_y, roi_w, roi_h = self._roi_editor.current_roi()
        if roi_w > 0 and roi_h > 0:
            region = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            self._roi_label.setText(f"ROI: ({roi_x}, {roi_y}) {roi_w}×{roi_h}")
        else:
            region = image
            roi_x = roi_y = 0
            self._roi_label.setText("ROI: full image")

        blobs, mask = find_dark_blobs(
            region,
            threshold=self._threshold.value(),
            adaptive=self._adaptive.isChecked(),
            blur_kernel=self._blur.value(),
            morph_kernel=self._morph_kernel.value(),
            morph_iterations=self._morph_iter.value(),
            min_diameter=self._min_diameter.value(),
            max_diameter=self._max_diameter.value(),
            min_circularity=self._min_circularity.value(),
            min_aspect_ratio=self._min_aspect.value(),
        )
        for blob in blobs:
            blob.x += roi_x
            blob.y += roi_y

        self._current_blobs = blobs
        self._current_mask = mask

        self._roi_editor.set_frame(draw_overlay(image, blobs))
        self._mask_view.set_frame(mask)
        self._fill_table(blobs)

        name = self._current_path.name if self._current_path else "—"
        self._status_label.setText(f"{name} — {len(blobs)} contour(s)")

    def _fill_table(self, blobs: list[Blob]) -> None:
        self._table.setRowCount(len(blobs))
        for row, blob in enumerate(blobs):
            values = [
                str(row), blob.shape, f"({blob.x:.1f}, {blob.y:.1f})",
                f"{blob.diameter:.1f}", f"{blob.circularity:.2f}",
            ]
            for col, value in enumerate(values):
                self._table.setItem(row, col, QTableWidgetItem(value))

    # -------------------------------------------------------------- export
    def _on_save(self) -> None:
        if self._current_image is None or self._current_path is None or self._current_mask is None:
            return
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = self._current_path.stem
        overlay = draw_overlay(self._current_image, self._current_blobs)
        cv2.imwrite(str(DEFAULT_OUTPUT_DIR / f"{stem}_overlay.png"), overlay)
        cv2.imwrite(str(DEFAULT_OUTPUT_DIR / f"{stem}_mask.png"), self._current_mask)
        self._status_label.setText(f"Saved {stem}_overlay.png / {stem}_mask.png to {DEFAULT_OUTPUT_DIR}")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Dark Contour Lab")
    apply_dark_theme(app)
    window = DarkContourLab()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

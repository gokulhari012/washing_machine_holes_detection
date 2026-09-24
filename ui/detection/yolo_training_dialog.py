"""Dataset labelling and YOLO training, launched from the Detection page.

Opened by the "Dataset && Training..." button that appears on the Detection
page only while ``yolo`` is the selected strategy, because it is the only
strategy that needs a trained file before it can do anything at all.

Why a dialog rather than more rows on the Detection page: a run prints
continuously for minutes to hours, and that console is most of the value —
it is where an operator sees the loss falling, the mAP, and the traceback
when a dataset is malformed. Inline it would either be too small to read or
would crowd out the parameter form beside it. The dialog is modeless, so the
rest of the application keeps working (an inspection cycle included) while
training runs.

Two steps, in the order they are actually done:

1. **Label** — opens the bundled ``YoloLabel.exe`` against a folder of
   captured frames. It writes one ``.txt`` of boxes beside each image, which
   is the dataset layout the training step expects; the dialog counts both
   and says what it found, since an unlabelled image trains as a background
   sample rather than failing loudly.
2. **Train** — writes a ``data.yaml`` into the output folder and runs the
   training in a :class:`~workers.yolo_training_worker.YoloTrainingWorker`.

On success it offers to load the produced ``best.pt`` straight into the
camera's "Model Weights" field, which is the whole point of doing this here
instead of at a command prompt — the trained model is one click from being
the strategy the station runs.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.utilities.exceptions import VisionSystemError
from services.yolo_training_service import YOLO_MODELS, YoloTrainingService
from workers.yolo_training_worker import YoloTrainingWorker

#: Keeps the log bounded — a long run prints tens of thousands of lines and an
#: unbounded QPlainTextEdit would grow until the station ran out of memory.
LOG_LINE_LIMIT = 5000


def _browse_row(line_edit: QLineEdit, on_click) -> QWidget:
    """A path field with a "..." button beside it."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    button = QPushButton("…")
    button.setFixedWidth(32)
    button.clicked.connect(on_click)
    row.addWidget(line_edit)
    row.addWidget(button)
    holder = QWidget()
    holder.setLayout(row)
    return holder


class YoloTrainingDialog(QDialog):
    """Label a dataset and train a YOLO model without leaving the station."""

    def __init__(
        self,
        service: YoloTrainingService,
        settings: dict | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("YOLO — Dataset & Training")
        self.setMinimumSize(760, 620)
        self._service = service
        self._worker: YoloTrainingWorker | None = None
        self._trained_weights = ""

        root = QVBoxLayout(self)
        root.addWidget(self._build_label_box())
        root.addWidget(self._build_train_box())
        root.addWidget(QLabel("Training output"))
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(LOG_LINE_LIMIT)
        self._log.setProperty("class", "mono")
        root.addWidget(self._log, stretch=1)
        root.addLayout(self._build_buttons())

        self._load(settings or {})
        self._refresh_dataset_summary()

    # ------------------------------------------------------------- step 1
    def _build_label_box(self) -> QWidget:
        box = QGroupBox("1 — Label the images")
        layout = QVBoxLayout(box)
        hint = QLabel(
            "Put the captured frames in a folder, then label every hole in "
            "them. The tool writes a .txt of boxes beside each image — that "
            "folder is then the dataset below."
        )
        hint.setWordWrap(True)
        hint.setProperty("class", "dim")
        layout.addWidget(hint)

        row = QHBoxLayout()
        self._label_btn = QPushButton("Open Labeling Tool")
        self._label_btn.clicked.connect(self._on_open_labeling_tool)
        row.addWidget(self._label_btn)
        self._label_status = QLabel("")
        self._label_status.setProperty("class", "dim")
        self._label_status.setWordWrap(True)
        row.addWidget(self._label_status, stretch=1)
        layout.addLayout(row)

        if not self._service.labeling_tool_available():
            self._label_btn.setEnabled(False)
            self._label_status.setText(
                f"Not found: {self._service.labeling_tool}. It ships in the "
                "source tree and is not copied into a built station."
            )
        return box

    def _on_open_labeling_tool(self) -> None:
        try:
            self._service.open_labeling_tool()
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Labeling Tool", str(exc))
            return
        self._label_status.setText(
            "Opened. Choose your image folder and the class list in the tool, "
            "then come back here when the folder is labelled."
        )

    # ------------------------------------------------------------- step 2
    def _build_train_box(self) -> QWidget:
        box = QGroupBox("2 — Train the model")
        form = QFormLayout(box)

        self._dataset = QLineEdit()
        self._dataset.setToolTip("Folder holding the labelled images and their .txt files")
        self._dataset.textChanged.connect(self._refresh_dataset_summary)
        form.addRow("Dataset Folder", _browse_row(self._dataset, self._pick_dataset))

        self._validation = QLineEdit()
        self._validation.setPlaceholderText("(optional — defaults to the dataset folder)")
        self._validation.setToolTip(
            "Held-out images to measure against. Left blank the dataset "
            "folder is used for both, exactly as training_all_folder.py does — "
            "training still works, but the reported mAP is measured on images "
            "the model already trained on, so read it as optimistic."
        )
        form.addRow("Validation Folder", _browse_row(self._validation, self._pick_validation))

        self._output = QLineEdit()
        self._output.setToolTip("Where the run's weights, plots and metrics are written")
        form.addRow("Output Folder", _browse_row(self._output, self._pick_output))

        self._dataset_status = QLabel("")
        self._dataset_status.setWordWrap(True)
        self._dataset_status.setProperty("class", "dim")
        form.addRow("", self._dataset_status)

        self._classes = QLineEdit("hole")
        self._classes.setToolTip(
            "What the class ids in the label files mean — a box records only "
            "the number, so data.yaml has to name it. Filled in automatically "
            "from the class list in the dataset folder when there is one; "
            "otherwise type them comma-separated in labelling order, class 0 "
            "first."
        )
        form.addRow("Class Names", self._classes)

        self._model = QComboBox()
        self._model.addItems(YOLO_MODELS)
        self._model.setToolTip(
            "Base weights to fine-tune, smallest first. yolo11n is fastest to "
            "train and to run — which matters here, because every inspection "
            "cycle waits for it. Step up only if the small model misses holes."
        )
        form.addRow("YOLO Model", self._model)

        self._epochs = QSpinBox()
        self._epochs.setRange(1, 10000)
        self._epochs.setValue(100)
        form.addRow("Epochs", self._epochs)

        self._imgsz = QSpinBox()
        self._imgsz.setRange(32, 4096)
        self._imgsz.setSingleStep(32)
        self._imgsz.setValue(640)
        self._imgsz.setToolTip(
            "Training resolution. Set the camera's 'Inference Size' to the "
            "same number afterwards. A wide ROI (3500x1200 on this station) "
            "is letterboxed into a square, so 640 leaves a 100 px bore only "
            "~18 px across — 1280 is usually the better trade here."
        )
        form.addRow("Image Size", self._imgsz)

        self._device = QComboBox()
        self._device.setEditable(True)
        self._device.addItems(["", "cpu", "0", "cuda:0"])
        self._device.setToolTip(
            "Blank lets ultralytics choose. Training on CPU works but is "
            "hours rather than minutes."
        )
        form.addRow("Device", self._device)
        return box

    # ------------------------------------------------------------ buttons
    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._start_btn = QPushButton("Start Training")
        self._start_btn.setProperty("class", "primary")
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        row.addWidget(self._start_btn)
        row.addWidget(self._stop_btn)
        row.addStretch()
        row.addWidget(close_btn)
        return row

    # ------------------------------------------------------------ pickers
    def _pick_dataset(self) -> None:
        self._pick_into(self._dataset, "Select the labelled dataset folder")

    def _pick_validation(self) -> None:
        self._pick_into(self._validation, "Select the validation folder")

    def _pick_output(self) -> None:
        self._pick_into(self._output, "Select where training results go")

    def _pick_into(self, target: QLineEdit, caption: str) -> None:
        folder = QFileDialog.getExistingDirectory(self, caption, target.text().strip())
        if folder:
            target.setText(folder)

    def _refresh_dataset_summary(self) -> None:
        """Say what is in the chosen folder, before a run is started on it.

        Also adopts the folder's own class list when it has one. The boxes
        carry only a class *index*, so the list the images were labelled with
        is the authority on what those indices mean — retyping it here is
        both a chore and a chance to get the order wrong.
        """
        folder = self._dataset.text().strip()
        if not folder:
            self._dataset_status.setText("")
            return
        try:
            summary = self._service.inspect_dataset(folder)
        except VisionSystemError as exc:
            self._dataset_status.setText(str(exc))
            return

        message = summary.describe()
        detected = self._service.detect_class_names(folder)
        if detected:
            self._classes.setText(", ".join(detected))
            message += f" — classes from this folder: {', '.join(detected)}"
        self._dataset_status.setText(message)

    # ----------------------------------------------------------- settings
    def _load(self, settings: dict) -> None:
        # Class names first, because setting the dataset path fires
        # _refresh_dataset_summary, and the class list found *in that folder*
        # has to win over the remembered one — it is what the images were
        # actually labelled with, and it may have been re-edited since.
        self._classes.setText(str(settings.get("class_names", "hole") or "hole"))
        self._dataset.setText(str(settings.get("dataset_dir", "") or ""))
        self._validation.setText(str(settings.get("validation_dir", "") or ""))
        self._output.setText(str(settings.get("output_dir", "") or ""))
        model = str(settings.get("model", YOLO_MODELS[0]))
        if model in YOLO_MODELS:
            self._model.setCurrentText(model)
        self._epochs.setValue(int(settings.get("epochs", 100) or 100))
        self._imgsz.setValue(int(settings.get("imgsz", 640) or 640))
        self._device.setCurrentText(str(settings.get("device", "") or ""))

    def settings(self) -> dict:
        """The dialog's choices, for the page to persist in app_config.json."""
        return {
            "dataset_dir": self._dataset.text().strip(),
            "validation_dir": self._validation.text().strip(),
            "output_dir": self._output.text().strip(),
            "class_names": self._classes.text().strip(),
            "model": self._model.currentText(),
            "epochs": self._epochs.value(),
            "imgsz": self._imgsz.value(),
            "device": self._device.currentText().strip(),
        }

    @property
    def trained_weights(self) -> str:
        """Path to the ``best.pt`` of the last successful run, else ""."""
        return self._trained_weights

    # ----------------------------------------------------------- training
    def _on_start(self) -> None:
        if self._worker is not None:
            return
        dataset = self._dataset.text().strip()
        if not dataset:
            QMessageBox.information(self, "Start Training", "Choose a dataset folder first.")
            return
        if not self._service.ultralytics_installed():
            QMessageBox.warning(
                self, "Start Training",
                "The 'ultralytics' package is not installed in this "
                "interpreter, so there is nothing to train with.\n\n"
                "Install it with:  pip install ultralytics",
            )
            return

        try:
            summary = self._service.inspect_dataset(dataset)
            if not summary.labelled:
                raise VisionSystemError(
                    f"No labelled images in {dataset}.\n\n{summary.describe()}\n\n"
                    "Label the folder with the tool in step 1 first — training "
                    "on unlabelled images teaches the model that every frame "
                    "is empty."
                )
            output = self._service.prepare_output_dir(self._output.text().strip())
            data_yaml = self._service.write_data_yaml(
                output / "data.yaml",
                dataset,
                self._validation.text().strip() or dataset,
                self._classes.text().split(","),
            )
            command = self._service.build_train_command(
                data_yaml=data_yaml,
                output_dir=output,
                model=self._model.currentText(),
                epochs=self._epochs.value(),
                imgsz=self._imgsz.value(),
                device=self._device.currentText().strip(),
            )
        except VisionSystemError as exc:
            QMessageBox.warning(self, "Start Training", str(exc))
            return

        if summary.unlabelled:
            proceed = QMessageBox.question(
                self, "Start Training",
                f"{summary.describe()}.\n\nImages with no label file are "
                "trained as background samples — deliberate for empty-part "
                "frames, a mistake if you simply have not labelled them yet.\n\n"
                "Start training anyway?",
            )
            if proceed != QMessageBox.StandardButton.Yes:
                return

        self._log.clear()
        self._trained_weights = ""
        self._append(f"$ {' '.join(command)}")
        self._worker = YoloTrainingWorker(
            command, environment=self._service.training_environment()
        )
        self._worker.output.connect(self._append)
        self._worker.finished_run.connect(self._on_finished)
        self._set_running(True)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is None:
            return
        self._append("Stopping…")
        self._stop_btn.setEnabled(False)
        self._worker.stop()

    def _on_finished(self, code: int, best_weights: str) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.wait()
            worker.deleteLater()
        self._set_running(False)

        if code == 0 and best_weights:
            self._trained_weights = best_weights
            self._append(f"Done. Trained weights: {best_weights}")
            use_it = QMessageBox.question(
                self, "Training Complete",
                f"Training finished.\n\n{best_weights}\n\n"
                "Load it into this camera's Model Weights field now?",
            )
            if use_it == QMessageBox.StandardButton.Yes:
                self.accept()  # the page reads trained_weights on accept
            return

        if code == 0:
            self._append("Training finished, but no best.pt was produced.")
            QMessageBox.warning(
                self, "Training Complete",
                "The run finished without producing a best.pt. Check the log.",
            )
            return
        self._append(f"Training failed (exit code {code}).")
        QMessageBox.warning(
            self, "Training Failed",
            f"Training exited with code {code}. The log has the details.",
        )

    def _set_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        # Editing the inputs mid-run would describe a run that is not the one
        # actually going; they would also be silently ignored.
        for widget in (
            self._dataset, self._validation, self._output, self._classes,
            self._model, self._epochs, self._imgsz, self._device,
        ):
            widget.setEnabled(not running)

    def _append(self, line: str) -> None:
        self._log.appendPlainText(line)

    # ------------------------------------------------------------- closing
    def closeEvent(self, event) -> None:
        """A run in progress owns a subprocess, so closing has to decide its fate.

        Stopping it is the only honest option: the worker's output pipe dies
        with this dialog, so a "leave it running" would keep a torch process
        holding the GPU with nothing reading it or able to stop it.
        """
        if self._worker is not None:
            answer = QMessageBox.question(
                self, "Training In Progress",
                "Training is still running. Stop it and close?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.stop()
            self._worker.wait()
            self._worker = None
        super().closeEvent(event)

"""Labelling-tool launch and YOLO training preparation.

The ``yolo`` detection strategy needs a trained ``.pt`` before it can do
anything, and producing one is a three-step job that used to happen entirely
outside this application: label images with the bundled ``YoloLabel.exe``,
write a ``data.yaml``, then run a training command. This service owns those
steps so the Detection page can offer them where the strategy is chosen.

It is deliberately Qt-free and starts **no** threads: it prepares and
validates, and hands back a command for :class:`~workers.yolo_training_worker`
to run off the GUI thread. Nothing here touches an inspection cycle — a
training run is offline tooling that happens while the station is idle.

Dataset shape
-------------
``YoloLabel.exe`` writes each image's boxes to a ``.txt`` of the same stem
*beside the image*, which is also what ultralytics falls back to when a path
contains no ``/images/`` segment — so one flat folder of images + labels is a
valid dataset and needs no rearranging. That is the shape
``training_all_folder.py`` already assumed, and the one
:meth:`inspect_dataset` checks.

A validation folder is optional and defaults to the training folder, matching
``training_all_folder.py``'s ``val_path = train_path``. That makes the
reported mAP optimistic — it is measured on images the model trained on — so
the dialog says so rather than letting the number be read as accuracy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from core.logging import get_logger
from core.utilities.enums import LogSource
from core.utilities.exceptions import TrainingError

logger = get_logger(LogSource.VISION)

#: Base weights offered in the dialog, smallest (fastest) first. n/s/m/l/x are
#: the ultralytics size ladder; the same list ``training_all_folder.py`` takes.
YOLO_MODELS = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
]

#: Class-list filenames the common labelling tools use. Only a tie-breaker:
#: detect_class_names finds the file by elimination, since YoloLabel lets the
#: operator call it anything (this station's is "lables.txt").
_CLASS_FILE_NAMES = ("classes.txt", "obj.names", "predefined_classes.txt")

#: Image suffixes ultralytics will pick up out of a dataset folder.
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

DEFAULT_LABELING_TOOL = Path("tools") / "YOLO Training" / "yolo labling" / "YoloLabel.exe"
DEFAULT_TRAIN_RUNNER = Path("tools") / "YOLO Training" / "run_training.py"


@dataclass(frozen=True)
class DatasetSummary:
    """What :meth:`YoloTrainingService.inspect_dataset` found in a folder."""

    images: int
    labelled: int
    empty_labels: int  # a label file that exists but has no boxes in it

    @property
    def unlabelled(self) -> int:
        return self.images - self.labelled

    def describe(self) -> str:
        """One line for the dialog, naming the problem when there is one.

        An unlabelled image is not an error — ultralytics reads it as a
        deliberate background/negative sample — but it is the single most
        common reason a run trains on far less data than the operator thinks,
        so it is always stated rather than only warned about.
        """
        if not self.images:
            return "No images found in this folder."
        parts = [f"{self.images} image(s), {self.labelled} labelled"]
        if self.unlabelled:
            parts.append(
                f"{self.unlabelled} with no label file (trained as background)"
            )
        if self.empty_labels:
            parts.append(f"{self.empty_labels} label file(s) with no boxes")
        return " — ".join(parts)


class YoloTrainingService:
    """Prepares YOLO datasets, launches the labelling tool, builds train commands."""

    def __init__(
        self,
        labeling_tool: Path | str | None = None,
        train_runner: Path | str | None = None,
    ) -> None:
        # Relative paths resolve against the app directory, which main.py has
        # already made the working directory (frozen builds included).
        self._labeling_tool = Path(labeling_tool or DEFAULT_LABELING_TOOL)
        self._train_runner = Path(train_runner or DEFAULT_TRAIN_RUNNER)

    # ------------------------------------------------------ labelling tool
    @property
    def labeling_tool(self) -> Path:
        return self._labeling_tool

    def labeling_tool_available(self) -> bool:
        return self._labeling_tool.is_file()

    def open_labeling_tool(self) -> None:
        """Launch ``YoloLabel.exe`` and return immediately.

        Started detached rather than waited on: labelling a folder is a
        session of its own, and the operator keeps using this application
        while it runs. ``cwd`` is set to the tool's own directory so it finds
        the Qt DLLs shipped beside it whatever directory the station's app
        was started from.

        Raises:
            TrainingError: the tool is missing (it is not copied into a
                PyInstaller build) or the OS refused to start it.
        """
        tool = self._labeling_tool
        if not tool.is_file():
            raise TrainingError(
                f"Labelling tool not found at {tool.resolve()}.\n\n"
                "It ships in the source tree under 'tools/YOLO Training/' and "
                "is not copied into a built station. Copy that folder beside "
                "the executable, or point 'labeling_tool' in app_config.json "
                "at wherever YoloLabel.exe lives."
            )
        try:
            subprocess.Popen([str(tool.resolve())], cwd=str(tool.resolve().parent))
        except OSError as exc:
            raise TrainingError(f"Could not start the labelling tool: {exc}") from exc
        logger.info("Launched labelling tool: %s", tool)

    # ------------------------------------------------------------- dataset
    @staticmethod
    def inspect_dataset(folder: Path | str) -> DatasetSummary:
        """Count images and their sidecar labels in *folder* (non-recursive).

        Raises:
            TrainingError: the folder does not exist.
        """
        path = Path(folder)
        if not path.is_dir():
            raise TrainingError(f"Dataset folder does not exist: {path}")

        images = 0
        labelled = 0
        empty = 0
        for entry in path.iterdir():
            if not entry.is_file() or entry.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            images += 1
            label = entry.with_suffix(".txt")
            if not label.is_file():
                continue
            labelled += 1
            try:
                if not label.read_text(encoding="utf-8", errors="replace").strip():
                    empty += 1
            except OSError:  # unreadable sidecar counts as empty, not fatal
                empty += 1
        return DatasetSummary(images=images, labelled=labelled, empty_labels=empty)

    @classmethod
    def detect_class_names(cls, folder: Path | str) -> list[str] | None:
        """The class list the labelling tool was given, read out of *folder*.

        ``YoloLabel.exe`` is opened with an image folder *and* a class-list
        file, and the boxes it writes carry only the class *index* — so that
        file is the only record of what index 0 means, and the operator
        should not have to retype it into ``data.yaml``. Worse than the
        retyping is the mismatch it invites: names in a different order from
        the ones the images were labelled with silently mislabel every box.

        Found by elimination rather than by a fixed filename, because the
        tool lets the operator name the file anything (this station's is
        ``lables.txt``): a ``.txt``/``.names`` with **no image of the same
        stem beside it** cannot be a label sidecar, so it is the class list.
        Conventional names win when several candidates exist; genuine
        ambiguity returns None rather than guessing.

        Returns:
            The names in index order, or None when there is no candidate,
            it is unreadable, or it holds box data rather than names.
        """
        path = Path(folder)
        if not path.is_dir():
            return None

        image_stems = set()
        candidates: dict[str, Path] = {}
        for entry in path.iterdir():
            if not entry.is_file():
                continue
            suffix = entry.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                image_stems.add(entry.stem)
            elif suffix in {".txt", ".names"}:
                candidates[entry.name.lower()] = entry

        orphans = [
            entry for name, entry in candidates.items() if entry.stem not in image_stems
        ]
        chosen: Path | None = None
        for known in _CLASS_FILE_NAMES:
            if known in candidates and candidates[known] in orphans:
                chosen = candidates[known]
                break
        if chosen is None:
            chosen = orphans[0] if len(orphans) == 1 else None
        if chosen is None:
            return None

        try:
            lines = chosen.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None
        names = [line.strip() for line in lines if line.strip()]
        if not names or any(cls._looks_like_a_box(name) for name in names):
            return None  # a stray label file, not a class list
        return names

    @staticmethod
    def _looks_like_a_box(line: str) -> bool:
        """Whether *line* is a YOLO annotation rather than a class name."""
        parts = line.split()
        if len(parts) != 5:
            return False
        try:
            [float(part) for part in parts]
        except ValueError:
            return False
        return True

    @staticmethod
    def write_data_yaml(
        destination: Path | str,
        train_dir: Path | str,
        val_dir: Path | str,
        class_names: list[str],
    ) -> Path:
        """Write the ultralytics ``data.yaml`` and return its path.

        Paths are written with forward slashes: a Windows dataset path lands
        in a YAML scalar, where ``D:\\parts\\new`` would make ``\\n`` an
        escape in any quoted reading of it.

        Raises:
            TrainingError: no class names, or the file could not be written.
        """
        names = [name.strip() for name in class_names if name.strip()]
        if not names:
            raise TrainingError(
                "At least one class name is required — it has to match the "
                "classes the images were labelled with."
            )
        destination = Path(destination)
        train_path = Path(train_dir).resolve().as_posix()
        val_path = Path(val_dir or train_dir).resolve().as_posix()
        body = (
            "# Generated by the Detection page's YOLO training dialog.\n"
            f"train: {train_path}\n"
            f"val: {val_path}\n"
            "\n"
            f"nc: {len(names)}\n"
            f"names: {names}\n"
        )
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(body, encoding="utf-8")
        except OSError as exc:
            raise TrainingError(f"Could not write {destination}: {exc}") from exc
        return destination

    # ------------------------------------------------------------ training
    @staticmethod
    def training_interpreter() -> str:
        """The Python that runs the training script.

        ``sys.executable`` when running from source — training then uses the
        very interpreter the station's app runs in, so whatever ultralytics is
        installed there is the one used, with no PATH lookup to get wrong.

        **A frozen build is the exception and the reason this is not just
        ``sys.executable``:** under PyInstaller that path is
        ``WMHoleDetection.exe``, so handing it the runner script would
        relaunch the station instead of training anything. A built station
        falls back to a real Python on PATH, and says so plainly when there
        is none — training belongs on an engineering machine anyway, since
        neither torch nor the dataset lives on the line.

        Raises:
            TrainingError: frozen, with no Python interpreter on PATH.
        """
        if not getattr(sys, "frozen", False):
            return sys.executable
        for candidate in ("python", "python3", "py"):
            found = shutil.which(candidate)
            if found:
                return found
        raise TrainingError(
            "Training needs a Python installation with 'ultralytics' in it, "
            "and this is a built station with no Python on PATH.\n\n"
            "Train on an engineering machine that has the source checkout "
            "(pip install ultralytics), then copy the resulting best.pt here "
            "and point 'Model Weights' at it."
        )

    def build_train_command(
        self,
        data_yaml: Path | str,
        output_dir: Path | str,
        run_name: str = "train",
        model: str = "yolo11n.pt",
        epochs: int = 100,
        imgsz: int = 640,
        device: str = "",
        batch: int = -1,
    ) -> list[str]:
        """The argv for one training run.

        A list, never a shell string: an operator-chosen dataset folder with a
        space or an ``&`` in it is a normal Windows path and must not need
        quoting to survive.

        Raises:
            TrainingError: the runner script is missing, or (frozen build)
                there is no interpreter to run it with.
        """
        runner = self._train_runner
        if not runner.is_file():
            raise TrainingError(f"Training runner not found at {runner.resolve()}")
        return [
            self.training_interpreter(),
            str(runner.resolve()),
            "--data", str(Path(data_yaml).resolve()),
            "--output", str(Path(output_dir).resolve()),
            "--name", run_name,
            "--model", model,
            "--epochs", str(int(epochs)),
            "--imgsz", str(int(imgsz)),
            "--device", device,
            "--batch", str(int(batch)),
        ]

    @staticmethod
    def training_environment() -> dict[str, str]:
        """Environment for the training subprocess.

        ``PYTHONUNBUFFERED`` is the whole point: without it the child buffers
        stdout when it is a pipe, and the dialog's log would sit empty for
        minutes and then dump an entire epoch's output at once.
        """
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    @staticmethod
    def prepare_output_dir(output_dir: Path | str) -> Path:
        """Create the results directory, failing early with a clear message.

        Checked before the run rather than after the first epoch, because an
        unwritable output path otherwise surfaces as an ultralytics traceback
        several minutes in.
        """
        path = Path(output_dir)
        if not str(path).strip():
            raise TrainingError("Choose an output folder for the training results.")
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TrainingError(f"Cannot create output folder {path}: {exc}") from exc
        return path

    @staticmethod
    def ultralytics_installed() -> bool:
        """Whether the training dependency is importable in this interpreter.

        Checked up front so "Start Training" can say what is wrong instead of
        spawning a process that dies on its import line.

        Only meaningful when training runs in *this* interpreter, so a frozen
        build answers True and defers to the runner's own import error — the
        packages it would be asked about live in a different Python entirely
        (see :meth:`training_interpreter`).
        """
        import importlib.util

        if getattr(sys, "frozen", False):
            return True
        return importlib.util.find_spec("ultralytics") is not None

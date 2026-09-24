"""YOLO labelling/training tooling: dataset checks, data.yaml, command, worker.

The training run itself is an ultralytics process this station does not have
installed, so what is pinned here is everything *around* it — the parts that
decide whether a run starts correctly or fails three minutes in with a
traceback: what counts as a labelled dataset, what lands in the generated
data.yaml, and that a path with a space in it survives into the command.

The worker is driven against a real subprocess (a short python -c), because
the thing worth testing is precisely the stream/terminate behaviour a mock
would paper over.
"""

import json
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from core.utilities.exceptions import TrainingError
from services.yolo_training_service import (
    YOLO_MODELS,
    DatasetSummary,
    YoloTrainingService,
)
from workers.yolo_training_worker import BEST_WEIGHTS_PREFIX, YoloTrainingWorker


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def service() -> YoloTrainingService:
    return YoloTrainingService()


def make_dataset(folder: Path, labelled: int = 3, unlabelled: int = 0, empty: int = 0) -> Path:
    """A flat folder of images with their sidecar labels, as YoloLabel writes."""
    folder.mkdir(parents=True, exist_ok=True)
    index = 0
    for _ in range(labelled):
        (folder / f"img_{index}.png").write_bytes(b"\x89PNG")
        (folder / f"img_{index}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        index += 1
    for _ in range(empty):
        (folder / f"img_{index}.png").write_bytes(b"\x89PNG")
        (folder / f"img_{index}.txt").write_text("", encoding="utf-8")
        index += 1
    for _ in range(unlabelled):
        (folder / f"img_{index}.png").write_bytes(b"\x89PNG")
        index += 1
    return folder


# --------------------------------------------------------------- dataset
def test_counts_images_and_their_labels(service, tmp_path) -> None:
    summary = service.inspect_dataset(make_dataset(tmp_path / "ds", labelled=4))
    assert summary == DatasetSummary(images=4, labelled=4, empty_labels=0)
    assert summary.unlabelled == 0


def test_unlabelled_images_are_reported_not_hidden(service, tmp_path) -> None:
    """The commonest way a run silently trains on a third of the data."""
    summary = service.inspect_dataset(make_dataset(tmp_path / "ds", labelled=3, unlabelled=2))
    assert summary.images == 5
    assert summary.labelled == 3
    assert summary.unlabelled == 2
    assert "2 with no label file" in summary.describe()


def test_empty_label_files_are_counted_separately(service, tmp_path) -> None:
    summary = service.inspect_dataset(make_dataset(tmp_path / "ds", labelled=2, empty=1))
    assert summary.labelled == 3  # the file exists...
    assert summary.empty_labels == 1  # ...but has no boxes in it
    assert "1 label file(s) with no boxes" in summary.describe()


def test_classes_txt_is_not_counted_as_an_image(service, tmp_path) -> None:
    folder = make_dataset(tmp_path / "ds", labelled=2)
    (folder / "classes.txt").write_text("hole\n", encoding="utf-8")
    assert service.inspect_dataset(folder).images == 2


@pytest.mark.parametrize("suffix", [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".BMP", ".PNG"])
def test_every_camera_image_format_is_recognised(service, tmp_path, suffix) -> None:
    """The station's Basler cameras save .bmp; the app's own images/ are .png."""
    folder = tmp_path / "ds"
    folder.mkdir()
    (folder / f"frame{suffix}").write_bytes(b"x")
    assert service.inspect_dataset(folder).images == 1


def test_empty_folder_says_so(service, tmp_path) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()
    assert service.inspect_dataset(folder).describe() == "No images found in this folder."


def test_missing_folder_raises(service, tmp_path) -> None:
    with pytest.raises(TrainingError, match="does not exist"):
        service.inspect_dataset(tmp_path / "nope")


# -------------------------------------------------------------- data.yaml
def test_data_yaml_holds_the_paths_and_classes(service, tmp_path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    out = service.write_data_yaml(tmp_path / "out" / "data.yaml", dataset, dataset, ["hole"])
    text = out.read_text(encoding="utf-8")
    assert "nc: 1" in text
    assert "names: ['hole']" in text
    assert dataset.resolve().as_posix() in text


def test_windows_paths_are_written_with_forward_slashes(service, tmp_path) -> None:
    """A backslash in an unquoted YAML scalar is asking for trouble — a
    dataset under \\new or \\test would carry an escape into any reader."""
    dataset = make_dataset(tmp_path / "ds")
    out = service.write_data_yaml(tmp_path / "data.yaml", dataset, dataset, ["hole"])
    body = out.read_text(encoding="utf-8")
    train_line = next(line for line in body.splitlines() if line.startswith("train:"))
    assert "\\" not in train_line


def test_validation_defaults_to_the_training_folder(service, tmp_path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    out = service.write_data_yaml(tmp_path / "data.yaml", dataset, "", ["hole"])
    lines = out.read_text(encoding="utf-8").splitlines()
    train = next(line for line in lines if line.startswith("train:"))
    val = next(line for line in lines if line.startswith("val:"))
    assert train.split(":", 1)[1] == val.split(":", 1)[1]


def test_multiple_classes_are_numbered_in_order(service, tmp_path) -> None:
    dataset = make_dataset(tmp_path / "ds")
    out = service.write_data_yaml(
        tmp_path / "data.yaml", dataset, dataset, ["hole", " burr ", "", "dent"]
    )
    text = out.read_text(encoding="utf-8")
    assert "nc: 3" in text
    assert "names: ['hole', 'burr', 'dent']" in text  # blanks dropped, order kept


def test_no_class_names_raises(service, tmp_path) -> None:
    with pytest.raises(TrainingError, match="class name"):
        service.write_data_yaml(tmp_path / "data.yaml", tmp_path, tmp_path, ["", "  "])


# ---------------------------------------------------------------- command
def test_command_is_argv_not_a_shell_string(service, tmp_path) -> None:
    """A dataset folder with a space in it is an ordinary Windows path and
    must survive without the caller quoting anything."""
    spaced = tmp_path / "My Parts" / "cam 2"
    spaced.mkdir(parents=True)
    yaml_path = service.write_data_yaml(spaced / "data.yaml", spaced, spaced, ["hole"])
    command = service.build_train_command(yaml_path, spaced, model="yolo11s.pt", epochs=7)

    assert command[0] == sys.executable  # same interpreter, not a PATH lookup
    assert "--data" in command
    data_value = command[command.index("--data") + 1]
    assert data_value == str(yaml_path.resolve())
    assert " " in data_value  # the space is there, unescaped, in one argv slot
    assert command[command.index("--model") + 1] == "yolo11s.pt"
    assert command[command.index("--epochs") + 1] == "7"


def test_command_carries_every_training_choice(service, tmp_path) -> None:
    command = service.build_train_command(
        tmp_path / "data.yaml", tmp_path, run_name="cam3",
        model="yolo11m.pt", epochs=50, imgsz=1280, device="cpu",
    )
    pairs = dict(zip(command[2::2], command[3::2]))
    assert pairs["--name"] == "cam3"
    assert pairs["--imgsz"] == "1280"
    assert pairs["--device"] == "cpu"


def test_every_offered_model_is_a_yolo11_weight() -> None:
    assert YOLO_MODELS[0] == "yolo11n.pt"  # smallest first — it is the default
    assert all(name.startswith("yolo11") and name.endswith(".pt") for name in YOLO_MODELS)


def test_missing_runner_raises(tmp_path) -> None:
    service = YoloTrainingService(train_runner=tmp_path / "absent.py")
    with pytest.raises(TrainingError, match="runner not found"):
        service.build_train_command(tmp_path / "data.yaml", tmp_path)


def test_training_environment_unbuffers_the_child(service) -> None:
    """Without this the log sits empty for minutes, then dumps everything."""
    assert service.training_environment()["PYTHONUNBUFFERED"] == "1"


def test_output_dir_is_created_up_front(service, tmp_path) -> None:
    target = tmp_path / "runs" / "2026"
    assert service.prepare_output_dir(target) == target
    assert target.is_dir()


def test_blank_output_dir_raises(service) -> None:
    with pytest.raises(TrainingError, match="output folder"):
        service.prepare_output_dir("   ")


# --------------------------------------------------------- labelling tool
def test_bundled_labelling_tool_is_where_the_service_looks(service) -> None:
    """Pins the path against the folder actually in the tree — note the
    'yolo labling' spelling, which is the directory's real name."""
    assert service.labeling_tool.as_posix().endswith(
        "tools/YOLO Training/yolo labling/YoloLabel.exe"
    )
    assert service.labeling_tool_available(), "YoloLabel.exe missing from the tree"


def test_missing_labelling_tool_explains_the_frozen_build_case(tmp_path) -> None:
    service = YoloTrainingService(labeling_tool=tmp_path / "YoloLabel.exe")
    assert not service.labeling_tool_available()
    with pytest.raises(TrainingError, match="not copied into a built station"):
        service.open_labeling_tool()


def test_labelling_tool_path_is_overridable(tmp_path) -> None:
    """app_config.json can point a relocated station at its own copy."""
    custom = tmp_path / "elsewhere" / "YoloLabel.exe"
    custom.parent.mkdir(parents=True)
    custom.write_bytes(b"MZ")
    assert YoloTrainingService(labeling_tool=custom).labeling_tool_available()


# ----------------------------------------------------------------- worker
def run_worker(qt_app, command, timeout_ms: int = 20000):
    """Drive a worker to completion on a real event loop; collect its output."""
    worker = YoloTrainingWorker(command, environment=YoloTrainingService.training_environment())
    lines: list[str] = []
    result: dict = {}
    loop = QEventLoop()
    worker.output.connect(lines.append)

    def done(code: int, weights: str) -> None:
        result["code"] = code
        result["weights"] = weights
        loop.quit()

    worker.finished_run.connect(done)
    QTimer.singleShot(timeout_ms, loop.quit)
    worker.start()
    loop.exec()
    worker.wait()
    return lines, result


def test_worker_streams_output_and_reports_success(qt_app) -> None:
    lines, result = run_worker(
        qt_app, [sys.executable, "-c", "print('epoch 1/10'); print('epoch 2/10')"]
    )
    assert result["code"] == 0
    assert "epoch 1/10" in lines
    assert "epoch 2/10" in lines


def test_worker_folds_stderr_into_the_log(qt_app) -> None:
    """A traceback must appear in the log where it happened, not vanish."""
    lines, result = run_worker(
        qt_app,
        [sys.executable, "-c", "import sys; sys.stderr.write('ImportError: ultralytics\\n')"],
    )
    assert any("ImportError: ultralytics" in line for line in lines)


def test_worker_reports_a_failing_exit_code(qt_app) -> None:
    _lines, result = run_worker(qt_app, [sys.executable, "-c", "raise SystemExit(2)"])
    assert result["code"] == 2
    assert result["weights"] == ""


def test_worker_extracts_the_trained_weights_path(qt_app, tmp_path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    script = f"print(r'{BEST_WEIGHTS_PREFIX}{weights}')"
    lines, result = run_worker(qt_app, [sys.executable, "-c", script])
    assert result["weights"] == str(weights)
    # the protocol line is consumed, not shown as training output
    assert not any(BEST_WEIGHTS_PREFIX in line for line in lines)


def test_worker_ignores_a_weights_path_that_does_not_exist(qt_app, tmp_path) -> None:
    """Guards against offering the Model Weights field a file that is gone."""
    script = f"print(r'{BEST_WEIGHTS_PREFIX}{tmp_path / 'absent.pt'}')"
    _lines, result = run_worker(qt_app, [sys.executable, "-c", script])
    assert result["code"] == 0
    assert result["weights"] == ""


def test_worker_reports_a_command_that_cannot_start(qt_app) -> None:
    lines, result = run_worker(qt_app, ["definitely-not-a-real-executable-xyz"])
    assert result["code"] == -1
    assert any("Could not start training" in line for line in lines)


def test_worker_stop_ends_a_long_run(qt_app) -> None:
    """Stop has to actually kill it — training sits in native torch calls
    that ignore a plain terminate for a long time."""
    worker = YoloTrainingWorker(
        [sys.executable, "-c", "import time\nwhile True: time.sleep(0.2)"]
    )
    result: dict = {}
    loop = QEventLoop()
    worker.finished_run.connect(lambda code, w: (result.update(code=code), loop.quit()))
    worker.start()
    QTimer.singleShot(700, worker.stop)
    QTimer.singleShot(20000, loop.quit)
    loop.exec()
    worker.wait()
    assert "code" in result, "worker never reported finishing after stop()"
    assert not worker.running


# ----------------------------------------------------------- runner script
def test_runner_reports_a_missing_ultralytics_clearly() -> None:
    """The station has no ultralytics; the runner must say so rather than
    dying on an import traceback."""
    import subprocess

    runner = Path("tools") / "YOLO Training" / "run_training.py"
    proc = subprocess.run(
        [sys.executable, str(runner), "--data", "d.yaml", "--output", "out"],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode == 2:
        assert "pip install ultralytics" in proc.stderr
    else:  # ultralytics is installed here — then it must not be the import
        assert "not installed" not in proc.stderr


def test_runner_rejects_missing_required_arguments() -> None:
    import subprocess

    runner = Path("tools") / "YOLO Training" / "run_training.py"
    proc = subprocess.run(
        [sys.executable, str(runner)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode != 0
    assert "--data" in proc.stderr


# --------------------------------------------------------- config lockstep
def test_shipped_config_carries_every_training_key() -> None:
    """§9's lockstep rule: a key missing from defaults/ means Restore
    Defaults silently drops the feature."""
    live = json.loads(Path("config/app_config.json").read_text(encoding="utf-8"))
    shipped = json.loads(Path("config/defaults/app_config.json").read_text(encoding="utf-8"))
    assert "yolo_training" in live
    assert set(live["yolo_training"]) == set(shipped["yolo_training"])
    assert live["yolo_training"]["model"] in YOLO_MODELS


# ------------------------------------------------- frozen-build interpreter
def test_training_runs_in_this_interpreter_from_source(service) -> None:
    assert service.training_interpreter() == sys.executable


def test_frozen_build_does_not_relaunch_the_station(monkeypatch, service) -> None:
    """Under PyInstaller sys.executable is WMHoleDetection.exe — handing that
    the runner script would start another copy of the app, not train."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\station\WMHoleDetection.exe", raising=False)
    monkeypatch.setattr("services.yolo_training_service.shutil.which",
                        lambda name: r"C:\Python312\python.exe" if name == "python" else None)
    assert service.training_interpreter() == r"C:\Python312\python.exe"


def test_frozen_build_without_python_says_what_to_do(monkeypatch, service) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr("services.yolo_training_service.shutil.which", lambda name: None)
    with pytest.raises(TrainingError, match="engineering machine"):
        service.training_interpreter()


def test_frozen_build_defers_the_ultralytics_check(monkeypatch, service) -> None:
    """The packages live in a different Python, so this interpreter's answer
    is meaningless — the runner reports the real import error instead."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert service.ultralytics_installed() is True


# ------------------------------------------------- runner end-to-end
def fake_ultralytics(tmp_path: Path) -> Path:
    """A stand-in ultralytics package that records how it was called.

    The station has no real one, and installing torch to test argument
    passing would be absurd — but the runner's contract (build a YOLO, call
    .train() with the right overrides, print the best.pt line) is exactly
    what breaks silently, so it is exercised against this instead.
    """
    package = tmp_path / "fakelib" / "ultralytics"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "import json, os, pathlib\n"
        "class _Results:\n"
        "    def __init__(self, save_dir): self.save_dir = save_dir\n"
        "class YOLO:\n"
        "    def __init__(self, weights): self.weights = weights\n"
        "    def train(self, **kw):\n"
        "        save_dir = pathlib.Path(kw['project']) / kw['name']\n"
        "        (save_dir / 'weights').mkdir(parents=True, exist_ok=True)\n"
        "        (save_dir / 'weights' / 'best.pt').write_bytes(b'trained')\n"
        "        record = dict(kw); record['model'] = self.weights\n"
        "        (save_dir / 'call.json').write_text(json.dumps(record))\n"
        "        print('fake training ran')\n"
        "        return _Results(save_dir)\n",
        encoding="utf-8",
    )
    return package.parent


def test_runner_trains_and_announces_its_weights(tmp_path) -> None:
    import subprocess

    service = YoloTrainingService()
    dataset = make_dataset(tmp_path / "ds")
    output = tmp_path / "runs"
    output.mkdir()
    data_yaml = service.write_data_yaml(output / "data.yaml", dataset, dataset, ["hole"])
    command = service.build_train_command(
        data_yaml, output, run_name="cam1", model="yolo11s.pt",
        epochs=3, imgsz=1280, device="cpu",
    )
    env = service.training_environment()
    env["PYTHONPATH"] = str(fake_ultralytics(tmp_path))

    proc = subprocess.run(command, capture_output=True, text=True, env=env, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "fake training ran" in proc.stdout

    best = output / "cam1" / "weights" / "best.pt"
    assert best.is_file()
    assert f"{BEST_WEIGHTS_PREFIX}{best.resolve()}" in proc.stdout

    call = json.loads((output / "cam1" / "call.json").read_text(encoding="utf-8"))
    assert call["model"] == "yolo11s.pt"
    assert call["epochs"] == 3
    assert call["imgsz"] == 1280
    assert call["device"] == "cpu"
    assert Path(call["data"]) == data_yaml.resolve()


def test_runner_omits_device_and_batch_when_unset(tmp_path) -> None:
    """Passing device="" or batch=-1 through would override ultralytics'
    own auto-selection with nonsense."""
    import subprocess

    service = YoloTrainingService()
    dataset = make_dataset(tmp_path / "ds")
    output = tmp_path / "runs"
    output.mkdir()
    data_yaml = service.write_data_yaml(output / "data.yaml", dataset, dataset, ["hole"])
    command = service.build_train_command(data_yaml, output, run_name="auto", device="")
    env = service.training_environment()
    env["PYTHONPATH"] = str(fake_ultralytics(tmp_path))

    proc = subprocess.run(command, capture_output=True, text=True, env=env, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    call = json.loads((output / "auto" / "call.json").read_text(encoding="utf-8"))
    assert "device" not in call
    assert "batch" not in call


# ------------------------------------------------------ class-list pickup
def test_class_names_come_from_the_folders_own_list(service, tmp_path) -> None:
    """The boxes carry only an index, so the list the images were labelled
    with is the only record of what that index means."""
    folder = make_dataset(tmp_path / "ds")
    (folder / "classes.txt").write_text("hole\n", encoding="utf-8")
    assert service.detect_class_names(folder) == ["hole"]


def test_class_list_is_found_whatever_it_is_called(service, tmp_path) -> None:
    """YoloLabel lets the operator name that file anything — this station's
    is 'lables.txt' — so it is found by elimination, not by filename: a .txt
    with no image of the same stem cannot be a label sidecar."""
    folder = make_dataset(tmp_path / "ds")
    (folder / "lables.txt").write_text("Holes\n", encoding="utf-8")
    assert service.detect_class_names(folder) == ["Holes"]


def test_multiple_classes_keep_their_labelling_order(service, tmp_path) -> None:
    """Order is the index mapping — sorting it would mislabel every box."""
    folder = make_dataset(tmp_path / "ds")
    (folder / "obj.names").write_text("hole\nburr\ndent\n", encoding="utf-8")
    assert service.detect_class_names(folder) == ["hole", "burr", "dent"]


def test_a_conventional_name_wins_over_another_orphan(service, tmp_path) -> None:
    folder = make_dataset(tmp_path / "ds")
    (folder / "classes.txt").write_text("hole\n", encoding="utf-8")
    (folder / "notes.txt").write_text("scratch pad\n", encoding="utf-8")
    assert service.detect_class_names(folder) == ["hole"]


def test_ambiguous_orphans_are_not_guessed_at(service, tmp_path) -> None:
    folder = make_dataset(tmp_path / "ds")
    (folder / "one.txt").write_text("hole\n", encoding="utf-8")
    (folder / "two.txt").write_text("burr\n", encoding="utf-8")
    assert service.detect_class_names(folder) is None


def test_label_sidecars_are_never_mistaken_for_a_class_list(service, tmp_path) -> None:
    folder = make_dataset(tmp_path / "ds", labelled=3)
    assert service.detect_class_names(folder) is None  # every .txt has an image


def test_a_file_of_boxes_is_rejected_as_a_class_list(service, tmp_path) -> None:
    """An orphaned label file (its image was deleted) must not become the
    class list — 'nc: 1, names: [0 0.5 0.5 0.1 0.1]' would be nonsense."""
    folder = make_dataset(tmp_path / "ds")
    (folder / "orphaned.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    assert service.detect_class_names(folder) is None


def test_no_class_list_is_not_an_error(service, tmp_path) -> None:
    assert service.detect_class_names(tmp_path / "absent") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert service.detect_class_names(empty) is None


def test_blank_lines_in_the_class_list_are_dropped(service, tmp_path) -> None:
    folder = make_dataset(tmp_path / "ds")
    (folder / "classes.txt").write_text("hole\n\n  \nburr\n", encoding="utf-8")
    assert service.detect_class_names(folder) == ["hole", "burr"]


def test_persisting_the_dialog_keeps_keys_it_has_no_field_for() -> None:
    """The page merges into the block rather than assigning over it:
    'labeling_tool' is read only by main.py and has no widget, so a replace
    would delete a relocated station's tool path on first use."""
    from ui.detection.detection_page import _TRAINING_SETTINGS_KEY

    live = json.loads(Path("config/app_config.json").read_text(encoding="utf-8"))
    assert _TRAINING_SETTINGS_KEY == "yolo_training"
    assert "labeling_tool" in live["yolo_training"]

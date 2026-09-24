"""Off-thread YOLO training run for the Detection page's training dialog.

A training run is minutes to hours of work that prints continuously. Driving
it from the GUI thread would freeze the window for the whole run, so this
worker owns the subprocess and streams its output back as queued signals —
the same reasoning as :mod:`workers.checkerboard_worker`, just at a much
longer timescale.

Thread ownership only, no business logic: the command, the environment and
the "which file did it produce" convention all come from
:class:`~services.yolo_training_service.YoloTrainingService`, which this
worker never imports (``workers/`` sits below ``services/``) — the page hands
it a ready-made argv.

Stopping is a two-step: ``stop()`` asks the child to terminate and, if it is
still alive after :data:`KILL_GRACE_SECONDS`, kills it. Training spends most
of its time inside a native ``torch`` call that does not check for a Python
signal, so a terminate alone can be ignored for a long time, and leaving an
orphaned run holding the GPU is worse than a hard kill.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from core.logging import get_logger
from core.utilities.enums import LogSource

logger = get_logger(LogSource.VISION)

#: How long a terminate() is given before the child is killed outright.
KILL_GRACE_SECONDS = 10

#: Line prefix the runner prints to name the weights it produced.
BEST_WEIGHTS_PREFIX = "WMHD_BEST_WEIGHTS="


class YoloTrainingWorker(QThread):
    """Runs one training subprocess, streaming its stdout line by line."""

    #: one line of training output, already stripped of its newline
    output = Signal(str)
    #: (exit_code, best_weights_path) — the path is "" when none was produced
    finished_run = Signal(int, str)

    def __init__(
        self,
        command: list[str],
        environment: dict[str, str] | None = None,
        working_dir: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._command = list(command)
        self._environment = environment
        self._working_dir = working_dir
        self._process: subprocess.Popen[str] | None = None
        self._stopping = False

    # ---------------------------------------------------------------- run
    def run(self) -> None:  # noqa: D102 - QThread entry point
        best_weights = ""
        try:
            # stderr folded into stdout so a traceback appears in the log in
            # the right place, rather than arriving separately at the end.
            self._process = subprocess.Popen(
                self._command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                env=self._environment,
                cwd=self._working_dir,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            self.output.emit(f"Could not start training: {exc}")
            self.finished_run.emit(-1, "")
            return

        try:
            assert self._process.stdout is not None
            for line in self._process.stdout:
                line = line.rstrip("\r\n")
                if line.startswith(BEST_WEIGHTS_PREFIX):
                    best_weights = line[len(BEST_WEIGHTS_PREFIX):].strip()
                    continue  # a protocol line, not something to show
                self.output.emit(line)
        except (OSError, ValueError) as exc:
            # ValueError: the pipe was closed under us by stop()
            if not self._stopping:
                self.output.emit(f"Training output stream ended: {exc}")
        finally:
            code = self._process.wait()
            self._process = None

        if self._stopping:
            self.output.emit("Training stopped.")
        logger.info("Training run finished with exit code %s", code)
        self.finished_run.emit(code, best_weights if Path(best_weights).is_file() else "")

    # --------------------------------------------------------------- stop
    def stop(self) -> None:
        """Ask the training process to end, killing it if it will not.

        Safe to call from the GUI thread and safe to call twice; a run that
        has already finished is a no-op.
        """
        self._stopping = True
        process = self._process
        if process is None:
            return
        try:
            process.terminate()
            try:
                process.wait(timeout=KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning("Training process ignored terminate; killing it")
                process.kill()
        except OSError as exc:  # already gone between the check and the call
            logger.debug("Training process could not be signalled: %s", exc)

    @property
    def running(self) -> bool:
        return self._process is not None

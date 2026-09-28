"""Off-thread dispatch for the LED Controller page's raw command tester.

Every other LED command in this application is fire-and-forget (see
``core.led.led_manager``): the bytes are written synchronously and nothing
waits for the controller's "!" acknowledgement, so a mute controller costs
a capture or an inspection cycle nothing. The raw command tester is the one
deliberate exception - the operator is asking the hardware a question and
the answer is the whole point, so it *must* wait.

Waiting on the GUI thread is what this worker exists to prevent. The wait
is bounded by ``connection.timeout_ms`` from ``led.json``, but that is an
operator-tunable value with no upper bound worth relying on, and a frozen
window during commissioning is exactly when the operator is most likely to
assume the station has crashed. One command, one thread, one signal back.

Page-owned rather than built in the composition root, the same way
``CheckerboardScanWorker`` belongs to the Calibration page: it exists only
for the length of one button press.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from core.logging import get_logger
from core.utilities.enums import LogSource
from core.utilities.exceptions import VisionSystemError

logger = get_logger(LogSource.LED)


class LedCommandWorker(QThread):
    """Sends one raw command on its own thread and reports the outcome.

    Exactly one of the two signals is emitted, always, so the page can
    re-enable its controls in either branch without a third "finished"
    signal to forget about.
    """

    #: The controller's reply, as received (never fabricated).
    succeeded = Signal(str)
    #: A ``VisionSystemError`` message - a timeout, or a broken link.
    failed = Signal(str)

    def __init__(self, send, command: str, *, append_terminator: bool, parent=None) -> None:
        """*send* is a plain callable (``LedService.send_raw``), not the
        service itself, so this worker keeps the dependency direction of
        ``workers/`` pointing downward and stays trivially fakeable."""
        super().__init__(parent)
        self._send = send
        self._command = command
        self._append_terminator = append_terminator

    def run(self) -> None:  # noqa: D102 - see class docstring
        try:
            response = self._send(self._command, append_terminator=self._append_terminator)
        except VisionSystemError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # a worker must never take the app down
            logger.exception("Unexpected error sending raw LED command")
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(response)

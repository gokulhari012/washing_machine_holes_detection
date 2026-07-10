"""Central logging configuration for the application.

Design
------
All application loggers live under the ``wmhd`` namespace, one child per
subsystem (``wmhd.plc``, ``wmhd.camera`` ...) obtained via :func:`get_logger`.
:class:`LogManager` configures the ``wmhd`` root once at startup with:

- a size-rotating file handler (``logs/application.log``),
- a console handler (useful when run from a terminal),
- optional pluggable handlers added later by other layers, e.g. the database
  log handler (Logs table) and the Qt bridge feeding the live Logs page —
  both built on :class:`CallbackLogHandler` so this module stays Qt-free.
"""

from __future__ import annotations

import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable

from core.utilities.enums import LogSource

ROOT_LOGGER_NAME = "wmhd"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-14s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(source: LogSource | str) -> logging.Logger:
    """Return the subsystem logger for *source* (e.g. ``wmhd.plc``).

    Accepts a :class:`LogSource` or a plain string for ad-hoc names.
    """
    name = source.value if isinstance(source, LogSource) else str(source)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name.lower()}")


class CallbackLogHandler(logging.Handler):
    """Fans every log record out to registered callables.

    Used by the database writer (persist to the ``logs`` table) and by the UI
    bridge (live Logs page) without this core module depending on either.
    Callbacks run on the emitting thread and must be fast and thread-safe;
    a raising callback is isolated so it can never break logging.
    """

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self._callbacks: list[Callable[[logging.LogRecord], None]] = []
        self._callback_lock = threading.Lock()

    def add_callback(self, callback: Callable[[logging.LogRecord], None]) -> None:
        with self._callback_lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[logging.LogRecord], None]) -> None:
        with self._callback_lock:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

    def emit(self, record: logging.LogRecord) -> None:
        with self._callback_lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(record)
            except Exception:
                self.handleError(record)


class LogManager:
    """Configures the application logger tree. Instantiate once in ``main.py``."""

    def __init__(
        self,
        log_dir: str | Path,
        level: str | int = logging.INFO,
        max_file_size_mb: int = 5,
        backup_count: int = 10,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._level = logging.getLevelNamesMapping().get(level, level) if isinstance(level, str) else level
        self._max_bytes = max_file_size_mb * 1024 * 1024
        self._backup_count = backup_count
        self._root = logging.getLogger(ROOT_LOGGER_NAME)
        self.fanout = CallbackLogHandler()

    @property
    def log_file(self) -> Path:
        return self._log_dir / "application.log"

    def setup(self) -> None:
        """Install handlers. Idempotent — safe to call again after config changes."""
        self._log_dir.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self._max_bytes,
            backupCount=self._backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self._root.setLevel(self._level)
        self._root.propagate = False  # keep third-party root logger untouched
        for handler in list(self._root.handlers):  # idempotency: drop old handlers
            self._root.removeHandler(handler)
            handler.close()

        self._root.addHandler(file_handler)
        self._root.addHandler(console_handler)
        self._root.addHandler(self.fanout)

        get_logger(LogSource.SYSTEM).info(
            "Logging initialised (level=%s, file=%s)",
            logging.getLevelName(self._level),
            self.log_file,
        )

    def set_level(self, level: str | int) -> None:
        """Change the runtime log level (Settings page)."""
        self._level = logging.getLevelNamesMapping().get(level, level) if isinstance(level, str) else level
        self._root.setLevel(self._level)

    def shutdown(self) -> None:
        """Flush and close all handlers (called from MainWindow.closeEvent)."""
        for handler in list(self._root.handlers):
            self._root.removeHandler(handler)
            handler.close()

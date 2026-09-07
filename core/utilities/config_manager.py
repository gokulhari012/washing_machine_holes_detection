"""Thread-safe manager for the JSON configuration files in ``config/``.

Responsibilities
----------------
- Load/save the four configuration domains (``app_config``, ``plc``,
  ``camera``, ``detection``) with caching and deep-copy isolation.
- Atomic saves (write to a temp file, then ``os.replace``) so a crash or
  power loss can never leave a half-written config on disk.
- Dot-path access helpers (``get_value("plc", "connection.ip")``).
- "Restore defaults" from the pristine copies shipped in ``config/defaults/``.
- Change notification via plain callables (this module is Qt-free; the UI
  layer wraps subscriptions in signals where needed).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

from core.utilities.exceptions import ConfigurationError

logger = logging.getLogger("wmhd.system")

ConfigCallback = Callable[[dict[str, Any]], None]


class ConfigManager:
    """Loads, caches, saves and validates the application's JSON config files."""

    DEFAULTS_SUBDIR = "defaults"
    KNOWN_CONFIGS = ("app_config", "plc", "camera", "detection", "machine_models")

    def __init__(self, config_dir: str | Path) -> None:
        self._config_dir = Path(config_dir)
        self._lock = threading.RLock()
        self._cache: dict[str, dict[str, Any]] = {}
        self._subscribers: dict[str, list[ConfigCallback]] = {}

        if not self._config_dir.is_dir():
            raise ConfigurationError(f"Config directory not found: {self._config_dir}")

    # ------------------------------------------------------------------ paths
    def path_for(self, name: str) -> Path:
        """Absolute path of the JSON file backing configuration *name*."""
        return self._config_dir / f"{name}.json"

    def defaults_path_for(self, name: str) -> Path:
        """Absolute path of the shipped pristine copy used by restore-defaults."""
        return self._config_dir / self.DEFAULTS_SUBDIR / f"{name}.json"

    # ------------------------------------------------------------------- load
    def load(self, name: str, *, force_reload: bool = False) -> dict[str, Any]:
        """Return configuration *name* as a dict (deep copy — safe to mutate).

        Results are cached; pass ``force_reload=True`` to re-read from disk.

        Raises:
            ConfigurationError: file missing or invalid JSON.
        """
        with self._lock:
            if not force_reload and name in self._cache:
                return copy.deepcopy(self._cache[name])

            path = self.path_for(name)
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except FileNotFoundError as exc:
                raise ConfigurationError(f"Configuration file not found: {path}") from exc
            except json.JSONDecodeError as exc:
                raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc

            if not isinstance(data, dict):
                raise ConfigurationError(f"{path} must contain a JSON object at top level")

            self._cache[name] = data
            logger.debug("Loaded configuration '%s' from %s", name, path)
            return copy.deepcopy(data)

    # ------------------------------------------------------------------- save
    def save(self, name: str, data: dict[str, Any]) -> None:
        """Atomically persist *data* as configuration *name* and notify subscribers.

        Raises:
            ConfigurationError: the file could not be written.
        """
        path = self.path_for(name)
        tmp_path = path.with_suffix(".json.tmp")
        with self._lock:
            try:
                with tmp_path.open("w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)
                    fh.write("\n")
                os.replace(tmp_path, path)  # atomic on Windows and POSIX
            except OSError as exc:
                tmp_path.unlink(missing_ok=True)
                raise ConfigurationError(f"Failed to save {path}: {exc}") from exc

            self._cache[name] = copy.deepcopy(data)
            callbacks = list(self._subscribers.get(name, ()))

        logger.info("Configuration '%s' saved", name)
        self._notify(name, callbacks, data)

    def load_defaults(self, name: str) -> dict[str, Any]:
        """Read configuration *name*'s shipped default without touching the
        live file — unlike :meth:`restore_defaults`, this never saves or
        notifies subscribers. Use it to peek at (or selectively merge from)
        the pristine copy, e.g. resetting one sub-section of a config domain.

        Raises:
            ConfigurationError: no defaults file exists for *name*, or it's invalid.
        """
        defaults_path = self.defaults_path_for(name)
        try:
            with defaults_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError as exc:
            raise ConfigurationError(f"No defaults shipped for '{name}' ({defaults_path})") from exc
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON in defaults {defaults_path}: {exc}") from exc

    def restore_defaults(self, name: str) -> dict[str, Any]:
        """Overwrite configuration *name* with its shipped default and return it.

        Raises:
            ConfigurationError: no defaults file exists for *name*.
        """
        data = self.load_defaults(name)
        self.save(name, data)
        logger.info("Configuration '%s' restored to defaults", name)
        return copy.deepcopy(data)

    # -------------------------------------------------------------- accessors
    def get_value(self, name: str, key_path: str, default: Any = None) -> Any:
        """Read a nested value with a dot path, e.g. ``get_value("plc", "connection.ip")``.

        Returns *default* if any segment of the path is missing.
        """
        node: Any = self.load(name)
        for key in key_path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set_value(self, name: str, key_path: str, value: Any) -> None:
        """Write a nested value with a dot path and persist immediately.

        Intermediate objects are created as needed.
        """
        data = self.load(name)
        node = data
        keys = key_path.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ConfigurationError(
                    f"Cannot set '{key_path}' in '{name}': '{key}' is not an object"
                )
        node[keys[-1]] = value
        self.save(name, data)

    # ------------------------------------------------------------ subscribers
    def subscribe(self, name: str, callback: ConfigCallback) -> None:
        """Register *callback* to be invoked (with the new dict) after each save of *name*.

        Callbacks run on the saving thread; keep them short and thread-safe.
        """
        with self._lock:
            self._subscribers.setdefault(name, []).append(callback)

    def unsubscribe(self, name: str, callback: ConfigCallback) -> None:
        """Remove a previously registered callback (no-op if absent)."""
        with self._lock:
            try:
                self._subscribers.get(name, []).remove(callback)
            except ValueError:
                pass

    @staticmethod
    def _notify(name: str, callbacks: list[ConfigCallback], data: dict[str, Any]) -> None:
        for callback in callbacks:
            try:
                callback(copy.deepcopy(data))
            except Exception:  # a bad subscriber must never break a save
                logger.exception("Config subscriber for '%s' raised", name)

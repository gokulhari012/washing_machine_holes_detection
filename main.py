"""Washing Machine Bottom Hole Detection System — application entry point.

Composition root: builds the object graph (config → logging → database →
hardware managers → services → workers → UI), wires cross-cutting concerns
(global exception hook, config-change subscriptions, maintenance timer) and
owns the ordered shutdown sequence. No other module constructs dependencies.

Usage:
    python main.py                # normal start
    python main.py --selftest 8   # smoke test: run hidden for 8 s, save a
                                  # screenshot to logs/selftest.png, exit 0
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import threading
from pathlib import Path

# Frozen (PyInstaller) builds run from a temp extraction dir, so anchor on the
# executable instead: config/ and the runtime artifacts sit beside the .exe.
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)  # runtime artifacts (data/, logs/, images/, backups/) live beside the app
sys.path.insert(0, str(BASE_DIR))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from core.calibration import CalibrationManager
from core.camera import CameraManager
from core.database import DatabaseEngine
from core.logging import LogManager, get_logger
from core.plc import PlcManager, RegisterMap, create_plc_client
from core.utilities import ConfigManager
from core.utilities.enums import LogSource
from core.utilities.exceptions import VisionSystemError
from core.vision import VisionEngine, migrate_legacy_detection_config
from models import AppState
from services import (
    AuthService,
    BackupService,
    CameraService,
    DatabaseService,
    ExportService,
    InspectionService,
    MachineModelService,
    PlcService,
)
from workers import (
    DatabaseWorker,
    InspectionWorker,
    PlcPollWorker,
    create_acquisition_workers,
)
from ui.dashboard import DashboardPage
from ui.camera import CameraPage 
from ui.plc import PlcPage
from ui.detection import DetectionPage
from ui.calibration import CalibrationPage
from ui.database import DatabasePage
from ui.logs import LogsPage
from ui.machine_models import MachineModelsPage
from ui.settings import SettingsPage
from ui.main_window import MainWindow
from ui.theme import apply_dark_theme

MAINTENANCE_INTERVAL_MS = 30 * 60 * 1000  # backup/retention check cadence


class Application:
    """Owns every long-lived object and the startup/shutdown order."""

    def __init__(self) -> None:
        # ------------------------------------------------ config & logging
        self.config = ConfigManager(BASE_DIR / "config")
        app_cfg = self.config.load("app_config")
        log_cfg = app_cfg.get("logging", {})
        self.log_manager = LogManager(
            log_cfg.get("directory", "logs"),
            level=log_cfg.get("level", "INFO"),
            max_file_size_mb=int(log_cfg.get("max_file_size_mb", 5)),
            backup_count=int(log_cfg.get("backup_count", 10)),
        )
        self.log_manager.setup()
        self.logger = get_logger(LogSource.SYSTEM)
        self._install_excepthook()

        # ---------------------------------------------------- persistence
        self.db = DatabaseEngine(app_cfg.get("database", {}).get("path", "data/inspection.db"))
        self.db.create_schema()
        self.database = DatabaseService(self.db)

        # ------------------------------------------------------ app state
        self.app_state = AppState()
        self.db_worker = DatabaseWorker(self.database, self.app_state)
        self.log_manager.fanout.add_callback(self.db_worker.enqueue_log_record)

        # ------------------------------------------------------- hardware
        self.camera_configs = self.config.load("camera").get("cameras", [])
        self.cameras = CameraManager(self.camera_configs)
        self.cameras.subscribe_state(
            lambda index, state: self.app_state.update_camera_state(index, state)
        )
        detection_doc = self.config.load("detection")
        camera_indices = [int(cfg["index"]) for cfg in self.camera_configs]
        migrated_detection = migrate_legacy_detection_config(detection_doc, camera_indices)
        if migrated_detection is not detection_doc:
            self.config.save("detection", migrated_detection)
        self.vision = VisionEngine(migrated_detection)
        self.calibration = CalibrationManager(self.database.calibrations)
        self.calibration.load_all()

        plc_cfg = self.config.load("plc")
        connection_cfg = plc_cfg.get("connection", {})
        self.register_map = RegisterMap.from_config(plc_cfg)
        plc_client = create_plc_client(plc_cfg, self.register_map)
        self.plc = PlcManager(
            plc_client,
            self.register_map,
            connection_cfg.get("reconnect_backoff_ms"),
        )
        self.plc.subscribe_state(self.app_state.update_plc_state)

        # ------------------------------------------------------- services
        self.inspection = InspectionService(
            self.cameras, self.vision, self.calibration,
            self.plc, self.database, self.app_state, self.config,
        )
        self.plc_service = PlcService(self.plc, self.config, self.database)
        self.camera_service = CameraService(
            self.cameras, self.config, self.database, self.plc_service
        )
        self.machine_models = MachineModelService(
            self.config, self.camera_service, self.vision, self.plc_service, self.calibration
        )
        self.export_service = ExportService()
        self.backup_service = BackupService(self.db, self.database, self.config)
        self.auth_service = AuthService(self.database)
        self.auth_service.ensure_default_admin()

        # -------------------------------------------------------- workers
        self.inspection_worker = InspectionWorker(self.inspection)
        self.poll_worker = self._build_poll_worker(connection_cfg)
        preview_fps = float(self.config.get_value("app_config", "ui.live_preview_fps", 15))
        self.acquisition_workers = create_acquisition_workers(
            self.cameras, self.app_state, preview_fps
        )

        # ------------------------------------------------------------- UI
        self._manual_machine = itertools.count(9001)
        self.window = MainWindow(
            self.app_state,
            self.auth_service,
            factory_name=str(app_cfg.get("application", {}).get("factory_name", "")),
            camera_indexes=[int(cfg["index"]) for cfg in self.camera_configs],
            on_simulate_trigger=self._simulate_trigger,
            on_shutdown=self.shutdown,
        )
        # Dashboard/Database/Logs are what an operator needs day to day and
        # stay visible always; the engineering consoles below (hardware
        # tuning, PLC register map, detection algorithm parameters,
        # calibration) are hidden from the nav rail until an administrator
        # logs in via the toolbar — see MainWindow._refresh_nav_visibility.
        self.window.add_page(
            "Dashboard",
            "▦",
            DashboardPage(
                self.app_state,
                self.database,
                self.camera_configs,
                self.plc_service,
                self.auth_service,
                self.machine_models,
                config_manager=self.config,
                on_simulate_trigger=self._simulate_trigger,
                on_camera_trigger=self._trigger_camera,
            ),
        )
        self.window.add_page(
            "Cameras", "◉",
            CameraPage(
                self.app_state, self.camera_service, self.plc_service,
                self.machine_models, self.auth_service,
            ),
            admin_only=True,
        )
        self.window.add_page(
            "PLC", "⇄", PlcPage(self.app_state, self.plc_service, self.auth_service), admin_only=True
        )
        self.window.add_page(
            "Detection", "◎", DetectionPage(self.config, self.vision, self.camera_service),
            admin_only=True,
        )
        self.window.add_page(
            "Calibration", "⌖", CalibrationPage(self.camera_service, self.calibration, self.vision),
            admin_only=True,
        )
        self.window.add_page(
            "Machine Models", "▣",
            MachineModelsPage(self.machine_models, self.auth_service, self.app_state),
            admin_only=True,
        )
        self.window.add_page("Database", "▤", DatabasePage(self.database, self.export_service))
        self.window.add_page("Logs", "≡", LogsPage(self.app_state, self.database))
        self.window.add_page(
            "Settings", "⚙", SettingsPage(self.config, self.auth_service, self.backup_service),
            admin_only=True,
        )

        # -------------------------------------------------- cross-cutting
        self.config.subscribe("camera", self._on_camera_config_saved)
        self.config.subscribe("plc", self._on_plc_config_saved)
        self._maintenance_timer = QTimer(self.window)
        self._maintenance_timer.timeout.connect(self._run_maintenance_async)
        self._shutdown_done = False

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.window.showMaximized()

        errors = self.cameras.connect_all()
        for index, message in errors.items():
            self.app_state.raise_alarm("warning", f"Camera {index}: {message}")

        counts = self.database.daily_counts()
        self.app_state.set_counters(counts.total, counts.good, counts.ng)

        self.db_worker.start()
        self.inspection_worker.start()
        self.poll_worker.start()  # connects to the PLC via ensure_connected()
        for worker in self.acquisition_workers:
            worker.start()

        self._maintenance_timer.start(MAINTENANCE_INTERVAL_MS)
        self._run_maintenance_async()
        self.logger.info("Application started")

    def shutdown(self) -> None:
        """Ordered stop; idempotent (called from closeEvent and from main())."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self.logger.info("Shutting down")
        self._maintenance_timer.stop()
        self.poll_worker.stop()                       # no new triggers
        for worker in self.acquisition_workers:       # stop preview grabs
            worker.stop()
        self.inspection_worker.stop()                 # let in-flight cycle finish
        self.plc.disconnect()
        self.db_worker.stop()                         # final log flush
        self.db.dispose()
        self.log_manager.shutdown()

    # ------------------------------------------------------------- helpers
    def _simulate_trigger(self) -> None:
        self.inspection_worker.trigger_requested.emit(next(self._manual_machine))

    def _camera_availability(self) -> dict[int, bool]:
        """Per-camera usability for the PLC status registers, read by the poll
        worker on its own thread.

        Uses ``CameraHealth.healthy`` rather than the raw connection flag: a
        camera whose link is open but whose grabs are failing is just as
        unusable to the line, and reporting it as available would let the PLC
        keep running the station against a camera that cannot answer.
        """
        return {
            index: health.healthy for index, health in self.cameras.all_health().items()
        }

    def _trigger_camera(self, camera_index: int) -> None:
        """Dashboard per-camera Trigger button — inspect that camera alone."""
        self.inspection_worker.camera_trigger_requested.emit(
            camera_index, next(self._manual_machine)
        )

    def _on_camera_config_saved(self, camera_cfg: dict) -> None:
        """Rebuild the camera manager + acquisition workers after a save
        (runs on the UI thread — configuration saves originate from pages)."""
        self.logger.info("Camera configuration changed — rebuilding acquisition")
        for worker in self.acquisition_workers:
            worker.stop()
        self.cameras.rebuild(camera_cfg.get("cameras", []))
        errors = self.cameras.connect_all()
        for index, message in errors.items():
            self.app_state.raise_alarm("warning", f"Camera {index}: {message}")
        preview_fps = float(self.config.get_value("app_config", "ui.live_preview_fps", 15))
        self.acquisition_workers = create_acquisition_workers(
            self.cameras, self.app_state, preview_fps
        )
        for worker in self.acquisition_workers:
            worker.start()

    def _build_poll_worker(self, connection_cfg: dict) -> PlcPollWorker:
        """Construct + wire a poll worker against ``self.plc``. Shared by
        startup and :meth:`_on_plc_config_saved` so both build it identically —
        cadence intervals and per-camera-trigger edge state are baked into
        the worker at construction, so a config change needs a fresh
        instance, not just a fresh client on the existing one."""
        worker = PlcPollWorker(
            self.plc,
            poll_interval_ms=int(connection_cfg.get("poll_interval_ms", 50)),
            heartbeat_interval_ms=int(connection_cfg.get("heartbeat_interval_ms", 500)),
            model_poll_interval_ms=int(connection_cfg.get("model_poll_interval_ms", 1000)),
            camera_status_provider=self._camera_availability,
        )
        worker.trigger_detected.connect(self.inspection_worker.on_trigger)
        worker.camera_trigger_detected.connect(self.inspection_worker.on_camera_trigger)
        worker.machine_model_changed.connect(self._on_machine_model_changed)
        return worker

    def _on_plc_config_saved(self, plc_cfg: dict) -> None:
        """Rebuild the PLC client/register map + poll worker after a save
        (runs on the UI thread — configuration saves originate from pages).

        Stops the current poll worker first (blocking, but bounded to one
        poll tick) so nothing touches the outgoing client while it's being
        swapped, rebuilds ``PlcManager`` in place (see ``PlcManager.rebuild``
        — every other holder of it keeps the same instance), then starts a
        fresh poll worker sized to the new poll/heartbeat/model intervals.
        The new worker's own re-baselining logic (see
        ``workers.plc_poll_worker``) handles a changed register map the same
        way it already handles a real reconnect.
        """
        self.logger.info("PLC configuration changed — rebuilding connection")
        self.poll_worker.stop()
        try:
            register_map = RegisterMap.from_config(plc_cfg)
            client = create_plc_client(plc_cfg, register_map)
        except VisionSystemError as exc:
            message = f"PLC configuration rebuild failed: {exc}"
            self.logger.error(message)
            self.app_state.raise_alarm("error", message)
            return
        connection_cfg = plc_cfg.get("connection", {})
        self.plc.rebuild(client, register_map, connection_cfg.get("reconnect_backoff_ms"))
        self.register_map = register_map
        self.poll_worker = self._build_poll_worker(connection_cfg)
        self.poll_worker.start()

    def _on_machine_model_changed(self, code: int) -> None:
        """PLC reported a new model_select value (runs on the UI thread via
        the queued Qt signal from PlcPollWorker's thread)."""
        profile = self.machine_models.get_by_code(code)
        if profile is None:
            message = f"No machine model profile registered for PLC code {code}"
            self.logger.warning(message)
            self.app_state.raise_alarm("warning", message)
            return
        try:
            warnings = self.machine_models.apply_profile(profile)
        except VisionSystemError as exc:
            message = f"Machine model {profile['name']!r} (code {code}) failed to apply: {exc}"
            self.logger.error(message)
            self.app_state.raise_alarm("error", message)
            return
        for warning in warnings:
            self.logger.warning("Machine model %r: %s", profile["name"], warning)
        self.app_state.set_active_machine_model(profile["name"], code)
        self.logger.info("Machine model -> %r (code %d)", profile["name"], code)

    def _run_maintenance_async(self) -> None:
        threading.Thread(
            target=self.backup_service.run_maintenance,
            name="Maintenance",
            daemon=True,
        ).start()

    def _install_excepthook(self) -> None:
        def hook(exc_type, exc_value, exc_tb) -> None:
            self.logger.critical(
                "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb)
            )
            try:  # surface it without a modal that could deadlock shutdown
                self.app_state.raise_alarm("error", f"Internal error: {exc_value}")
            except Exception:
                pass

        sys.excepthook = hook


def main() -> int:
    parser = argparse.ArgumentParser(description="Washing Machine Bottom Hole Detection")
    parser.add_argument(
        "--selftest",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="run hidden for N seconds, save logs/selftest.png, then exit",
    )
    args = parser.parse_args()

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("WM Hole Detection")
    apply_dark_theme(qt_app)

    application = Application()
    if args.selftest > 0:
        application.window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    application.start()

    if args.selftest > 0:
        def _finish_selftest() -> None:
            application.window.grab().save(str(BASE_DIR / "logs" / "selftest.png"))
            application.window.close()

        QTimer.singleShot(int(args.selftest * 1000), _finish_selftest)

    exit_code = qt_app.exec()
    application.shutdown()  # no-op if closeEvent already ran it
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

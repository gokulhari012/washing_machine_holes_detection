"""Dashboard: manual trigger bar, 2×2 camera grid, summary column.

Entirely event-driven off :class:`AppState` signals — the page performs no
polling, touches no hardware and reads no database; the counters it starts
from come from :class:`AppState`'s snapshot. Inspection history lives on the
Database page.

The trigger bar drives the manual cycle: "Simulate Trigger" runs one complete
inspection, and the delay spin box sets how long the pipeline waits between
cameras when the sequential capture mode is on. The delay is persisted to
app_config.json immediately, so the next cycle — and the next start of the
application — uses it.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.logging import get_logger
from core.utilities import ConfigManager
from core.utilities.enums import LogSource
from core.utilities.exceptions import ConfigurationError, VisionSystemError
from models.app_state import AppState
from models.dto import CameraInspectionData, InspectionCycleData
from services.auth_service import AuthService
from services.inspection_service import DEFAULT_CAMERA_DELAY_MS, SEQUENTIAL_MODE
from services.machine_model_service import MachineModelService
from services.plc_service import PlcService
from ui.dashboard.camera_panel import CameraPanel
from ui.dashboard.summary_panels import CameraCoordinatesPanel, CycleSummaryPanel

logger = get_logger(LogSource.UI)

MAX_CAMERA_DELAY_MS = 60000

class DashboardPage(QWidget):
    """The operator's main screen."""

    def __init__(
        self,
        app_state: AppState,
        camera_configs: list[dict],
        plc_service: PlcService,
        auth_service: AuthService,
        machine_model_service: MachineModelService,
        config_manager: ConfigManager | None = None,
        on_simulate_trigger=None,
        on_camera_trigger=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._app_state = app_state
        self._plc = plc_service
        self._auth = auth_service
        self._machine_models = machine_model_service
        self._config = config_manager
        self._on_simulate_trigger = on_simulate_trigger
        self._on_camera_trigger = on_camera_trigger
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(10)

        # ------------------------------------------------------ trigger bar
        header = QHBoxLayout()
        title = QLabel("Dashboard")
        title.setProperty("class", "pageTitle")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self._build_trigger_controls())
        root.addLayout(header)

        # --------------------- 2×2 camera grid + summary cards on the right
        content = QHBoxLayout()
        content.setSpacing(10)

        grid = QGridLayout()
        grid.setSpacing(10)
        self._panels: dict[int, CameraPanel] = {}
        camera_names: list[tuple[int, str]] = []
        enabled = [cfg for cfg in camera_configs if cfg.get("enabled", True)][:4]
        for position, cfg in enumerate(sorted(enabled, key=lambda c: int(c["index"]))):
            index = int(cfg["index"])
            name = str(cfg.get("name", f"Camera {cfg['index']}"))
            camera_names.append((index, name))
            panel = CameraPanel(index, name)
            panel.set_home_enabled(self._plc.jog_configured(index))
            panel.set_trigger_enabled(on_camera_trigger is not None)
            panel.home_requested.connect(self._on_home_requested)
            panel.trigger_requested.connect(self._on_camera_trigger_clicked)
            self._panels[index] = panel
            grid.addWidget(panel, position // 2, position % 2)
        content.addLayout(grid, stretch=4)

        # The right-hand column: cycle summary on top, the cameras' hole
        # coordinates below, both fed from AppState signals.
        side = QVBoxLayout()
        side.setSpacing(10)
        self._summary = CycleSummaryPanel()
        self._coordinates = CameraCoordinatesPanel(camera_names)
        model_name, model_code = app_state.active_machine_model
        if model_name:
            self._summary.set_model(f"{model_name} ({model_code})")
        self._summary.set_shift(self._stored_shift())
        side.addWidget(self._summary)
        side.addWidget(self._coordinates)
        side.addStretch()
        content.addLayout(side, stretch=1)

        root.addLayout(content, stretch=1)

        # --------------------------------------------------------- wiring
        app_state.preview_frame.connect(self._on_preview)
        app_state.camera_captured.connect(self._on_camera_captured)
        app_state.camera_inspected.connect(self._on_camera_inspected)
        app_state.camera_state_changed.connect(self._on_camera_state)
        app_state.trigger_received.connect(self._on_trigger)
        app_state.inspection_completed.connect(self._on_inspection)
        app_state.counters_changed.connect(self._on_counters)
        app_state.active_machine_model_changed.connect(self._on_machine_model_changed)

        self._load_initial()

    # ------------------------------------------------------- trigger bar
    def _build_trigger_controls(self) -> QWidget:
        """Manual trigger button + the delay between cameras."""
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        caption = QLabel("Delay between cameras")
        caption.setProperty("class", "dim")
        layout.addWidget(caption)

        self._delay = QSpinBox()
        self._delay.setRange(0, MAX_CAMERA_DELAY_MS)
        self._delay.setSingleStep(100)
        self._delay.setSuffix(" ms")
        self._delay.setToolTip(
            "Pause between one camera finishing and the next one taking its picture"
        )
        self._delay.setValue(self._stored_delay_ms())
        self._delay.valueChanged.connect(self._on_delay_changed)
        layout.addWidget(self._delay)

        self._simulate_button = QPushButton("▶  Simulate Trigger")
        self._simulate_button.setProperty("class", "primary")
        self._simulate_button.setToolTip(
            "Run one inspection: every enabled camera takes a picture in turn"
        )
        if self._on_simulate_trigger is None:
            self._simulate_button.setEnabled(False)
        else:
            self._simulate_button.clicked.connect(self._on_simulate_clicked)
        layout.addWidget(self._simulate_button)
        return bar

    def _stored_delay_ms(self) -> int:
        """Configured inter-camera delay, or the pipeline default."""
        if self._config is None:
            return DEFAULT_CAMERA_DELAY_MS
        try:
            value = self._config.get_value(
                "app_config", "inspection.camera_delay_ms", DEFAULT_CAMERA_DELAY_MS
            )
            return max(0, min(int(value), MAX_CAMERA_DELAY_MS))
        except (ConfigurationError, TypeError, ValueError):
            return DEFAULT_CAMERA_DELAY_MS

    def _on_delay_changed(self, value: int) -> None:
        """Persist the delay so the running pipeline picks it up next cycle."""
        if self._config is None or self._loading:
            return
        try:
            document = self._config.load("app_config")
            inspection = document.setdefault("inspection", {})
            inspection["camera_delay_ms"] = int(value)
            inspection.setdefault("capture_mode", SEQUENTIAL_MODE)
            self._config.save("app_config", document)
        except ConfigurationError as exc:
            logger.error("Could not save the camera delay: %s", exc)
            self._app_state.raise_alarm("warning", f"Could not save the camera delay: {exc}")

    def _on_simulate_clicked(self) -> None:
        self._set_triggers_enabled(False)  # re-enabled when the cycle ends
        self._arm_button_watchdog()
        self._on_simulate_trigger()

    def _on_camera_trigger_clicked(self, camera_index: int) -> None:
        """Per-camera Trigger button: inspect this camera alone.

        Locks every trigger button for the duration, because the inspection
        worker runs one cycle at a time — a second request while one is in
        flight would simply be dropped.
        """
        if self._on_camera_trigger is None:
            return
        self._set_triggers_enabled(False)
        self._arm_button_watchdog()
        self._on_camera_trigger(camera_index)

    def _set_triggers_enabled(self, enabled: bool) -> None:
        """Enable/disable every manual trigger control at once."""
        self._simulate_button.setEnabled(enabled and self._on_simulate_trigger is not None)
        for panel in self._panels.values():
            panel.set_trigger_enabled(enabled and self._on_camera_trigger is not None)

    def _arm_button_watchdog(self) -> None:
        """Never leave the operator without a button if a cycle dies silently."""
        budget_ms = 60000 + len(self._panels) * self._delay.value()
        QTimer.singleShot(budget_ms, lambda: self._set_triggers_enabled(True))

    # ------------------------------------------------------------- initial
    def _load_initial(self) -> None:
        total, good, ng = self._app_state.counters
        self._on_counters(total, good, ng)

    # ----------------------------------------------------------------- slots
    def _on_preview(self, camera_index: int, frame) -> None:
        panel = self._panels.get(camera_index)
        if panel is not None:
            panel.update_preview(frame)

    def _on_camera_captured(self, camera_index: int, frame) -> None:
        """A camera just took its picture — show it before detection runs."""
        panel = self._panels.get(camera_index)
        if panel is not None:
            panel.show_capture(frame)

    def _on_camera_inspected(self, camera_index: int, data: CameraInspectionData) -> None:
        """A camera has been judged, mid-cycle."""
        panel = self._panels.get(camera_index)
        if panel is not None:
            panel.show_result(data)
        self._show_coordinates(camera_index, data)

    def _on_camera_state(self, camera_index: int, state: str) -> None:
        panel = self._panels.get(camera_index)
        if panel is not None:
            panel.set_camera_state(state)

    def _on_trigger(self, machine_number: int) -> None:
        self._summary.set_serial(str(machine_number))
        # Last cycle's coordinates are about to be replaced camera by
        # camera; blanking them keeps a stale reading from being read as
        # this machine's.
        self._coordinates.clear()
        self._set_triggers_enabled(False)  # one cycle at a time

    def _on_inspection(self, cycle: InspectionCycleData) -> None:
        self._set_triggers_enabled(True)
        self._summary.set_serial(cycle.serial_number)
        self._summary.set_last_trigger(cycle.started_at.strftime("%H:%M:%S"))
        self._summary.push_cycle_time(cycle.plc_cycle_time_ms)
        if cycle.shift:
            self._summary.set_shift(cycle.shift)

        for camera_index, data in cycle.cameras.items():
            panel = self._panels.get(camera_index)
            if panel is not None:
                panel.show_result(data)
            self._show_coordinates(camera_index, data)

    def _on_machine_model_changed(self, name: str, code: int) -> None:
        self._summary.set_model(f"{name} ({code})")

    def _on_home_requested(self, camera_index: int) -> None:
        """Home button on a camera panel — moves the camera to the *active*
        machine model's saved image capture position (the same target
        "Go to Default" on the Camera Configuration page writes), not to
        zero; admin-gated like the PLC page's manual register write. No
        QMessageBox precedent on this page, so both the auth check and any
        failure surface as an alarm instead."""
        if not self._auth.is_admin:
            self._app_state.raise_alarm(
                "warning", "Administrator login required to move a camera (toolbar Login button)."
            )
            return
        _name, code = self._app_state.active_machine_model
        profile = self._machine_models.get_by_code(code) if code is not None else None
        position = (profile or {}).get("jog_positions", {}).get(str(camera_index))
        if position is None:
            self._app_state.raise_alarm(
                "warning",
                f"Camera {camera_index}: no image capture position saved for the "
                f"active machine model.",
            )
            return
        try:
            self._plc.set_camera_position(
                camera_index, int(position["x"]), int(position["y"]), int(position.get("z", 0))
            )
        except VisionSystemError as exc:
            self._app_state.raise_alarm("warning", f"Camera {camera_index} move failed: {exc}")

    def _on_counters(self, total: int, good: int, ng: int) -> None:
        """Only the total is shown; good/ng still arrive on the signal and
        remain available in history and reports."""
        self._summary.set_product_count(total)

    # ------------------------------------------------------------- internal
    def _show_coordinates(self, camera_index: int, data: CameraInspectionData) -> None:
        """A camera with no hole has no coordinate to show — its registers
        carry the no-hole sentinel, not a position."""
        if data.hole_found:
            self._coordinates.set_position(camera_index, data.x_mm, data.y_mm)
        else:
            self._coordinates.clear_position(camera_index)

    def _stored_shift(self) -> str:
        """Shift as configured on the Settings page (what the pipeline
        stamps on each cycle); a completed cycle overwrites it."""
        if self._config is None:
            return ""
        try:
            return str(self._config.get_value("app_config", "application.shift", ""))
        except (ConfigurationError, TypeError, ValueError):
            return ""


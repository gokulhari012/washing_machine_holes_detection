"""Dashboard: manual trigger bar, 2×2 camera grid, summary column.

Entirely event-driven off :class:`AppState` signals — the page performs no
polling, touches no hardware and reads no database; the counters it starts
from come from :class:`AppState`'s snapshot. Inspection history lives on the
Database page.

The trigger bar drives the manual cycle: "Simulate Trigger" runs one complete
inspection in the configured capture mode, and the delay spin box sets how long
the pipeline waits between cameras when the sequential capture mode is on. The
delay is persisted to app_config.json immediately, so the next cycle — and the
next start of the application — uses it.

The spin box is ``app_config.inspection.camera_delay_ms`` itself, **not** a
setting local to the button beside it: it paces every sequential cycle,
PLC-triggered ones included, which is why the bar says so on screen as well as
in the tooltip. Three gaps between four cameras means the PLC waits 3x the
delay longer for ``vision_complete`` (register 119) on every part.

That whole bar is **developer-only**, like the toolbar's own trigger button:
firing the station by hand and re-timing its capture sequence are commissioning
acts, not shift work, so the bar is hidden (not merely disabled) for the
logged-out operator and for admins. Visibility is re-read from
:class:`AuthService` on every login and logout, via its observer hook — a page
that only checked at construction would keep an operator view after a developer
logs in. The per-camera panel triggers are unaffected.

The summary card's "Current Shift" tile follows ``AppState.current_shift``,
which the composition root republishes as the configured rota crosses a
handover. It is deliberately *not* driven by the finished cycle's own
``shift`` stamp: that records the shift a cycle began in, which goes stale on
the tile the moment the rota moves on.
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
from core.utilities.enums import LogSource, UserRole
from core.utilities.exceptions import ConfigurationError
from models.app_state import AppState
from models.dto import CameraInspectionData, InspectionCycleData
from services.auth_service import AuthService
from services.inspection_service import DEFAULT_CAMERA_DELAY_MS, SEQUENTIAL_MODE
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
        config_manager: ConfigManager | None = None,
        on_simulate_trigger=None,
        on_camera_trigger=None,
        auth_service: AuthService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._app_state = app_state
        self._config = config_manager
        self._auth = auth_service
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
            panel.set_trigger_enabled(on_camera_trigger is not None)
            panel.trigger_requested.connect(self._on_camera_trigger_clicked)
            self._panels[index] = panel
            grid.addWidget(panel, position // 2, position % 2)
        content.addLayout(grid, stretch=4)

        # The right-hand column: cycle summary on top, the cameras' hole
        # coordinates below, both fed from AppState signals. Both are given a
        # layout stretch factor (5:4, matching their row counts) rather than
        # a trailing addStretch(), so the two bordered cards fill the whole
        # column height with no dead space beneath them.
        side = QVBoxLayout()
        side.setSpacing(10)
        self._summary = CycleSummaryPanel()
        self._coordinates = CameraCoordinatesPanel(camera_names)
        model_name, model_code = app_state.active_machine_model
        if model_name:
            self._summary.set_model(f"{model_name} ({model_code})")
        self._summary.set_shift(app_state.current_shift)
        side.addWidget(self._summary, stretch=5)
        side.addWidget(self._coordinates, stretch=4)
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
        app_state.current_shift_changed.connect(self._summary.set_shift)
        if self._auth is not None:
            self._auth.subscribe(self._refresh_access)
        self._refresh_access()

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
            "Pause between one camera finishing and the next one taking its "
            "picture.\n\nThis is a station setting, not a button setting: it "
            "applies to every sequential cycle, including the ones the PLC "
            "triggers on register 100 - four cameras means three of these "
            "pauses, so the PLC waits that much longer for vision_complete."
        )
        self._delay.setValue(self._stored_delay_ms())
        self._delay.valueChanged.connect(self._on_delay_changed)
        layout.addWidget(self._delay)

        # Said on screen, not just in the tooltip: the spin box sits beside a
        # manual trigger button, which makes it look like it only paces that
        # button's cycle. It paces PLC-triggered cycles too, and at a few
        # seconds x3 gaps that is enough to trip a PLC waiting on 119.
        note = QLabel("(PLC triggers too)")
        note.setProperty("class", "dim")
        note.setToolTip(self._delay.toolTip())
        layout.addWidget(note)

        self._simulate_button = QPushButton("▶  Simulate Trigger")
        self._simulate_button.setProperty("class", "primary")
        self._simulate_button.setToolTip(
            "Run one inspection: every enabled camera takes a picture in turn, "
            "in the station's configured capture mode — the same cycle a PLC "
            "trigger on register 100 runs"
        )
        if self._on_simulate_trigger is None:
            self._simulate_button.setEnabled(False)
        else:
            self._simulate_button.clicked.connect(self._on_simulate_clicked)
        layout.addWidget(self._simulate_button)
        self._trigger_bar = bar
        return bar

    def _refresh_access(self) -> None:
        """Show the manual trigger bar to developers only.

        No auth service (tests, a station built without login) leaves it
        visible — that is the behaviour this page had before the gate existed.
        """
        visible = self._auth is None or self._auth.has_role(UserRole.DEVELOPER)
        self._trigger_bar.setVisible(visible)

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
        # The serial belongs to the PLC (register + configured prefix) and
        # arrives with the finished cycle, not with the trigger; blanking it
        # and the coordinates keeps last machine's readings from being read
        # as this one's.
        self._summary.set_serial("")
        self._coordinates.clear()
        self._set_triggers_enabled(False)  # one cycle at a time

    def _on_inspection(self, cycle: InspectionCycleData) -> None:
        self._set_triggers_enabled(True)
        self._summary.set_serial(cycle.serial_number)
        self._summary.set_last_trigger(cycle.started_at.strftime("%H:%M:%S"))
        self._summary.push_cycle_time(cycle.plc_cycle_time_ms)
        # The shift tile is fed by current_shift_changed, not by the finished
        # cycle: cycle.shift is the stamp of the shift the cycle *started* in,
        # which is a moment in the past and would read as stale on the tile
        # for any cycle that straddled a handover.

        for camera_index, data in cycle.cameras.items():
            panel = self._panels.get(camera_index)
            if panel is not None:
                panel.show_result(data)
            self._show_coordinates(camera_index, data)

    def _on_machine_model_changed(self, name: str, code: int) -> None:
        self._summary.set_model(f"{name} ({code})")

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



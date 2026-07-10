"""Application frame: left navigation rail, toolbar, alarm banner, pages,
status bar (PLC + camera LEDs, factory name, live clock).

The window is generic — the composition root constructs the pages and adds
them via :meth:`add_page`, and passes callbacks for the two things the frame
itself triggers: simulating a trigger and shutting the application down.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.logging import get_logger
from core.utilities.enums import LogSource
from models.app_state import AppState
from ui.widgets import AlarmBanner, LabeledLed

logger = get_logger(LogSource.UI)

NAV_WIDTH = 190
STATUS_MESSAGE_MS = 5000


class MainWindow(QMainWindow):
    """Industrial-style shell hosting all pages."""

    def __init__(
        self,
        app_state: AppState,
        factory_name: str,
        camera_indexes: list[int],
        on_simulate_trigger: Callable[[], None] | None = None,
        on_shutdown: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._app_state = app_state
        self._on_simulate_trigger = on_simulate_trigger
        self._on_shutdown = on_shutdown
        self._shutdown_done = False

        self.setWindowTitle("Washing Machine Bottom Hole Detection System")
        self.resize(1440, 900)

        self._build_toolbar()
        self._build_central()
        self._build_status_bar(factory_name, camera_indexes)
        self._wire_app_state()

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()

    # -------------------------------------------------------------- toolbar
    def _build_toolbar(self) -> None:
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        brand = QLabel("  ◉ WM Hole Detection")
        brand.setStyleSheet("font-size: 15px; font-weight: 700; color: #e8ecf2;")
        toolbar.addWidget(brand)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self._simulate_button = QPushButton("Simulate Trigger")
        self._simulate_button.setProperty("class", "primary")
        self._simulate_button.setToolTip("Run one manual inspection cycle")
        if self._on_simulate_trigger is None:
            self._simulate_button.setEnabled(False)
        else:
            self._simulate_button.clicked.connect(self._on_simulate_trigger)
        toolbar.addWidget(self._simulate_button)

    # -------------------------------------------------------------- central
    def _build_central(self) -> None:
        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._nav = QListWidget()
        self._nav.setObjectName("navRail")
        self._nav.setFixedWidth(NAV_WIDTH)
        self._nav.currentRowChanged.connect(self._on_nav_changed)
        outer.addWidget(self._nav)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        self.alarm_banner = AlarmBanner()
        banner_wrap = QVBoxLayout()
        banner_wrap.setContentsMargins(10, 8, 10, 0)
        banner_wrap.addWidget(self.alarm_banner)
        right.addLayout(banner_wrap)

        self._stack = QStackedWidget()
        right.addWidget(self._stack, stretch=1)
        outer.addLayout(right, stretch=1)

        self.setCentralWidget(central)

    # ------------------------------------------------------------ statusbar
    def _build_status_bar(self, factory_name: str, camera_indexes: list[int]) -> None:
        bar = self.statusBar()

        self._plc_led = LabeledLed("PLC")
        bar.addWidget(self._space(8))
        bar.addWidget(self._plc_led)
        bar.addWidget(self._space(16))

        self._camera_leds: dict[int, LabeledLed] = {}
        for index in sorted(camera_indexes):
            led = LabeledLed(f"C{index}")
            self._camera_leds[index] = led
            bar.addWidget(led)
            bar.addWidget(self._space(8))

        self._clock = QLabel("")
        factory = QLabel(factory_name)
        factory.setProperty("class", "dim")
        bar.addPermanentWidget(factory)
        bar.addPermanentWidget(self._space(16))
        bar.addPermanentWidget(self._clock)
        bar.addPermanentWidget(self._space(8))

    @staticmethod
    def _space(width: int) -> QWidget:
        spacer = QWidget()
        spacer.setFixedWidth(width)
        return spacer

    # ---------------------------------------------------------------- pages
    def add_page(self, title: str, icon_glyph: str, page: QWidget) -> None:
        """Register a page in the nav rail (order of calls = nav order)."""
        item = QListWidgetItem(f"{icon_glyph}  {title}")
        item.setSizeHint(QSize(0, 40))
        self._nav.addItem(item)
        self._stack.addWidget(page)
        if self._nav.count() == 1:
            self._nav.setCurrentRow(0)

    def show_page(self, index: int) -> None:
        self._nav.setCurrentRow(index)

    def _on_nav_changed(self, row: int) -> None:
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)

    # --------------------------------------------------------------- wiring
    def _wire_app_state(self) -> None:
        state = self._app_state
        state.alarm_raised.connect(self.alarm_banner.show_alarm)
        state.plc_state_changed.connect(
            lambda value: self._plc_led.set_state(value, f"PLC {value}")
        )
        state.camera_state_changed.connect(self._on_camera_state)
        state.status_message.connect(
            lambda message: self.statusBar().showMessage(message, STATUS_MESSAGE_MS)
        )

    def _on_camera_state(self, camera_index: int, state: str) -> None:
        led = self._camera_leds.get(camera_index)
        if led is not None:
            led.set_state(state)

    def _update_clock(self) -> None:
        self._clock.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    # ------------------------------------------------------------- shutdown
    def closeEvent(self, event) -> None:  # noqa: N802
        if not self._shutdown_done and self._on_shutdown is not None:
            self._shutdown_done = True
            logger.info("Main window closing — shutting down")
            try:
                self._on_shutdown()
            except Exception:
                logger.exception("Shutdown callback raised")
        event.accept()

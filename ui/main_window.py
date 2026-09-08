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
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.logging import get_logger
from core.utilities.enums import LogSource
from models.app_state import AppState
from services.auth_service import AuthService
from ui.widgets import AlarmBanner, LabeledLed, LoginDialog, install_wheel_guard

logger = get_logger(LogSource.UI)

NAV_WIDTH = 190
STATUS_MESSAGE_MS = 5000


class MainWindow(QMainWindow):
    """Industrial-style shell hosting all pages."""

    def __init__(
        self,
        app_state: AppState,
        auth_service: AuthService,
        factory_name: str,
        camera_indexes: list[int],
        on_simulate_trigger: Callable[[], None] | None = None,
        on_shutdown: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._app_state = app_state
        self._auth = auth_service
        self._on_simulate_trigger = on_simulate_trigger
        self._on_shutdown = on_shutdown
        self._shutdown_done = False
        self._pages: list[tuple[QListWidgetItem, bool]] = []  # (nav item, admin_only)

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

        self._nav_button = QPushButton("Menu")
        self._nav_button.setCheckable(True)
        self._nav_button.setToolTip("Show/hide the navigation rail")
        self._nav_button.toggled.connect(self._set_nav_visible)
        toolbar.addWidget(self._space(6))
        toolbar.addWidget(self._nav_button)

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

        toolbar.addWidget(self._space(16))
        self._session_label = QLabel()
        self._session_label.setProperty("class", "dim")
        toolbar.addWidget(self._session_label)
        toolbar.addWidget(self._space(8))
        self._login_button = QPushButton()
        self._login_button.setToolTip(
            "Administrator login — unlocks Cameras, PLC, Detection, "
            "Calibration and Settings in the nav rail"
        )
        self._login_button.clicked.connect(self._on_login_clicked)
        toolbar.addWidget(self._login_button)
        self._update_session_label()

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
        # Hidden on startup: the operator lives on the Dashboard, and the
        # extra width belongs to the camera pictures. The toolbar's Menu
        # button
        # brings it back. Hiding the widget (rather than removing it) keeps
        # row selection, page order and admin visibility working untouched.
        self._nav.setVisible(False)
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
        # The shift sits beside the clock because that is what it is derived
        # from: an operator glancing at the bar can see the handover happen.
        self._shift = QLabel("")
        self._shift.setProperty("class", "dim")
        self._shift.setToolTip("Current production shift (Settings -> Shift Schedule)")
        factory = QLabel(factory_name)
        factory.setProperty("class", "dim")
        bar.addPermanentWidget(factory)
        bar.addPermanentWidget(self._space(16))
        bar.addPermanentWidget(self._shift)
        bar.addPermanentWidget(self._space(16))
        bar.addPermanentWidget(self._clock)
        bar.addPermanentWidget(self._space(8))
        self._on_shift_changed(self._app_state.current_shift)

    @staticmethod
    def _space(width: int) -> QWidget:
        spacer = QWidget()
        spacer.setFixedWidth(width)
        return spacer

    # ---------------------------------------------------------------- pages
    def add_page(
        self, title: str, icon_glyph: str, page: QWidget, admin_only: bool = False
    ) -> None:
        """Register a page in the nav rail (order of calls = nav order).

        ``admin_only`` pages stay in the stack (so the composition root can
        wire signals normally) but are hidden from the nav rail until an
        administrator logs in — see :meth:`_refresh_nav_visibility`.

        The page is wrapped in a borderless, resizable :class:`QScrollArea`:
        several pages (PLC register map, Calibration's homography grid) are
        tall enough to clip on a 1080p screen once DPI scaling or a taskbar
        eats into the usable height. ``setWidgetResizable`` makes the page
        fill the viewport and stretch normally when there is room, and only
        a plain vertical scrollbar appears when there is not — so this
        changes nothing on a screen tall enough for the page as-is.

        Because of that wrapping the page also gets a wheel guard, so
        scrolling past a spin box or combo box cannot silently edit it —
        see :func:`ui.widgets.install_wheel_guard`.
        """
        item = QListWidgetItem(f"{icon_glyph}  {title}")
        item.setSizeHint(QSize(0, 40))
        self._nav.addItem(item)

        # Called once the page is fully built, so a single walk catches every
        # spin box / combo box / slider on it.
        install_wheel_guard(page)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(page)
        self._stack.addWidget(scroll)
        self._pages.append((item, admin_only))
        if self._nav.count() == 1:
            self._nav.setCurrentRow(0)
        self._refresh_nav_visibility()

    def _set_nav_visible(self, visible: bool) -> None:
        self._nav.setVisible(visible)

    def show_page(self, index: int) -> None:
        self._nav.setCurrentRow(index)

    def _on_nav_changed(self, row: int) -> None:
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)

    # ------------------------------------------------------------------ auth
    def _refresh_nav_visibility(self) -> None:
        """Show/hide admin-only nav entries for the current session.

        Called after every ``add_page`` and every login/logout. If the
        currently selected row just became hidden (an admin logged out while
        looking at an admin-only page), falls back to the first visible row
        — Dashboard is always visible, so this always finds one.
        """
        first_visible = None
        for row, (item, admin_only) in enumerate(self._pages):
            hidden = admin_only and not self._auth.is_admin
            item.setHidden(hidden)
            if not hidden and first_visible is None:
                first_visible = row

        current = self._nav.currentRow()
        currently_hidden = not (0 <= current < len(self._pages)) or self._pages[current][0].isHidden()
        if currently_hidden and first_visible is not None:
            self._nav.setCurrentRow(first_visible)

    def _update_session_label(self) -> None:
        user = self._auth.current_user
        if user is None:
            self._session_label.setText("Not logged in")
            self._login_button.setText("Log in")
        else:
            self._session_label.setText(f"{user.username} ({user.role})")
            self._login_button.setText("Log out")

    def _on_login_clicked(self) -> None:
        if self._auth.current_user is not None:
            self._auth.logout()
        else:
            dialog = LoginDialog(self._auth, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
        self._update_session_label()
        self._refresh_nav_visibility()

    # --------------------------------------------------------------- wiring
    def _wire_app_state(self) -> None:
        state = self._app_state
        state.alarm_raised.connect(self.alarm_banner.show_alarm)
        state.plc_state_changed.connect(
            lambda value: self._plc_led.set_state(value, f"PLC {value}")
        )
        state.camera_state_changed.connect(self._on_camera_state)
        state.current_shift_changed.connect(self._on_shift_changed)
        state.status_message.connect(
            lambda message: self.statusBar().showMessage(message, STATUS_MESSAGE_MS)
        )
        # one cycle at a time — the toolbar button follows the dashboard one
        state.trigger_received.connect(lambda _machine: self._set_simulate_enabled(False))
        state.inspection_completed.connect(lambda _cycle: self._set_simulate_enabled(True))

    def _set_simulate_enabled(self, enabled: bool) -> None:
        self._simulate_button.setEnabled(enabled and self._on_simulate_trigger is not None)

    def _on_camera_state(self, camera_index: int, state: str) -> None:
        led = self._camera_leds.get(camera_index)
        if led is not None:
            led.set_state(state)

    def _on_shift_changed(self, name: str) -> None:
        """Blank the label rather than show a placeholder when no shift is
        resolved — an empty rota is not worth a permanent dash in the bar."""
        self._shift.setText(f"Shift: {name}" if name else "")

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

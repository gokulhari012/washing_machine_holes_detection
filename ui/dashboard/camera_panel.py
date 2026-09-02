"""One dashboard camera panel: live view + last inspection readout.

Behaviour: the panel streams live preview frames; when an inspection
completes it shows the annotated result frame and **holds** it for a short
time (so the operator can see what was judged) before resuming live video.
Preview frames are downscaled before display to keep the UI thread light at
4 cameras × 15 fps.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from core.utilities.enums import ConnectionState, InspectionResult
from models.dto import CameraInspectionData
from ui.widgets import ImageView, LabeledLed, home_icon, play_icon

RESULT_HOLD_S = 1.5
PREVIEW_MAX_WIDTH = 640
ICON_PX = 15  # icon size inside the panel's square header buttons


class CameraPanel(QFrame):
    """Live image + name/LED header + X/Y/confidence/result footer."""

    #: emitted when the Home button is clicked, with this panel's camera index —
    #: DashboardPage owns the PLC call (admin gate, error handling); the panel
    #: itself knows nothing about PLC/auth.
    home_requested = Signal(int)

    #: emitted when the Trigger button is clicked, with this panel's camera
    #: index — inspect this camera alone. DashboardPage owns the dispatch.
    trigger_requested = Signal(int)

    def __init__(self, camera_index: int, camera_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "panel")
        self.camera_index = camera_index
        self._hold_until = 0.0

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # header: LED + name + trigger/home buttons + result badge
        header = QHBoxLayout()
        self._led = LabeledLed(camera_name)
        # Painted icons rather than text glyphs — see ui/widgets/icons.py for
        # why (the old ⌂ is missing from several stock Windows UI fonts).
        self._trigger_btn = QPushButton()
        self._trigger_btn.setIcon(play_icon("#ffffff", "#7d8794"))
        self._trigger_btn.setIconSize(QSize(ICON_PX, ICON_PX))
        self._trigger_btn.setProperty("class", "panelIconAccent")
        self._trigger_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._trigger_btn.setToolTip(
            f"Inspect camera {camera_index} on its own — the other cameras are "
            f"not captured and their results are left unchanged"
        )
        self._trigger_btn.clicked.connect(
            lambda: self.trigger_requested.emit(self.camera_index)
        )
        self._home_btn = QPushButton()
        self._home_btn.setIcon(home_icon())
        self._home_btn.setIconSize(QSize(ICON_PX, ICON_PX))
        self._home_btn.setProperty("class", "panelIcon")
        self._home_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._home_btn.setToolTip("Return this camera to its home position")
        self._home_btn.clicked.connect(lambda: self.home_requested.emit(self.camera_index))
        self._result = QLabel("—")
        self._result.setProperty("result", "")
        header.addWidget(self._led)
        header.addStretch()
        header.addWidget(self._trigger_btn)
        header.addWidget(self._home_btn)
        header.addWidget(self._result)
        root.addLayout(header)

        # live view
        self._view = ImageView()
        self._view.setMinimumHeight(180)
        root.addWidget(self._view, stretch=1)

        # footer: measurements
        footer = QHBoxLayout()
        footer.setSpacing(14)
        self._x_value = self._add_readout(footer, "X")
        self._y_value = self._add_readout(footer, "Y")
        self._conf_value = self._add_readout(footer, "CONF")
        footer.addStretch()
        root.addLayout(footer)

    # ------------------------------------------------------------------ api
    def set_camera_state(self, state: ConnectionState | str) -> None:
        self._led.set_state(state)

    def set_trigger_enabled(self, enabled: bool) -> None:
        """Disabled while any inspection is running — one cycle at a time."""
        self._trigger_btn.setEnabled(enabled)

    def set_home_enabled(self, enabled: bool) -> None:
        """Grey out Home when this camera has no jog registers configured."""
        self._home_btn.setEnabled(enabled)
        self._home_btn.setToolTip(
            "Return this camera to its home position"
            if enabled
            else "No PLC jog registers configured for this camera"
        )

    def update_preview(self, frame: np.ndarray) -> None:
        """Live frame from the acquisition worker; ignored during result hold."""
        if time.monotonic() < self._hold_until:
            return
        self._view.set_frame(self._downscale(frame))

    def show_capture(self, frame: np.ndarray) -> None:
        """Picture just taken for this cycle — shown at once, hold or not.

        Displayed before detection runs, so on a sequential cycle the panel
        lights up the instant its camera fires instead of at the end.
        """
        self._hold_until = 0.0
        self._view.set_frame(self._downscale(frame))
        self._result.setText("…")
        self._result.setProperty("result", "")
        self._repolish_result()

    def show_result(self, data: CameraInspectionData) -> None:
        """Annotated inspection outcome; freezes the view briefly."""
        if data.frame is not None:
            self._view.set_frame(self._downscale(data.frame))
        self._hold_until = time.monotonic() + RESULT_HOLD_S

        if data.hole_found:
            self._x_value.setText(f"{data.x_mm:.1f} mm")
            self._y_value.setText(f"{data.y_mm:.1f} mm")
            self._conf_value.setText(f"{data.confidence:.2f}")
        else:
            self._x_value.setText("—")
            self._y_value.setText("—")
            self._conf_value.setText("—")

        result = data.result
        self._result.setText(
            result.value if result is not InspectionResult.ERROR else "ERROR"
        )
        self._result.setProperty("result", result.value)
        self._repolish_result()

    # ------------------------------------------------------------- internal
    def _repolish_result(self) -> None:
        """Re-apply the stylesheet after the ``result`` property changed."""
        style = self._result.style()
        style.unpolish(self._result)
        style.polish(self._result)

    @staticmethod
    def _add_readout(layout: QHBoxLayout, caption: str) -> QLabel:
        cap = QLabel(caption)
        cap.setProperty("class", "dim")
        value = QLabel("—")
        layout.addWidget(cap)
        layout.addWidget(value)
        return value

    @staticmethod
    def _downscale(frame: np.ndarray) -> np.ndarray:
        width = frame.shape[1]
        if width <= PREVIEW_MAX_WIDTH:
            return frame
        scale = PREVIEW_MAX_WIDTH / width
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

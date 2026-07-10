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
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from core.utilities.enums import ConnectionState, InspectionResult
from models.dto import CameraInspectionData
from ui.widgets import ImageView, LabeledLed

RESULT_HOLD_S = 1.5
PREVIEW_MAX_WIDTH = 640


class CameraPanel(QFrame):
    """Live image + name/LED header + X/Y/confidence/result footer."""

    def __init__(self, camera_index: int, camera_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "panel")
        self.camera_index = camera_index
        self._hold_until = 0.0

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # header: LED + name + result badge
        header = QHBoxLayout()
        self._led = LabeledLed(camera_name)
        self._result = QLabel("—")
        self._result.setProperty("result", "")
        header.addWidget(self._led)
        header.addStretch()
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

    def update_preview(self, frame: np.ndarray) -> None:
        """Live frame from the acquisition worker; ignored during result hold."""
        if time.monotonic() < self._hold_until:
            return
        self._view.set_frame(self._downscale(frame))

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
        style = self._result.style()
        style.unpolish(self._result)
        style.polish(self._result)

    # ------------------------------------------------------------- internal
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

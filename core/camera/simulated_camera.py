"""Synthetic camera for hardware-free demo and testing.

Two modes:

1. **Image directory** — if ``simulation.image_directory`` (from camera.json)
   exists and contains images, frames cycle through them (resized to the
   configured resolution). Drop real production photos there to replay them.
2. **Synthesis** (default) — renders a brushed-metal washing-machine bottom
   with one drain hole (position jitters a few pixels per frame, and with a
   small probability the hole is absent → produces NG cycles), plus small
   bolt-head distractors below the minimum hole diameter so the detector's
   size filtering is exercised.

Exposure, gain, brightness and gamma all affect the rendered image so the
Camera Configuration page sliders give visible feedback.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.camera.camera_base import CameraBase, CameraSettings

# nominal hole centre per camera index, as a fraction of (width, height)
_BASE_POSITIONS: dict[int, tuple[float, float]] = {
    1: (0.40, 0.45),
    2: (0.60, 0.50),
    3: (0.50, 0.58),
    4: (0.45, 0.40),
}
_IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


class SimulatedCamera(CameraBase):
    """Deterministic-per-camera synthetic frame source."""

    HOLE_RADIUS_PX = 30
    JITTER_PX = 8

    def __init__(self, settings: CameraSettings) -> None:
        super().__init__(settings)
        sim_cfg = settings.extra.get("simulation", {}) or {}
        self._no_hole_probability = float(sim_cfg.get("no_hole_probability", 0.10))
        self._image_dir = Path(sim_cfg.get("image_directory", "")) if sim_cfg.get("image_directory") else None
        self._image_files: list[Path] = []
        self._image_cursor = 0
        # seeded per camera: bolt layout is stable, jitter varies per frame
        self._rng = np.random.default_rng(seed=settings.index * 97)
        fx, fy = _BASE_POSITIONS.get(settings.index, (0.5, 0.5))
        self._base_xy = (int(settings.width * fx), int(settings.height * fy))
        self._bolts = [
            (
                int(self._rng.integers(40, settings.width - 40)),
                int(self._rng.integers(40, settings.height - 40)),
            )
            for _ in range(4)
        ]

    # ---------------------------------------------------------- driver hooks
    def _connect_device(self) -> None:
        if self._image_dir is not None and self._image_dir.is_dir():
            self._image_files = sorted(
                path for pattern in _IMAGE_PATTERNS for path in self._image_dir.glob(pattern)
            )

    def _disconnect_device(self) -> None:
        self._image_files = []

    def _apply_to_device(self, settings: CameraSettings) -> None:
        pass  # settings are read live during synthesis

    def _grab(self) -> np.ndarray:
        if self._image_files:
            return self._next_file_frame()
        return self._synthesize()

    # -------------------------------------------------------------- internal
    def _next_file_frame(self) -> np.ndarray:
        path = self._image_files[self._image_cursor % len(self._image_files)]
        self._image_cursor += 1
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Unreadable simulation image: {path}")
        return cv2.resize(frame, (self._settings.width, self._settings.height))

    def _synthesize(self) -> np.ndarray:
        s = self._settings
        w, h = s.width, s.height
        rng = self._rng

        # brushed-metal background: mid grey + noise + horizontal smear
        img = np.full((h, w), 110, dtype=np.float32)
        img += rng.normal(0.0, 12.0, size=(h, w)).astype(np.float32)
        img = cv2.blur(img, (9, 1))

        # soft vignette so the surface is not perfectly flat
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        img *= 1.0 - 0.22 * (xx * xx + yy * yy)
        img = np.clip(img, 0, 255).astype(np.uint8)

        # bolt-head distractors (diameter ~14 px, below the 20 px minimum)
        for bx, by in self._bolts:
            cv2.circle(img, (bx, by), 7, 55, -1)
            cv2.circle(img, (bx, by), 7, 150, 1)

        # the drain hole (absent with a small probability -> NG demo cycles)
        if float(rng.random()) >= self._no_hole_probability:
            jitter = rng.integers(-self.JITTER_PX, self.JITTER_PX + 1, size=2)
            cx = int(self._base_xy[0] + jitter[0])
            cy = int(self._base_xy[1] + jitter[1])
            cv2.circle(img, (cx, cy), self.HOLE_RADIUS_PX, 22, -1)     # dark interior
            cv2.circle(img, (cx, cy), self.HOLE_RADIUS_PX + 2, 165, 2)  # chamfer ring

        # exposure / gain / brightness / gamma response for UI feedback
        factor = (s.exposure_us / 10000.0) * (2.0 ** (s.gain_db / 6.0))
        img = np.clip(img.astype(np.float32) * factor + s.brightness, 0, 255).astype(np.uint8)
        if abs(s.gamma - 1.0) > 1e-3:
            lut = np.clip(
                ((np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / s.gamma)) * 255.0, 0, 255
            ).astype(np.uint8)
            img = lut[img]

        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

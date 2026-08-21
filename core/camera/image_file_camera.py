"""Still-image adapter — replays an uploaded picture as a live camera.

Chosen on the Camera Configuration page by selecting the ``image_file``
driver and picking a picture with "Choose Image…" (or a folder with
"Choose Folder…"). The picked path is stored per camera as ``image_source``
in camera.json.

Because it is an ordinary :class:`CameraBase`, everything downstream treats
it exactly like a real device: the acquisition worker streams it into the
live preview and the inspection pipeline runs the detector on it, so the
uploaded image is inspected live. A single file is re-served on every grab;
a folder cycles through its images, one per grab.

The picture is fitted into the configured resolution — aspect ratio
preserved, padded with black — so holes stay circular and their pixel
diameters stay comparable with the thresholds on the Detection page.
``brightness`` and ``gamma`` are applied so those fields still give visible
feedback; ``exposure_us``/``gain_db`` are sensor concepts with no meaning
for a file and are ignored.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.camera.camera_base import CameraBase, CameraSettings
from core.utilities.exceptions import CameraCaptureError, CameraConnectionError

IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
IMAGE_NAME_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"


def read_image(path: Path | str) -> np.ndarray | None:
    """Decode an image file as BGR, or ``None`` if it cannot be read.

    Goes through ``np.fromfile`` + ``imdecode`` because ``cv2.imread``
    cannot open paths with non-ASCII characters on Windows.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class ImageFileCamera(CameraBase):
    """Serves an uploaded image (or a folder of images) as camera frames."""

    def __init__(self, settings: CameraSettings) -> None:
        super().__init__(settings)
        self._paths: list[Path] = []
        self._cursor = 0
        self._cache: dict[Path, tuple[float, np.ndarray]] = {}  # path -> (mtime, frame)

    # ---------------------------------------------------------- driver hooks
    def _connect_device(self) -> None:
        source = source_path(self._settings)
        if source is None:
            raise CameraConnectionError(
                f"{self.name}: no image chosen — pick one with 'Choose Image…' "
                f"on the Camera Configuration page"
            )
        if source.is_dir():
            paths = sorted(p for pattern in IMAGE_PATTERNS for p in source.glob(pattern))
            if not paths:
                raise CameraConnectionError(f"{self.name}: no images in folder {source}")
        elif source.is_file():
            paths = [source]
        else:
            raise CameraConnectionError(f"{self.name}: image not found: {source}")

        self._paths = paths
        self._cursor = 0
        self._cache.clear()
        self._load(paths[0])  # fail fast on an unreadable/unsupported file

    def _disconnect_device(self) -> None:
        self._paths = []
        self._cache.clear()

    def _apply_to_device(self, settings: CameraSettings) -> None:
        pass  # settings are read live on every grab

    def _grab(self) -> np.ndarray:
        if not self._paths:
            raise CameraCaptureError(f"{self.name}: no image source")
        path = self._paths[self._cursor % len(self._paths)]
        self._cursor += 1
        source = self._load(path)
        frame = self._fit(source)
        if frame is source:
            frame = frame.copy()  # never hand out the cached array
        return self._adjust(frame)

    # -------------------------------------------------------------- internal
    def _load(self, path: Path) -> np.ndarray:
        """Decoded image for ``path``, re-read when the file changed on disk."""
        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            raise CameraCaptureError(f"{self.name}: image unavailable: {path} ({exc})") from exc
        cached = self._cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        frame = read_image(path)
        if frame is None:
            raise CameraCaptureError(f"{self.name}: cannot decode image {path}")
        self._cache[path] = (mtime, frame)
        return frame

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox into the configured resolution (aspect preserved)."""
        target_w, target_h = self._settings.width, self._settings.height
        height, width = frame.shape[:2]
        if (width, height) == (target_w, target_h):
            return frame
        scale = min(target_w / width, target_h / height)
        new_w = max(1, min(target_w, round(width * scale)))
        new_h = max(1, min(target_h, round(height * scale)))
        resized = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        canvas = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas

    def _adjust(self, frame: np.ndarray) -> np.ndarray:
        """Brightness offset + gamma curve, so those form fields still act."""
        s = self._settings
        if s.brightness:
            frame = np.clip(frame.astype(np.int16) + s.brightness, 0, 255).astype(np.uint8)
        if abs(s.gamma - 1.0) > 1e-3:
            lut = np.clip(
                ((np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / s.gamma)) * 255.0, 0, 255
            ).astype(np.uint8)
            frame = lut[frame]
        return frame


def source_path(settings: CameraSettings) -> Path | None:
    """The configured ``image_source`` for a camera, or ``None`` if unset."""
    raw = str(settings.extra.get("image_source", "") or "").strip()
    return Path(raw) if raw else None

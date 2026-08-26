"""Camera calibration model: pixel→mm conversion and perspective correction.

Two accuracy levels, chosen automatically by what has been calibrated:

- **Scale only** — ``pixels_per_mm_x/y`` from a two-point distance
  measurement; adequate when the camera is square to the surface.
- **Homography** — a 3×3 projective mapping from ≥4 pixel↔mm point pairs
  (``cv2.findHomography``); corrects perspective when the camera views the
  surface at an angle.

Both can be filled in by hand (typed distances / typed point pairs) or by
:meth:`CameraCalibration.find_checkerboard`, which turns one photo of a
checkerboard placed on/near the inspection plane into dozens of pixel↔mm
correspondences automatically — see the Calibration page's "Auto Calibrate"
section.

The *reference point* is the nominal hole position in mm; ``deviation_mm``
against it drives the position-tolerance judgement.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.database.models import Calibration as CalibrationRow
from core.utilities.exceptions import CalibrationError


@dataclass
class CheckerboardDetection:
    """One checkerboard photo, resolved into calibration-ready correspondences."""

    pixel_points: list[tuple[float, float]]
    mm_points: list[tuple[float, float]]
    pixels_per_mm_x: float
    pixels_per_mm_y: float
    corners_px: np.ndarray  # (N,1,2) — for cv2.drawChessboardCorners overlays


@dataclass
class CameraCalibration:
    """In-memory calibration for one camera."""

    camera_index: int
    pixels_per_mm_x: float = 1.0
    pixels_per_mm_y: float = 1.0
    homography: np.ndarray | None = None  # 3x3, maps pixel -> mm plane
    ref_point_mm: tuple[float, float] = (0.0, 0.0)
    rms_error: float = 0.0
    calibrated_by: str = field(default="", compare=False)

    # ------------------------------------------------------------ conversion
    def pixel_to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convert one pixel coordinate to millimetres."""
        if self.homography is not None:
            point = np.array([[[float(x_px), float(y_px)]]], dtype=np.float64)
            mapped = cv2.perspectiveTransform(point, self.homography)
            return float(mapped[0, 0, 0]), float(mapped[0, 0, 1])
        if self.pixels_per_mm_x <= 0 or self.pixels_per_mm_y <= 0:
            raise CalibrationError(
                f"Camera {self.camera_index}: non-positive pixels_per_mm"
            )
        return x_px / self.pixels_per_mm_x, y_px / self.pixels_per_mm_y

    def deviation_mm(self, x_mm: float, y_mm: float) -> float:
        """Euclidean distance from the reference (nominal) hole position."""
        return math.hypot(x_mm - self.ref_point_mm[0], y_mm - self.ref_point_mm[1])

    # ----------------------------------------------------------- computation
    @staticmethod
    def compute_homography(
        pixel_points: list[tuple[float, float]],
        mm_points: list[tuple[float, float]],
    ) -> tuple[np.ndarray, float]:
        """Fit pixel→mm homography from ≥4 correspondences; returns (H, rms_mm).

        Raises:
            CalibrationError: too few points or degenerate geometry.
        """
        if len(pixel_points) < 4 or len(pixel_points) != len(mm_points):
            raise CalibrationError(
                "Homography needs >= 4 matched point pairs "
                f"(got {len(pixel_points)} px / {len(mm_points)} mm)"
            )
        src = np.asarray(pixel_points, dtype=np.float64).reshape(-1, 1, 2)
        dst = np.asarray(mm_points, dtype=np.float64).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(src, dst, method=0)
        if homography is None:
            raise CalibrationError("Homography fit failed (degenerate points?)")

        projected = cv2.perspectiveTransform(src, homography)
        residuals = np.linalg.norm(projected.reshape(-1, 2) - dst.reshape(-1, 2), axis=1)
        return homography, float(np.sqrt(np.mean(residuals**2)))

    @staticmethod
    def scale_from_distance(pixel_distance: float, mm_distance: float) -> float:
        """Two-point calibration: pixels per millimetre.

        Raises:
            CalibrationError: non-positive input.
        """
        if pixel_distance <= 0 or mm_distance <= 0:
            raise CalibrationError("Calibration distances must be positive")
        return pixel_distance / mm_distance

    @staticmethod
    def find_checkerboard(
        image: np.ndarray, columns: int, rows: int, square_size_mm: float
    ) -> CheckerboardDetection:
        """Locate a checkerboard's inner corners and build the pixel↔mm
        correspondences a board of this size implies — ready to hand straight
        to :meth:`compute_homography` (or use the returned scale on its own).

        ``columns``/``rows`` count *inner* corners (one less than the number
        of squares each way), matching ``cv2.findChessboardCorners``'s own
        convention — a standard 9x6-square board has 8x5 inner corners.

        The mm grid is anchored at whichever corner OpenCV happens to return
        first — a checkerboard looks identical rotated 180°, so this may be
        either physical corner and the board's placement need not be marked
        or oriented any particular way. That is harmless here: nothing in
        this system reads an absolute mm coordinate, only ``deviation_mm``
        against a reference point captured afterwards through this same
        homography — self-consistent regardless of which corner is "first".

        Raises:
            CalibrationError: bad arguments, or no board of this size found
                (check the pattern size, lighting, and that it is fully
                visible and reasonably flat in the frame).
        """
        if columns < 2 or rows < 2:
            raise CalibrationError("Checkerboard columns/rows must each be >= 2")
        if square_size_mm <= 0:
            raise CalibrationError("Checkerboard square size must be positive")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        found, corners = cv2.findChessboardCorners(
            gray, (columns, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            raise CalibrationError(
                f"No {columns}x{rows}-corner checkerboard found — check the "
                f"pattern size matches the physical board, and that it is "
                f"fully visible, flat, and well lit"
            )
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )

        grid = corners.reshape(rows, columns, 2)
        pixel_spacing_x = float(np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2).mean())
        pixel_spacing_y = float(np.linalg.norm(grid[1:, :, :] - grid[:-1, :, :], axis=2).mean())

        mm_grid = np.zeros((rows, columns, 2), dtype=np.float64)
        mm_grid[..., 0] = np.arange(columns) * square_size_mm
        mm_grid[..., 1] = np.arange(rows)[:, None] * square_size_mm

        return CheckerboardDetection(
            pixel_points=[(float(x), float(y)) for x, y in grid.reshape(-1, 2)],
            mm_points=[(float(x), float(y)) for x, y in mm_grid.reshape(-1, 2)],
            pixels_per_mm_x=pixel_spacing_x / square_size_mm,
            pixels_per_mm_y=pixel_spacing_y / square_size_mm,
            corners_px=corners,
        )

    # ---------------------------------------------------------- persistence
    @classmethod
    def from_row(cls, row: CalibrationRow) -> "CameraCalibration":
        homography = None
        if row.homography_json:
            try:
                homography = np.asarray(json.loads(row.homography_json), dtype=np.float64)
                if homography.shape != (3, 3):
                    raise ValueError(f"expected 3x3, got {homography.shape}")
            except (ValueError, TypeError) as exc:
                raise CalibrationError(
                    f"Corrupt homography for camera {row.camera_index}: {exc}"
                ) from exc
        return cls(
            camera_index=row.camera_index,
            pixels_per_mm_x=row.pixels_per_mm_x,
            pixels_per_mm_y=row.pixels_per_mm_y,
            homography=homography,
            ref_point_mm=(row.ref_point_x_mm, row.ref_point_y_mm),
            rms_error=row.rms_error,
            calibrated_by=row.calibrated_by,
        )

    def to_row(self) -> CalibrationRow:
        return CalibrationRow(
            camera_index=self.camera_index,
            pixels_per_mm_x=self.pixels_per_mm_x,
            pixels_per_mm_y=self.pixels_per_mm_y,
            homography_json=(
                json.dumps(self.homography.tolist()) if self.homography is not None else None
            ),
            ref_point_x_mm=self.ref_point_mm[0],
            ref_point_y_mm=self.ref_point_mm[1],
            rms_error=self.rms_error,
            calibrated_by=self.calibrated_by,
        )

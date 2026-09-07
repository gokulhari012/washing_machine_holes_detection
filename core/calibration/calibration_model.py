"""Camera calibration model: pixel→mm conversion and perspective correction.

Three accuracy levels, chosen automatically by what has been calibrated:

- **Scale only** — ``pixels_per_mm_x/y`` from a two-point distance
  measurement, or from :meth:`CameraCalibration.scale_from_points` (a
  hand-clicked ruler); adequate when the camera is square to the surface.
- **Homography** — a 3×3 projective mapping from ≥4 pixel↔mm point pairs
  (``cv2.findHomography``); corrects perspective when the camera views the
  surface at an angle.
- **Lens distortion + homography** — :meth:`CameraCalibration.calibrate_lens`
  fits a camera matrix + distortion coefficients from several checkerboard
  photos of the *same* board at different poses (``cv2.calibrateCamera``);
  every pixel is undistorted through that model *before* the homography/scale
  step, in :meth:`pixel_to_mm`. A single photo cannot separate true lens
  distortion from perspective, which is why this needs multiple, differently
  posed views — see the Calibration page's "Auto Calibrate" section.

Scale and homography can be filled in by hand (typed distances / typed point
pairs) or automatically from one or more checkerboard photos via
:meth:`CameraCalibration.find_checkerboard`, which turns a photo of a
checkerboard placed on/near the inspection plane into dozens of pixel↔mm
correspondences.

The *reference point* is the nominal hole position in mm; ``deviation_mm``
against it drives the position-tolerance judgement.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

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
class LensCalibrationResult:
    """Multi-view lens calibration output — see :meth:`CameraCalibration.calibrate_lens`."""

    camera_matrix: np.ndarray  # 3x3 intrinsics
    dist_coeffs: np.ndarray  # (1, 5): k1, k2, p1, p2, k3
    overall_rms_px: float
    per_view_rms_px: list[float]


@dataclass
class CameraCalibration:
    """In-memory calibration for one camera."""

    camera_index: int
    pixels_per_mm_x: float = 1.0
    pixels_per_mm_y: float = 1.0
    homography: np.ndarray | None = None  # 3x3, maps pixel -> mm plane
    camera_matrix: np.ndarray | None = None  # 3x3 intrinsics, from calibrate_lens
    dist_coeffs: np.ndarray | None = None  # paired with camera_matrix
    ref_point_mm: tuple[float, float] = (0.0, 0.0)
    rms_error: float = 0.0
    calibrated_by: str = field(default="", compare=False)

    # ------------------------------------------------------------ conversion
    def pixel_to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convert one pixel coordinate to millimetres.

        When lens calibration is present, the pixel is undistorted first —
        homography/scale below then act on that corrected coordinate, exactly
        as they did when computed (see :meth:`calibrate_lens`'s caller in the
        Calibration page, which undistorts before fitting the homography).
        """
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            x_px, y_px = self.undistort_points(
                [(x_px, y_px)], self.camera_matrix, self.dist_coeffs
            )[0]
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
    def scale_from_points(
        points_px: list[tuple[float, float]], spacing_mm: float = 1.0
    ) -> float:
        """Ruler calibration: pixels per millimetre from several clicked points.

        Consecutive points are assumed to be exactly ``spacing_mm`` apart (e.g.
        successive 1 mm ticks on a ruler placed in frame), so each gap's pixel
        length directly is that segment's px/mm; the result is the mean over
        all gaps, which averages out clicking jitter better than a single
        two-point measurement. Points need not be axis-aligned — only
        consecutive-gap spacing matters, not direction.

        Raises:
            CalibrationError: fewer than 2 points, or non-positive spacing.
        """
        if len(points_px) < 2:
            raise CalibrationError("Need at least 2 ruler points to compute a scale")
        if spacing_mm <= 0:
            raise CalibrationError("Ruler point spacing must be positive")
        segments = [
            math.hypot(x2 - x1, y2 - y1)
            for (x1, y1), (x2, y2) in zip(points_px, points_px[1:])
        ]
        return float(np.mean(segments)) / spacing_mm

    @staticmethod
    def calibrate_lens(
        views: list["CheckerboardDetection"], image_size: tuple[int, int]
    ) -> LensCalibrationResult:
        """Fit a camera matrix + distortion coefficients from several
        checkerboard photos of the *same* board at different poses/tilts.

        Moving the board between shots is what makes this different from
        (and more accurate than) a single-photo homography: a lone view can't
        distinguish true lens distortion from perspective, but several views
        of the same rigid board, seen from different angles, let
        ``cv2.calibrateCamera`` solve for both independently.

        Args:
            views: checkerboard detections from :meth:`find_checkerboard`, all
                using the same board (same corner count and square size).
            image_size: ``(width, height)`` of the frames the views came from.

        Raises:
            CalibrationError: fewer than 3 views, mismatched board sizes
                across views, or OpenCV's solver fails (degenerate/too-similar
                poses).
        """
        if len(views) < 3:
            raise CalibrationError(
                "Lens calibration needs >= 3 checkerboard views at different "
                f"poses (got {len(views)})"
            )
        point_count = len(views[0].mm_points)
        if any(len(view.mm_points) != point_count for view in views):
            raise CalibrationError("All views must use the same checkerboard size")

        object_points = [
            np.array([(x, y, 0.0) for x, y in view.mm_points], dtype=np.float32)
            for view in views
        ]
        image_points = [
            view.corners_px.astype(np.float32).reshape(-1, 1, 2) for view in views
        ]

        try:
            overall_rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, image_size, None, None
            )
        except cv2.error as exc:
            raise CalibrationError(f"Lens calibration failed: {exc}") from exc
        if camera_matrix is None:
            raise CalibrationError("Lens calibration failed (degenerate views?)")

        per_view_rms: list[float] = []
        for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
            projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
            residuals = np.linalg.norm(
                projected.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1
            )
            per_view_rms.append(float(np.sqrt(np.mean(residuals**2))))

        return LensCalibrationResult(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            overall_rms_px=float(overall_rms),
            per_view_rms_px=per_view_rms,
        )

    @staticmethod
    def undistort_points(
        points_px: list[tuple[float, float]],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Map raw (lens-distorted) pixel coordinates to the equivalent ideal
        pinhole coordinates, in the same pixel units — ``P=camera_matrix``
        keeps the output on the original pixel scale rather than normalized
        camera coordinates, so callers can keep treating it as a pixel."""
        points = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
        return [(float(x), float(y)) for x, y in undistorted.reshape(-1, 2)]

    @staticmethod
    def _subpix_window(
        corners_px: np.ndarray, rows: int, columns: int, scale: float
    ) -> tuple[int, int]:
        """Half-window for :func:`cv2.cornerSubPix`, sized for a seed found at
        ``scale``.

        Corners located on a copy downscaled by ``scale`` and multiplied back up
        land within roughly ``1/scale`` full-resolution pixels of the true
        corner — at this station's 5496 px sensor screened at 1000 px that is
        ~5 px, well outside the fixed 11 px window OpenCV is normally given, so
        the refinement would polish the wrong spot or wander to a neighbour.
        The window therefore grows with the downscale factor, but is never
        allowed past a third of the measured corner spacing: past that it can
        swallow the adjacent corner and the refinement snaps to it.

        ``scale == 1.0`` returns the conventional 11 px window unchanged.
        """
        if scale >= 1.0:
            return (11, 11)
        grid = corners_px.reshape(rows, columns, 2)
        spacings = [
            float(np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2).min()),
            float(np.linalg.norm(grid[1:, :, :] - grid[:-1, :, :], axis=2).min()),
        ]
        ceiling = max(5, int(min(spacings) / 3.0))
        half = min(int(math.ceil(1.0 / scale)) + 6, ceiling)
        return (max(half, 5),) * 2

    @staticmethod
    def find_checkerboard(
        image: np.ndarray,
        columns: int,
        rows: int,
        square_size_mm: float,
        detect_max_dim: int | None = None,
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

        ``detect_max_dim`` bounds the longest edge the *search* runs on: the
        image is downscaled to it for ``findChessboardCorners`` (which costs
        1-2 s at 12-20 MP and ~0.15 s at 1000 px, with no loss in the
        board-is-here decision), the corners are scaled back up, and the
        sub-pixel refinement then runs against the **full-resolution** pixels.
        So the correspondences — and therefore every calibration fitted from
        them — remain in the actual image's coordinate basis; only the coarse
        search is cheapened. ``None`` (default) searches at full resolution.

        Raises:
            CalibrationError: bad arguments, or no board of this size found
                (check the pattern size, lighting, and that it is fully
                visible and reasonably flat in the frame).
        """
        if columns < 2 or rows < 2:
            raise CalibrationError("Checkerboard columns/rows must each be >= 2")
        if square_size_mm <= 0:
            raise CalibrationError("Checkerboard square size must be positive")

        if detect_max_dim is not None and detect_max_dim < 2:
            raise CalibrationError("Checkerboard detect_max_dim must be >= 2")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        longest = max(gray.shape[:2])
        scale = 1.0
        if detect_max_dim is not None and longest > detect_max_dim:
            scale = detect_max_dim / longest
        search = (
            cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else gray
        )
        found, corners = cv2.findChessboardCorners(
            search, (columns, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            raise CalibrationError(
                f"No {columns}x{rows}-corner checkerboard found — check the "
                f"pattern size matches the physical board, and that it is "
                f"fully visible, flat, and well lit"
            )
        if scale < 1.0:
            corners = (corners / scale).astype(np.float32)
        corners = cv2.cornerSubPix(
            gray, corners, CameraCalibration._subpix_window(corners, rows, columns, scale),
            (-1, -1),
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
    @staticmethod
    def _matrix_from_json(
        raw: str | None, camera_index: int, label: str, expected_shape: tuple[int, int] | None
    ) -> np.ndarray | None:
        if not raw:
            return None
        try:
            matrix = np.asarray(json.loads(raw), dtype=np.float64)
            if expected_shape is not None and matrix.shape != expected_shape:
                raise ValueError(f"expected {expected_shape}, got {matrix.shape}")
        except (ValueError, TypeError) as exc:
            raise CalibrationError(
                f"Corrupt {label} for camera {camera_index}: {exc}"
            ) from exc
        return matrix

    @classmethod
    def from_row(cls, row: CalibrationRow) -> "CameraCalibration":
        homography = cls._matrix_from_json(
            row.homography_json, row.camera_index, "homography", (3, 3)
        )
        camera_matrix = cls._matrix_from_json(
            row.camera_matrix_json, row.camera_index, "camera matrix", (3, 3)
        )
        dist_coeffs = cls._matrix_from_json(
            row.dist_coeffs_json, row.camera_index, "distortion coefficients", None
        )
        return cls(
            camera_index=row.camera_index,
            pixels_per_mm_x=row.pixels_per_mm_x,
            pixels_per_mm_y=row.pixels_per_mm_y,
            homography=homography,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
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
            camera_matrix_json=(
                json.dumps(self.camera_matrix.tolist())
                if self.camera_matrix is not None
                else None
            ),
            dist_coeffs_json=(
                json.dumps(self.dist_coeffs.tolist()) if self.dist_coeffs is not None else None
            ),
            ref_point_x_mm=self.ref_point_mm[0],
            ref_point_y_mm=self.ref_point_mm[1],
            rms_error=self.rms_error,
            calibrated_by=self.calibrated_by,
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON-safe representation (nested lists, not the DB row's
        JSON-*strings*) — for embedding into a non-DB document, e.g. a
        machine-model profile snapshot (see ``MachineModelService``)."""
        return {
            "pixels_per_mm_x": self.pixels_per_mm_x,
            "pixels_per_mm_y": self.pixels_per_mm_y,
            "homography": self.homography.tolist() if self.homography is not None else None,
            "camera_matrix": (
                self.camera_matrix.tolist() if self.camera_matrix is not None else None
            ),
            "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            "ref_point_mm": list(self.ref_point_mm),
            "rms_error": self.rms_error,
            "calibrated_by": self.calibrated_by,
        }

    @classmethod
    def from_dict(cls, camera_index: int, data: dict[str, Any]) -> "CameraCalibration":
        """Inverse of :meth:`to_dict`.

        Raises:
            CalibrationError: a matrix field has the wrong shape.
        """

        def _matrix(key: str, expected_shape: tuple[int, int] | None) -> np.ndarray | None:
            value = data.get(key)
            if value is None:
                return None
            matrix = np.asarray(value, dtype=np.float64)
            if expected_shape is not None and matrix.shape != expected_shape:
                raise CalibrationError(
                    f"Camera {camera_index}: {key} expected shape {expected_shape}, "
                    f"got {matrix.shape}"
                )
            return matrix

        ref_point = data.get("ref_point_mm", (0.0, 0.0))
        return cls(
            camera_index=camera_index,
            pixels_per_mm_x=float(data.get("pixels_per_mm_x", 1.0)),
            pixels_per_mm_y=float(data.get("pixels_per_mm_y", 1.0)),
            homography=_matrix("homography", (3, 3)),
            camera_matrix=_matrix("camera_matrix", (3, 3)),
            dist_coeffs=_matrix("dist_coeffs", None),
            ref_point_mm=(float(ref_point[0]), float(ref_point[1])),
            rms_error=float(data.get("rms_error", 0.0)),
            calibrated_by=str(data.get("calibrated_by", "")),
        )

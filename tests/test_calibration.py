"""Calibration model: scaling, homography, reference deviation, persistence."""

import cv2
import numpy as np
import pytest

from core.calibration import CameraCalibration
from core.utilities.exceptions import CalibrationError


def make_checkerboard(
    square_px: int = 60, squares_x: int = 9, squares_y: int = 6, margin: int = 60
) -> np.ndarray:
    """A flat, head-on synthetic checkerboard: squares_x*squares_y squares,
    each square_px pixels wide, so pixels-per-unit is known exactly."""
    width = squares_x * square_px + 2 * margin
    height = squares_y * square_px + 2 * margin
    image = np.full((height, width), 255, dtype=np.uint8)
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                y0, x0 = margin + row * square_px, margin + col * square_px
                image[y0 : y0 + square_px, x0 : x0 + square_px] = 0
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_scale_from_distance() -> None:
    assert CameraCalibration.scale_from_distance(100.0, 10.0) == pytest.approx(10.0)
    with pytest.raises(CalibrationError):
        CameraCalibration.scale_from_distance(0.0, 10.0)


def test_pixel_to_mm_scale_and_deviation() -> None:
    calibration = CameraCalibration(
        camera_index=1,
        pixels_per_mm_x=10.0,
        pixels_per_mm_y=8.0,
        ref_point_mm=(50.0, 60.0),
    )
    x_mm, y_mm = calibration.pixel_to_mm(500.0, 480.0)
    assert (x_mm, y_mm) == (50.0, 60.0)
    assert calibration.deviation_mm(x_mm, y_mm) == pytest.approx(0.0)
    assert calibration.deviation_mm(53.0, 64.0) == pytest.approx(5.0)


def test_homography_exact_fit() -> None:
    pixel_points = [(100, 90), (1180, 120), (1150, 940), (130, 900)]
    mm_points = [(0, 0), (100, 0), (100, 80), (0, 80)]
    homography, rms = CameraCalibration.compute_homography(pixel_points, mm_points)
    assert rms < 1e-6
    calibration = CameraCalibration(camera_index=2, homography=homography)
    corner = calibration.pixel_to_mm(1180, 120)
    assert corner[0] == pytest.approx(100.0, abs=1e-6)
    assert corner[1] == pytest.approx(0.0, abs=1e-6)


def test_homography_needs_four_points() -> None:
    with pytest.raises(CalibrationError):
        CameraCalibration.compute_homography([(0, 0)], [(0, 0)])


def test_find_checkerboard_detects_grid_and_scale() -> None:
    square_px = 60
    image = make_checkerboard(square_px=square_px, squares_x=9, squares_y=6)
    detection = CameraCalibration.find_checkerboard(image, columns=8, rows=5, square_size_mm=20.0)

    assert len(detection.pixel_points) == 8 * 5
    assert len(detection.mm_points) == 8 * 5
    expected_ppmm = square_px / 20.0
    assert detection.pixels_per_mm_x == pytest.approx(expected_ppmm, rel=0.05)
    assert detection.pixels_per_mm_y == pytest.approx(expected_ppmm, rel=0.05)

    homography, rms = CameraCalibration.compute_homography(
        detection.pixel_points, detection.mm_points
    )
    assert rms < 0.5  # sub-pixel-consistent fit on a flat, head-on synthetic board
    calibration = CameraCalibration(camera_index=1, homography=homography)
    # opposite corners of the mm grid should map back close to what we asked for
    x_mm, y_mm = calibration.pixel_to_mm(*detection.pixel_points[-1])
    assert x_mm == pytest.approx(7 * 20.0, abs=1.0)
    assert y_mm == pytest.approx(4 * 20.0, abs=1.0)


def test_find_checkerboard_rejects_missing_board() -> None:
    blank = np.full((400, 500, 3), 200, dtype=np.uint8)
    with pytest.raises(CalibrationError):
        CameraCalibration.find_checkerboard(blank, columns=8, rows=5, square_size_mm=20.0)


def test_find_checkerboard_rejects_bad_arguments() -> None:
    image = make_checkerboard()
    with pytest.raises(CalibrationError):
        CameraCalibration.find_checkerboard(image, columns=1, rows=5, square_size_mm=20.0)
    with pytest.raises(CalibrationError):
        CameraCalibration.find_checkerboard(image, columns=8, rows=5, square_size_mm=0.0)


def test_row_round_trip() -> None:
    pixel_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
    mm_points = [(0, 0), (10, 0), (10, 10), (0, 10)]
    homography, rms = CameraCalibration.compute_homography(pixel_points, mm_points)
    original = CameraCalibration(
        camera_index=3,
        pixels_per_mm_x=10.0,
        pixels_per_mm_y=10.0,
        homography=homography,
        ref_point_mm=(5.0, 5.0),
        rms_error=rms,
        calibrated_by="tester",
    )
    restored = CameraCalibration.from_row(original.to_row())
    assert restored.pixel_to_mm(50, 50) == pytest.approx(original.pixel_to_mm(50, 50))
    assert restored.ref_point_mm == original.ref_point_mm

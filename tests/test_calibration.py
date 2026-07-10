"""Calibration model: scaling, homography, reference deviation, persistence."""

import pytest

from core.calibration import CameraCalibration
from core.utilities.exceptions import CalibrationError


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

"""Calibration model: scaling, homography, reference deviation, persistence."""

import cv2
import numpy as np
import pytest

from core.calibration import CameraCalibration, CheckerboardDetection
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


def test_scale_from_points_averages_segments() -> None:
    # Slight jitter around an exact 10 px/mm spacing — the mean should still
    # land very close to 10.0, which a single two-point read couldn't average out.
    points = [(0.0, 0.0), (10.2, 0.0), (19.8, 0.0), (30.1, 0.0)]
    assert CameraCalibration.scale_from_points(points, spacing_mm=1.0) == pytest.approx(
        10.0, abs=0.2
    )


def test_scale_from_points_ignores_direction() -> None:
    # A diagonal ruler: only consecutive-gap spacing matters, not axis alignment.
    points = [(0.0, 0.0), (6.0, 8.0), (12.0, 16.0)]  # each gap is 10 px
    assert CameraCalibration.scale_from_points(points, spacing_mm=2.0) == pytest.approx(5.0)


def test_scale_from_points_rejects_too_few_points() -> None:
    with pytest.raises(CalibrationError):
        CameraCalibration.scale_from_points([(0.0, 0.0)], spacing_mm=1.0)


def test_scale_from_points_rejects_non_positive_spacing() -> None:
    with pytest.raises(CalibrationError):
        CameraCalibration.scale_from_points([(0.0, 0.0), (10.0, 0.0)], spacing_mm=0.0)


def _synthetic_checkerboard_views(
    true_camera_matrix: np.ndarray, true_dist_coeffs: np.ndarray, rows: int, columns: int
) -> list[CheckerboardDetection]:
    square_mm = 20.0
    mm_points = [(c * square_mm, r * square_mm) for r in range(rows) for c in range(columns)]
    object_points = np.array([(x, y, 0.0) for x, y in mm_points], dtype=np.float64)

    # A handful of plausible board poses (rotation in degrees, translation in mm).
    poses = [
        (0.0, 0.0, 0.0, 0.0, 0.0, 400.0),
        (10.0, 5.0, 0.0, 20.0, -10.0, 420.0),
        (-8.0, 12.0, 5.0, -30.0, 15.0, 380.0),
        (5.0, -15.0, 0.0, 10.0, 30.0, 410.0),
    ]
    views = []
    for rx, ry, rz, tx, ty, tz in poses:
        rvec = np.radians([rx, ry, rz]).reshape(3, 1)
        tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, true_camera_matrix, true_dist_coeffs
        )
        views.append(
            CheckerboardDetection(
                pixel_points=[(float(x), float(y)) for x, y in projected.reshape(-1, 2)],
                mm_points=mm_points,
                pixels_per_mm_x=0.0,
                pixels_per_mm_y=0.0,
                corners_px=projected.reshape(-1, 1, 2).astype(np.float32),
            )
        )
    return views


def test_calibrate_lens_rejects_too_few_views() -> None:
    views = _synthetic_checkerboard_views(
        np.eye(3), np.zeros(5), rows=5, columns=8
    )[:2]
    with pytest.raises(CalibrationError):
        CameraCalibration.calibrate_lens(views, image_size=(1280, 960))


def test_calibrate_lens_rejects_mismatched_board_sizes() -> None:
    views = _synthetic_checkerboard_views(np.eye(3), np.zeros(5), rows=5, columns=8)
    views[0] = CheckerboardDetection(
        pixel_points=views[0].pixel_points[:-1],
        mm_points=views[0].mm_points[:-1],
        pixels_per_mm_x=0.0,
        pixels_per_mm_y=0.0,
        corners_px=views[0].corners_px[:-1],
    )
    with pytest.raises(CalibrationError):
        CameraCalibration.calibrate_lens(views, image_size=(1280, 960))


def test_calibrate_lens_recovers_known_camera_and_distortion() -> None:
    true_camera_matrix = np.array(
        [[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]]
    )
    true_dist_coeffs = np.array([-0.15, 0.05, 0.0, 0.0, 0.0])
    views = _synthetic_checkerboard_views(true_camera_matrix, true_dist_coeffs, rows=6, columns=8)

    result = CameraCalibration.calibrate_lens(views, image_size=(1280, 960))

    assert result.overall_rms_px < 1.0
    assert len(result.per_view_rms_px) == len(views)
    assert result.camera_matrix[0, 0] == pytest.approx(true_camera_matrix[0, 0], rel=0.05)
    assert result.camera_matrix[1, 1] == pytest.approx(true_camera_matrix[1, 1], rel=0.05)
    assert result.dist_coeffs.reshape(-1)[0] == pytest.approx(true_dist_coeffs[0], abs=0.05)


def test_undistort_points_is_identity_under_zero_distortion() -> None:
    camera_matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.zeros(5)
    result = CameraCalibration.undistort_points([(400.0, 300.0)], camera_matrix, dist_coeffs)
    assert result[0] == pytest.approx((400.0, 300.0), abs=1e-6)


def test_pixel_to_mm_undistorts_before_scaling() -> None:
    camera_matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([-0.2, 0.05, 0.0, 0.0, 0.0])
    calibration = CameraCalibration(
        camera_index=1,
        pixels_per_mm_x=10.0,
        pixels_per_mm_y=10.0,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    x_mm, y_mm = calibration.pixel_to_mm(400.0, 300.0)
    expected_x_px, expected_y_px = CameraCalibration.undistort_points(
        [(400.0, 300.0)], camera_matrix, dist_coeffs
    )[0]
    assert (x_mm, y_mm) == pytest.approx((expected_x_px / 10.0, expected_y_px / 10.0))


def test_pixel_to_mm_zero_distortion_is_a_no_op() -> None:
    camera_matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    calibration = CameraCalibration(
        camera_index=1,
        pixels_per_mm_x=10.0,
        pixels_per_mm_y=10.0,
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros(5),
    )
    assert calibration.pixel_to_mm(400.0, 300.0) == pytest.approx((40.0, 30.0), abs=1e-6)


def test_row_round_trip_with_lens_distortion() -> None:
    camera_matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([-0.2, 0.05, 0.0, 0.0, 0.0])
    original = CameraCalibration(
        camera_index=4,
        pixels_per_mm_x=10.0,
        pixels_per_mm_y=10.0,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    restored = CameraCalibration.from_row(original.to_row())
    assert restored.camera_matrix == pytest.approx(camera_matrix)
    assert restored.dist_coeffs == pytest.approx(dist_coeffs)
    assert restored.pixel_to_mm(400.0, 300.0) == pytest.approx(original.pixel_to_mm(400.0, 300.0))


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

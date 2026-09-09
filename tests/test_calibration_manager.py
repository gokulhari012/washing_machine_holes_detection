"""CalibrationManager.evaluate: pixel->mm conversion re-based to image centre.

The repository is never touched here — a stub with just enough surface for
``CalibrationManager.__init__`` covers it, and calibrations are seeded
directly into the cache, matching how the manager's own ``load_all`` would
populate it.
"""

from types import SimpleNamespace

import pytest

from core.calibration.calibration_manager import CalibrationManager
from core.calibration.calibration_model import CameraCalibration


def make_manager(calibration: CameraCalibration | None) -> CalibrationManager:
    manager = CalibrationManager(SimpleNamespace(get_all_active=lambda: {}))
    if calibration is not None:
        manager._calibrations[calibration.camera_index] = calibration
    return manager


def test_evaluate_without_image_dims_returns_raw_pixel_to_mm() -> None:
    calibration = CameraCalibration(camera_index=1, pixels_per_mm_x=10.0, pixels_per_mm_y=10.0)
    manager = make_manager(calibration)
    x_mm, y_mm, deviation = manager.evaluate(1, 500.0, 400.0)
    assert (x_mm, y_mm) == pytest.approx((50.0, 40.0))
    assert deviation == pytest.approx((50.0**2 + 40.0**2) ** 0.5)


def test_evaluate_with_image_dims_rebases_to_image_center() -> None:
    calibration = CameraCalibration(camera_index=1, pixels_per_mm_x=10.0, pixels_per_mm_y=10.0)
    manager = make_manager(calibration)
    # a 1000x800 image's centre pixel (500, 400) -> (50, 40) mm raw -> (0, 0) rebased
    x_mm, y_mm, _ = manager.evaluate(1, 500.0, 400.0, image_width=1000, image_height=800)
    assert (x_mm, y_mm) == pytest.approx((0.0, 0.0))

    # a hole 100px right / 80px below that centre pixel -> +10mm / -8mm from centre
    # (Y is flipped on re-basing: below centre is negative, up is positive)
    x_mm, y_mm, _ = manager.evaluate(1, 600.0, 480.0, image_width=1000, image_height=800)
    assert (x_mm, y_mm) == pytest.approx((10.0, -8.0))

    # ...and a hole above the centre reports positive y
    x_mm, y_mm, _ = manager.evaluate(1, 600.0, 320.0, image_width=1000, image_height=800)
    assert (x_mm, y_mm) == pytest.approx((10.0, 8.0))


def test_evaluate_rebase_does_not_shift_deviation_or_tolerance_judgement() -> None:
    calibration = CameraCalibration(
        camera_index=1, pixels_per_mm_x=10.0, pixels_per_mm_y=10.0, ref_point_mm=(50.0, 40.0)
    )
    manager = make_manager(calibration)
    _, _, deviation_rebased = manager.evaluate(1, 500.0, 400.0, image_width=1000, image_height=800)
    _, _, deviation_raw = manager.evaluate(1, 500.0, 400.0)
    assert deviation_rebased == pytest.approx(0.0)
    assert deviation_raw == pytest.approx(0.0)


def test_evaluate_uncalibrated_camera_falls_back_to_identity_rebased_to_center() -> None:
    manager = make_manager(None)
    x_mm, y_mm, deviation = manager.evaluate(9, 500.0, 400.0, image_width=1000, image_height=800)
    assert (x_mm, y_mm) == pytest.approx((0.0, 0.0))
    assert deviation is None

    # identity fallback flips Y the same way: below centre reads negative
    x_mm, y_mm, _ = manager.evaluate(9, 600.0, 480.0, image_width=1000, image_height=800)
    assert (x_mm, y_mm) == pytest.approx((100.0, -80.0))

    x_mm, y_mm, deviation = manager.evaluate(9, 500.0, 400.0)
    assert (x_mm, y_mm) == (500.0, 400.0)
    assert deviation is None


def test_apply_live_updates_cache_without_touching_repository() -> None:
    """Used when a machine-model profile switches (MachineModelService):
    the new calibration must be visible immediately, but never written to
    the database — the repository fake here has no `save`, so any attempt
    to persist would raise AttributeError."""
    manager = make_manager(None)
    calibration = CameraCalibration(camera_index=1, pixels_per_mm_x=12.5, pixels_per_mm_y=12.5)
    manager.apply_live(calibration)
    assert manager.get(1) is calibration


def test_apply_live_overwrites_only_the_matching_camera() -> None:
    manager = make_manager(
        CameraCalibration(camera_index=1, pixels_per_mm_x=10.0, pixels_per_mm_y=10.0)
    )
    replacement = CameraCalibration(camera_index=1, pixels_per_mm_x=20.0, pixels_per_mm_y=20.0)
    manager.apply_live(replacement)
    assert manager.get(1) is replacement


# ------------------------------------------------------------------ observers
# save() fans out to plain callables so the composition root can fold a freshly
# persisted calibration into the active machine-model profile; apply_live()
# deliberately does not, since a model switch pushing its own snapshot back
# into the cache is not the operator calibrating anything.


def _saving_manager() -> CalibrationManager:
    return CalibrationManager(
        SimpleNamespace(get_all_active=lambda: {}, save=lambda row: 1)
    )


def test_save_notifies_observers_with_the_persisted_calibration() -> None:
    manager = _saving_manager()
    seen: list[CameraCalibration] = []
    manager.subscribe(seen.append)

    calibration = CameraCalibration(camera_index=2, pixels_per_mm_x=8.0)
    manager.save(calibration)

    assert seen == [calibration]


def test_apply_live_does_not_notify_observers() -> None:
    manager = _saving_manager()
    seen: list[CameraCalibration] = []
    manager.subscribe(seen.append)

    manager.apply_live(CameraCalibration(camera_index=2, pixels_per_mm_x=8.0))

    assert seen == []


def test_failing_observer_never_breaks_the_save() -> None:
    manager = _saving_manager()
    reached: list[int] = []
    manager.subscribe(lambda _c: (_ for _ in ()).throw(RuntimeError("boom")))
    manager.subscribe(lambda c: reached.append(c.camera_index))

    assert manager.save(CameraCalibration(camera_index=2, pixels_per_mm_x=8.0)) == 1
    assert reached == [2]  # the later observer still ran
    assert manager.get(2) is not None  # ...and the cache was updated


def test_unsubscribe_stops_notifications_and_tolerates_an_unknown_callback() -> None:
    manager = _saving_manager()
    seen: list[CameraCalibration] = []

    manager.unsubscribe(seen.append)  # never registered -> no-op, not an error
    manager.subscribe(seen.append)
    manager.unsubscribe(seen.append)
    manager.save(CameraCalibration(camera_index=2, pixels_per_mm_x=8.0))

    assert seen == []

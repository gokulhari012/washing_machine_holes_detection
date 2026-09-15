"""Screw driver position compensation: stored on the camera's calibration,
applied to the PLC-bound position only, and live the moment it is saved.

The feature used to be a per-machine-model block pushed into the manager by
``MachineModelService.apply_profile``; it now rides the calibration row next
to the axis signs, so "Save Compensation" on the Calibration page has to take
effect immediately rather than waiting for a model switch — that immediacy is
what most of these tests pin.
"""

from types import SimpleNamespace

import pytest

from core.calibration.calibration_manager import CalibrationManager
from core.calibration.calibration_model import CameraCalibration


class FakeRepository:
    """Just enough of ``CalibrationRepository`` for the manager."""

    def __init__(self, active: set[int]) -> None:
        self._active = active
        self.writes: list[tuple[int, bool, float, float]] = []

    def get_all_active(self) -> dict:
        return {}

    def update_screw_compensation(
        self, camera_index: int, enabled: bool, x_mm: float, y_mm: float
    ) -> bool:
        if camera_index not in self._active:
            return False
        self.writes.append((camera_index, enabled, x_mm, y_mm))
        return True


def make_manager(*calibrations: CameraCalibration) -> tuple[CalibrationManager, FakeRepository]:
    repository = FakeRepository({c.camera_index for c in calibrations})
    manager = CalibrationManager(repository)
    for calibration in calibrations:
        manager._calibrations[calibration.camera_index] = calibration
    return manager, repository


def test_offset_is_zero_until_enabled() -> None:
    manager, _ = make_manager(
        CameraCalibration(camera_index=1, screw_offset_x_mm=-20.0, screw_offset_y_mm=200.0)
    )
    assert manager.screw_offset(1) == (0.0, 0.0)


def test_enabled_offset_is_reported() -> None:
    manager, _ = make_manager(
        CameraCalibration(
            camera_index=1,
            screw_compensation_enabled=True,
            screw_offset_x_mm=-20.0,
            screw_offset_y_mm=200.0,
        )
    )
    assert manager.screw_offset(1) == (-20.0, 200.0)


def test_uncalibrated_camera_has_no_offset() -> None:
    manager, _ = make_manager()
    assert manager.screw_offset(7) == (0.0, 0.0)


def test_save_persists_and_takes_effect_in_the_same_call() -> None:
    """The point of the feature's move: no model switch, no restart."""
    manager, repository = make_manager(CameraCalibration(camera_index=2))
    assert manager.save_screw_compensation(2, True, 1.5, -2.5) is True
    assert manager.screw_offset(2) == (1.5, -2.5)
    assert repository.writes == [(2, True, 1.5, -2.5)]
    # ...and the cached calibration carries it, so a later full save keeps it
    assert manager.get(2).screw_offset_x_mm == pytest.approx(1.5)


def test_save_disabling_clears_the_offset_reported() -> None:
    manager, _ = make_manager(
        CameraCalibration(
            camera_index=2,
            screw_compensation_enabled=True,
            screw_offset_x_mm=1.5,
            screw_offset_y_mm=-2.5,
        )
    )
    manager.save_screw_compensation(2, False, 1.5, -2.5)
    assert manager.screw_offset(2) == (0.0, 0.0)
    # the values survive the switch-off, ready to be re-enabled
    assert manager.get(2).screw_offset_x_mm == pytest.approx(1.5)


def test_save_refuses_a_camera_with_no_calibration() -> None:
    """An identity row invented here would silently enable that camera's
    tolerance judgement at 1 px = 1 mm — see the repository's docstring."""
    manager, repository = make_manager(CameraCalibration(camera_index=1))
    assert manager.save_screw_compensation(3, True, 5.0, 5.0) is False
    assert repository.writes == []
    assert manager.screw_offset(3) == (0.0, 0.0)


def test_save_does_not_notify_calibration_observers() -> None:
    """``subscribe`` means "the operator calibrated this camera", which folds
    into the active machine model — an offset is not that."""
    manager, _ = make_manager(CameraCalibration(camera_index=1))
    seen: list[CameraCalibration] = []
    manager.subscribe(seen.append)
    manager.save_screw_compensation(1, True, 1.0, 1.0)
    assert seen == []


def test_a_machine_model_switch_keeps_the_live_offset() -> None:
    """``apply_live`` pushes a profile's calibration, which never carries the
    offsets (``to_dict``) — taking its defaults would zero a rig fact."""
    manager, _ = make_manager(
        CameraCalibration(
            camera_index=1,
            screw_compensation_enabled=True,
            screw_offset_x_mm=-20.0,
            screw_offset_y_mm=200.0,
        )
    )
    incoming = CameraCalibration.from_dict(
        1, CameraCalibration(camera_index=1, pixels_per_mm_x=8.0).to_dict()
    )
    manager.apply_live(incoming)
    assert manager.screw_offset(1) == (-20.0, 200.0)


def test_profile_snapshot_never_carries_the_offsets() -> None:
    snapshot = CameraCalibration(
        camera_index=1,
        screw_compensation_enabled=True,
        screw_offset_x_mm=-20.0,
        screw_offset_y_mm=200.0,
    ).to_dict()
    assert "screw_offset_x_mm" not in snapshot
    assert "screw_compensation_enabled" not in snapshot


def test_row_round_trip_keeps_the_offsets() -> None:
    original = CameraCalibration(
        camera_index=4,
        screw_compensation_enabled=True,
        screw_offset_x_mm=3.25,
        screw_offset_y_mm=-4.75,
    )
    restored = CameraCalibration.from_row(original.to_row())
    assert restored.screw_compensation_enabled is True
    assert restored.screw_offset_x_mm == pytest.approx(3.25)
    assert restored.screw_offset_y_mm == pytest.approx(-4.75)


def test_a_legacy_row_predating_the_columns_reads_as_disabled() -> None:
    row = SimpleNamespace(
        camera_index=1,
        pixels_per_mm_x=1.0,
        pixels_per_mm_y=1.0,
        homography_json=None,
        camera_matrix_json=None,
        dist_coeffs_json=None,
        ref_point_x_mm=0.0,
        ref_point_y_mm=0.0,
        invert_x=None,
        invert_y=None,
        screw_compensation_enabled=None,
        screw_offset_x_mm=None,
        screw_offset_y_mm=None,
        rms_error=0.0,
        calibrated_by="",
    )
    calibration = CameraCalibration.from_row(row)
    assert calibration.screw_compensation_enabled is False
    assert (calibration.screw_offset_x_mm, calibration.screw_offset_y_mm) == (0.0, 0.0)

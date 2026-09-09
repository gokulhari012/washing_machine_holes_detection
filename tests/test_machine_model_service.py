"""MachineModelService: profile capture/apply, JSON-only persistence.

Camera application is exercised against a lightweight fake that mimics only
the CameraService methods the service actually calls (get_configs,
get_effective_configs, apply_live), validated the same way the real one is,
via CameraSettings.from_config, so a malformed merge would still be caught.
The fake keeps the persisted and live-effective views separate (see
``apply_live_override``) so the "snapshot what is running, not what the file
says" rule can be asserted rather than assumed.
Calibration application is exercised against a fake CalibrationManager that
mimics only get/apply_live, so an "applied live, not persisted" assertion can
check the fake's own state instead of a real database.
"""

import copy
import json

import pytest

from core.calibration import CameraCalibration
from core.camera.camera_base import CameraSettings
from core.utilities.config_manager import ConfigManager
from core.utilities.exceptions import ConfigurationError, VisionSystemError
from core.vision import VisionEngine
from services.machine_model_service import MachineModelService

CAMERA_DOC = {
    "cameras": [
        {
            "index": 1, "name": "Cam 1", "driver": "simulated", "connection_id": "",
            "enabled": True, "exposure_us": 10000, "gain_db": 0.0, "gamma": 1.0,
            "brightness": 0, "width": 1280, "height": 1024, "trigger_mode": "software",
            "roi": {"x": 0, "y": 0, "width": 0, "height": 0},
        },
    ]
}
DETECTION_DOC = {
    "cameras": {
        "1": {
            "active_detector": "opencv",
            "common": {
                "confidence_threshold": 0.6, "expected_hole_count": 1,
                "position_tolerance_mm": 0.0,
            },
            "opencv": {
                "detection_threshold": 60, "blur_kernel_size": 5,
                "morphology_operation": "close", "morphology_kernel_size": 5,
                "morphology_iterations": 1, "min_hole_diameter_px": 20,
                "max_hole_diameter_px": 200, "min_circularity": 0.7,
                "edge_threshold_low": 50, "edge_threshold_high": 150,
            },
            "dark_hole": {
                "channel": "auto", "blur_kernel_size": 3, "min_contrast": 18,
                "use_otsu": True, "morphology_kernel_size": 3, "min_hole_diameter_px": 15,
                "max_hole_diameter_px": 120, "min_fill_ratio": 0.35, "max_fit_error": 0.25,
            },
        },
    },
}


class FakeCameraService:
    """Mimics the CameraService methods MachineModelService calls."""

    def __init__(self, configs: list[dict]) -> None:
        self._configs = configs
        self._live: dict[int, dict] = {}  # index -> live-effective override
        self.applied: list[dict] = []

    def get_configs(self) -> list[dict]:
        """camera.json's persisted baseline."""
        return [dict(c) for c in self._configs]

    def get_effective_configs(self) -> list[dict]:
        """What the cameras are running — baseline with live overrides on top."""
        return [dict(c, **self._live.get(int(c["index"]), {})) for c in self._configs]

    def apply_live_override(self, index: int, **fields) -> None:
        """Stand in for a value pushed live but never written to camera.json."""
        self._live.setdefault(index, {}).update(fields)

    def apply_live(self, camera_config: dict) -> None:
        CameraSettings.from_config(camera_config)  # raises ConfigurationError if malformed
        self.applied.append(camera_config)


class FakePlcService:
    """Mimics the PlcService methods MachineModelService calls."""

    def __init__(self) -> None:
        self.model_select: int | None = None
        self.model_select_rejected = False

    def set_model_select(self, code: int) -> bool:
        if self.model_select_rejected:
            raise VisionSystemError("model_select register write rejected")
        self.model_select = code
        return True


class FakeCalibrationManager:
    """Mimics the two CalibrationManager methods MachineModelService calls."""

    def __init__(self) -> None:
        self._calibrations: dict[int, CameraCalibration] = {}
        self.applied: list[CameraCalibration] = []
        self.reject_camera: int | None = None
        self.screw_compensation: tuple[bool, dict[int, tuple[float, float]]] = (False, {})

    def seed(self, calibration: CameraCalibration) -> None:
        self._calibrations[calibration.camera_index] = calibration

    def get(self, camera_index: int) -> CameraCalibration | None:
        return self._calibrations.get(camera_index)

    def apply_live(self, calibration: CameraCalibration) -> None:
        if calibration.camera_index == self.reject_camera:
            raise VisionSystemError(f"camera {calibration.camera_index}: rejected")
        self.applied.append(calibration)

    def apply_screw_compensation(
        self, enabled: bool, positions: dict[int, tuple[float, float]]
    ) -> None:
        self.screw_compensation = (enabled, positions)


def make_service(tmp_path):
    config_dir = tmp_path / "config"
    (config_dir / "defaults").mkdir(parents=True)
    (config_dir / "camera.json").write_text(json.dumps(CAMERA_DOC))
    (config_dir / "detection.json").write_text(json.dumps(DETECTION_DOC))
    (config_dir / "machine_models.json").write_text(json.dumps({"profiles": []}))

    config = ConfigManager(config_dir)
    # Deep copies: tests that edit the fake's entries (standing in for a
    # settings change) must not leak into the next test through the shared
    # module-level documents.
    cameras = FakeCameraService(copy.deepcopy(CAMERA_DOC["cameras"]))
    engine = VisionEngine(copy.deepcopy(DETECTION_DOC))
    plc = FakePlcService()
    calibration = FakeCalibrationManager()
    return (
        MachineModelService(config, cameras, engine, plc, calibration),
        cameras, engine, plc, calibration,
    )


def test_capture_then_get_by_code_round_trip(tmp_path) -> None:
    service, _cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    assert profile["plc_code"] == 3
    assert service.get_by_code(3)["name"] == "Model A"
    assert service.get_by_id(profile["id"]) is not None
    assert service.list_profiles() == [profile]


def test_duplicate_plc_code_rejected(tmp_path) -> None:
    service, _cameras, _engine, _plc, _calibration = make_service(tmp_path)
    service.capture_current("Model A", 3, created_by="admin")
    with pytest.raises(ConfigurationError):
        service.capture_current("Model B", 3, created_by="admin")


def test_blank_name_rejected(tmp_path) -> None:
    service, _cameras, _engine, _plc, _calibration = make_service(tmp_path)
    with pytest.raises(ConfigurationError):
        service.capture_current("   ", 1, created_by="admin")


def test_apply_profile_merges_tunable_fields_only(tmp_path) -> None:
    service, cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    profile["cameras"]["1"]["roi"] = {"x": 10, "y": 20, "width": 300, "height": 200}
    profile["cameras"]["1"]["exposure_us"] = 25000

    warnings = service.apply_profile(profile)
    assert warnings == []
    assert len(cameras.applied) == 1
    applied = cameras.applied[0]
    assert applied["roi"] == {"x": 10, "y": 20, "width": 300, "height": 200}
    assert applied["exposure_us"] == 25000
    assert applied["driver"] == "simulated"  # identity field untouched by the profile
    assert applied["connection_id"] == ""


def test_apply_profile_skips_missing_camera_with_warning(tmp_path) -> None:
    service, cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    profile["cameras"]["9"] = profile["cameras"].pop("1")

    warnings = service.apply_profile(profile)
    assert len(warnings) == 1
    assert "camera 9" in warnings[0]
    assert cameras.applied == []


def test_apply_profile_hot_swaps_detection(tmp_path) -> None:
    service, _cameras, engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    profile["detection"]["cameras"]["1"]["active_detector"] = "dark_hole"

    service.apply_profile(profile)
    assert engine.active_detector_name(1) == "dark_hole"


def test_apply_profile_raises_on_bad_detection_block(tmp_path) -> None:
    service, _cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    profile["detection"]["cameras"]["1"]["active_detector"] = "not_a_real_detector"
    with pytest.raises(VisionSystemError):
        service.apply_profile(profile)


def test_apply_profile_upgrades_legacy_flat_detection_block(tmp_path) -> None:
    """A profile captured before per-camera detection existed still applies,
    its one shared block cloned onto every camera currently configured."""
    service, _cameras, engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    profile["detection"] = dict(DETECTION_DOC["cameras"]["1"], active_detector="dark_hole")

    service.apply_profile(profile)
    assert engine.active_detector_name(1) == "dark_hole"


def test_update_from_current_keeps_name_and_code(tmp_path) -> None:
    service, cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    cameras._configs[0]["exposure_us"] = 99999

    updated = service.update_from_current(profile["id"], updated_by="admin2")
    assert updated["name"] == "Model A"
    assert updated["plc_code"] == 3
    assert updated["cameras"]["1"]["exposure_us"] == 99999


def test_rename_validates_and_persists(tmp_path) -> None:
    service, _cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    service.rename(profile["id"], "Model A2", 4)
    assert service.get_by_code(4)["name"] == "Model A2"
    assert service.get_by_code(3) is None


def test_rename_rejects_code_already_used_by_another_profile(tmp_path) -> None:
    service, _cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile_a = service.capture_current("Model A", 3, created_by="admin")
    service.capture_current("Model B", 4, created_by="admin")
    with pytest.raises(ConfigurationError):
        service.rename(profile_a["id"], "Model A", 4)


def test_delete_removes_profile(tmp_path) -> None:
    service, _cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    service.delete(profile["id"])
    assert service.list_profiles() == []
    with pytest.raises(ConfigurationError):
        service.delete(profile["id"])


def test_apply_profile_writes_model_select_register(tmp_path) -> None:
    service, _cameras, _engine, plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")

    warnings = service.apply_profile(profile)
    assert warnings == []
    assert plc.model_select == 3


def test_apply_profile_warns_when_model_select_rejected(tmp_path) -> None:
    service, _cameras, _engine, plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    plc.model_select_rejected = True

    warnings = service.apply_profile(profile)
    assert len(warnings) == 1
    assert "model_select" in warnings[0]


def test_capture_snapshots_active_calibration(tmp_path) -> None:
    service, _cameras, _engine, _plc, calibration = make_service(tmp_path)
    calibration.seed(
        CameraCalibration(camera_index=1, pixels_per_mm_x=12.5, pixels_per_mm_y=12.5)
    )
    profile = service.capture_current("Model A", 3, created_by="admin")
    assert profile["calibration"]["1"]["pixels_per_mm_x"] == 12.5


def test_capture_omits_calibration_for_uncalibrated_camera(tmp_path) -> None:
    service, _cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    assert profile["calibration"] == {}


def test_apply_profile_applies_calibration_live_without_persisting(tmp_path) -> None:
    service, _cameras, _engine, _plc, calibration = make_service(tmp_path)
    calibration.seed(
        CameraCalibration(camera_index=1, pixels_per_mm_x=12.5, pixels_per_mm_y=12.5)
    )
    profile = service.capture_current("Model A", 3, created_by="admin")
    # Simulate the DB's baseline having moved on since capture — apply_profile
    # must push the profile's snapshot live, not whatever seed() left behind.
    calibration.seed(
        CameraCalibration(camera_index=1, pixels_per_mm_x=99.0, pixels_per_mm_y=99.0)
    )

    warnings = service.apply_profile(profile)
    assert warnings == []
    assert len(calibration.applied) == 1
    assert calibration.applied[0].pixels_per_mm_x == 12.5
    # apply_live (fake) never touches the "database" (here, the seed dict) —
    # only the real CalibrationManager.apply_live's non-persisting contract
    # is asserted directly in test_calibration_manager.py.


def test_apply_profile_warns_when_calibration_rejected(tmp_path) -> None:
    service, _cameras, _engine, _plc, calibration = make_service(tmp_path)
    calibration.seed(CameraCalibration(camera_index=1, pixels_per_mm_x=12.5))
    calibration.reject_camera = 1
    profile = service.capture_current("Model A", 3, created_by="admin")

    warnings = service.apply_profile(profile)
    assert len(warnings) == 1
    assert "camera 1" in warnings[0]
    assert calibration.applied == []


# ---------------------------------------------------- snapshot from live state
# A profile captures what the station is *running*, never what camera.json /
# detection.json say — applying a profile changes the former without writing
# the latter, so snapshotting the files would fold the previous model's values
# into the one now live.


def test_capture_snapshots_live_camera_settings_not_the_file(tmp_path) -> None:
    service, cameras, _engine, _plc, _calibration = make_service(tmp_path)
    cameras.apply_live_override(1, exposure_us=44000)  # pushed live, never persisted

    profile = service.capture_current("Model A", 3, created_by="admin")
    assert profile["cameras"]["1"]["exposure_us"] == 44000


def test_capture_snapshots_live_detection_block_not_the_file(tmp_path) -> None:
    service, _cameras, engine, _plc, _calibration = make_service(tmp_path)
    live = dict(DETECTION_DOC["cameras"]["1"], active_detector="dark_hole")
    engine.apply_camera_config(1, live)  # hot-swap without saving detection.json

    profile = service.capture_current("Model A", 3, created_by="admin")
    assert profile["detection"]["cameras"]["1"]["active_detector"] == "dark_hole"
    # ...while detection.json still holds the manually maintained baseline
    assert service._config.load("detection")["cameras"]["1"]["active_detector"] == "opencv"


# ------------------------------------------------------ automatic profile sync
# Saving on the Camera / Detection / Calibration page folds that one domain
# back into the applied profile, so nobody has to press "Update Selected from
# Current" afterwards.


def test_sync_is_a_no_op_while_no_profile_is_claimed(tmp_path) -> None:
    service, cameras, _engine, _plc, _calibration = make_service(tmp_path)
    service.capture_current("Model A", 3, created_by="admin")
    service.delete(1)  # the only profile, and the claimed target, is gone

    cameras.apply_live_override(1, exposure_us=44000)
    assert service.active_profile_id is None
    assert service.sync_active_profile("cameras") is None


def test_apply_profile_makes_it_the_sync_target(tmp_path) -> None:
    service, cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    service.capture_current("Model B", 4, created_by="admin")
    service.apply_profile(profile)
    assert service.active_profile_id == profile["id"]

    cameras.apply_live_override(1, exposure_us=44000)
    synced = service.sync_active_profile("cameras", updated_by="admin2")

    assert synced["id"] == profile["id"]
    assert service.get_by_id(profile["id"])["cameras"]["1"]["exposure_us"] == 44000
    assert service.get_by_id(profile["id"])["updated_by"] == "admin2"
    # Model B, merely captured earlier, is untouched.
    assert service.get_by_code(4)["cameras"]["1"]["exposure_us"] == 10000


def test_first_capture_becomes_the_sync_target_on_a_fresh_station(tmp_path) -> None:
    """Commissioning: "New from Current" before any model has been applied
    claims the still-unclaimed target, so the first profile starts tracking
    saves immediately."""
    service, cameras, _engine, _plc, _calibration = make_service(tmp_path)
    assert service.active_profile_id is None

    profile = service.capture_current("Model A", 3, created_by="admin")
    assert service.active_profile_id == profile["id"]

    cameras.apply_live_override(1, exposure_us=44000)
    service.sync_active_profile("cameras")
    assert service.get_by_id(profile["id"])["cameras"]["1"]["exposure_us"] == 44000


def test_capturing_a_second_profile_does_not_steal_the_live_model(tmp_path) -> None:
    """Copying the current settings into another profile must not redirect
    where the next Save lands — that is how the applied model starts drifting
    again, which is the bug the sync exists to stop."""
    service, cameras, _engine, _plc, _calibration = make_service(tmp_path)
    live = service.capture_current("Model A", 3, created_by="admin")
    service.apply_profile(live)

    other = service.capture_current("Model B", 4, created_by="admin")
    service.update_from_current(other["id"], updated_by="admin")
    assert service.active_profile_id == live["id"]

    cameras.apply_live_override(1, exposure_us=44000)
    service.sync_active_profile("cameras")

    assert service.get_by_id(live["id"])["cameras"]["1"]["exposure_us"] == 44000
    assert service.get_by_id(other["id"])["cameras"]["1"]["exposure_us"] == 10000


def test_sync_of_one_domain_leaves_the_others_alone(tmp_path) -> None:
    service, cameras, engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    service.apply_profile(profile)

    # Both a camera and a detection change are live, but only detection was saved.
    cameras.apply_live_override(1, exposure_us=44000)
    engine.apply_camera_config(1, dict(DETECTION_DOC["cameras"]["1"], active_detector="dark_hole"))
    service.sync_active_profile("detection")

    stored = service.get_by_id(profile["id"])
    assert stored["detection"]["cameras"]["1"]["active_detector"] == "dark_hole"
    assert stored["cameras"]["1"]["exposure_us"] == 10000  # not dragged along


def test_sync_persists_calibration_changes(tmp_path) -> None:
    service, _cameras, _engine, _plc, calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    service.apply_profile(profile)
    assert service.get_by_id(profile["id"])["calibration"] == {}

    calibration.seed(CameraCalibration(camera_index=1, pixels_per_mm_x=7.5))
    service.sync_active_profile("calibration")

    assert service.get_by_id(profile["id"])["calibration"]["1"]["pixels_per_mm_x"] == 7.5


def test_sync_that_changes_nothing_neither_rewrites_nor_notifies(tmp_path) -> None:
    service, _cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    service.apply_profile(profile)
    seen: list[tuple[dict, str]] = []
    service.subscribe(lambda p, domain: seen.append((p, domain)))

    assert service.sync_active_profile("cameras", updated_by="admin2") is None
    stored = service.get_by_id(profile["id"])
    assert stored["updated_at"] == profile["updated_at"]  # not bumped
    assert "updated_by" not in stored
    assert seen == []


def test_sync_notifies_observers_with_profile_and_domain(tmp_path) -> None:
    service, cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    service.apply_profile(profile)
    seen: list[tuple[str, str]] = []
    service.subscribe(lambda p, domain: seen.append((p["name"], domain)))

    cameras.apply_live_override(1, exposure_us=44000)
    service.sync_active_profile("cameras")

    assert seen == [("Model A", "cameras")]


def test_failing_observer_never_breaks_the_sync(tmp_path) -> None:
    service, cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    service.apply_profile(profile)
    service.subscribe(lambda p, d: (_ for _ in ()).throw(RuntimeError("boom")))

    cameras.apply_live_override(1, exposure_us=44000)
    assert service.sync_active_profile("cameras") is not None
    assert service.get_by_id(profile["id"])["cameras"]["1"]["exposure_us"] == 44000


def test_deleting_the_active_profile_stops_syncing_into_it(tmp_path) -> None:
    service, cameras, _engine, _plc, _calibration = make_service(tmp_path)
    profile = service.capture_current("Model A", 3, created_by="admin")
    service.apply_profile(profile)
    service.delete(profile["id"])

    cameras.apply_live_override(1, exposure_us=44000)
    assert service.active_profile_id is None
    assert service.sync_active_profile("cameras") is None


def test_sync_rejects_an_unknown_domain(tmp_path) -> None:
    service, _cameras, _engine, _plc, _calibration = make_service(tmp_path)
    with pytest.raises(ValueError):
        service.sync_active_profile("screw_driver_compensation")

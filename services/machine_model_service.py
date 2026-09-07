"""Machine-model profiles: capture the line's current per-camera settings —
camera ROI/exposure, per-camera detection strategy/parameters, and per-camera
pixel-to-mm calibration — into a named, PLC-code-tagged snapshot, and push a
profile back onto the live camera manager / vision engine / calibration
manager when the PLC reports a model change.

Profiles are JSON-primary (``config/machine_models.json``), like every other
settings domain — no DB mirror, matching ``detection.json``'s own precedent
(camera/PLC configs get a DB audit-mirror; detection does not). Calibration
*is* DB-backed (``CalibrationManager``), but applying a profile never writes
to that database — see below.

Applying a profile never persists ``camera.json``/``detection.json``, and
never writes a new row to the calibration database: it goes through the same
"preview, don't persist" entry points the Camera/Detection/Calibration pages
already use for their own live "Test"/"Apply Live" actions
(``CameraService.apply_live``, ``VisionEngine.apply_config``,
``CalibrationManager.apply_live``), so the manually maintained baseline
configuration — and the calibration history — is untouched by an automatic
switch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.calibration import CalibrationManager, CameraCalibration
from core.logging import get_logger
from core.utilities import ConfigManager
from core.utilities.enums import LogSource
from core.utilities.exceptions import ConfigurationError, VisionSystemError
from core.vision import VisionEngine, migrate_legacy_detection_config
from services.camera_service import CameraService
from services.plc_service import PlcService

logger = get_logger(LogSource.PLC)

# Camera fields a machine model is allowed to override — the part-dependent,
# tunable ones. Identity/wiring fields (driver, connection_id, name, enabled)
# stay whatever the camera is already configured with: they describe the
# physical rig, not the part, and CameraManager.apply_settings() doesn't even
# re-instantiate the driver, so overriding them here would silently do
# nothing useful while inviting a mismatch between camera.json and reality.
_TUNABLE_CAMERA_FIELDS = (
    "roi", "exposure_us", "gain_db", "gamma", "brightness",
    "width", "height", "trigger_mode",
)


class MachineModelService:
    """CRUD over machine-model profiles + pushing one live."""

    def __init__(
        self,
        config_manager: ConfigManager,
        camera_service: CameraService,
        vision_engine: VisionEngine,
        plc_service: PlcService,
        calibration_manager: CalibrationManager,
    ) -> None:
        self._config = config_manager
        self._cameras = camera_service
        self._engine = vision_engine
        self._plc = plc_service
        self._calibration = calibration_manager

    # -------------------------------------------------------------- queries
    def list_profiles(self) -> list[dict[str, Any]]:
        return self._config.load("machine_models").get("profiles", [])

    def get_by_id(self, profile_id: int) -> dict[str, Any] | None:
        for profile in self.list_profiles():
            if int(profile.get("id", -1)) == profile_id:
                return profile
        return None

    def get_by_code(self, plc_code: int) -> dict[str, Any] | None:
        for profile in self.list_profiles():
            if int(profile.get("plc_code", -1)) == plc_code:
                return profile
        return None

    # ------------------------------------------------------------- mutation
    def capture_current(self, name: str, plc_code: int, created_by: str) -> dict[str, Any]:
        """Snapshot current camera.json + (per-camera) detection.json + each
        camera's active calibration into a new profile.

        Raises:
            ConfigurationError: blank name, or plc_code already used.
        """
        document = self._config.load("machine_models")
        profiles = document.setdefault("profiles", [])
        self._validate_name_and_code(name, plc_code, profiles, exclude_id=None)

        now = datetime.now().isoformat(timespec="seconds")
        profile = {
            "id": max((int(p.get("id", 0)) for p in profiles), default=0) + 1,
            "name": name.strip(),
            "plc_code": plc_code,
            "cameras": self._snapshot_cameras(),
            "detection": self._config.load("detection"),
            "calibration": self._snapshot_calibrations(),
            "created_by": created_by,
            "created_at": now,
            "updated_at": now,
        }
        profiles.append(profile)
        self._config.save("machine_models", document)
        logger.info("Machine model profile %r captured (code %d)", name, plc_code)
        return profile

    def update_from_current(self, profile_id: int, updated_by: str) -> dict[str, Any]:
        """Re-capture current camera/detection/calibration settings into an
        existing profile.

        Raises:
            ConfigurationError: no profile with that id.
        """
        document = self._config.load("machine_models")
        profiles = document.setdefault("profiles", [])
        profile = self._find(profiles, profile_id)

        profile["cameras"] = self._snapshot_cameras()
        profile["detection"] = self._config.load("detection")
        profile["calibration"] = self._snapshot_calibrations()
        profile["updated_at"] = datetime.now().isoformat(timespec="seconds")
        profile["updated_by"] = updated_by
        self._config.save("machine_models", document)
        logger.info("Machine model profile %r updated from current settings", profile["name"])
        return profile

    def rename(self, profile_id: int, name: str, plc_code: int) -> None:
        """Raises ConfigurationError: no profile with that id, blank name, or
        plc_code already used by a different profile."""
        document = self._config.load("machine_models")
        profiles = document.setdefault("profiles", [])
        profile = self._find(profiles, profile_id)
        self._validate_name_and_code(name, plc_code, profiles, exclude_id=profile_id)
        profile["name"] = name.strip()
        profile["plc_code"] = plc_code
        profile["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._config.save("machine_models", document)

    def save_camera_position(
        self, profile_id: int, camera_index: int, x: int, y: int, z: int, updated_by: str
    ) -> dict[str, Any]:
        """Store *camera_index*'s current physical X/Y/Z position as this
        profile's image capture position — the position the camera mount is
        restored to whenever this machine model becomes active (see
        :meth:`apply_profile`).

        Raises:
            ConfigurationError: no profile with that id.
        """
        document = self._config.load("machine_models")
        profiles = document.setdefault("profiles", [])
        profile = self._find(profiles, profile_id)
        positions = profile.setdefault("jog_positions", {})
        positions[str(camera_index)] = {"x": int(x), "y": int(y), "z": int(z)}
        profile["updated_at"] = datetime.now().isoformat(timespec="seconds")
        profile["updated_by"] = updated_by
        self._config.save("machine_models", document)
        logger.info(
            "Machine model profile %r: camera %d image capture position -> (%d, %d, %d)",
            profile["name"], camera_index, x, y, z,
        )
        return profile

    def delete(self, profile_id: int) -> None:
        """Raises ConfigurationError: no profile with that id."""
        document = self._config.load("machine_models")
        profiles = document.get("profiles", [])
        remaining = [p for p in profiles if int(p.get("id", -1)) != profile_id]
        if len(remaining) == len(profiles):
            raise ConfigurationError(f"No machine model profile with id {profile_id}")
        document["profiles"] = remaining
        self._config.save("machine_models", document)
        logger.info("Machine model profile %d deleted", profile_id)

    # --------------------------------------------------------------- apply
    def apply_profile(self, profile: dict[str, Any]) -> list[str]:
        """Push *profile* live: per-camera ROI/exposure, X/Y/Z image capture
        position, per-camera detection, per-camera calibration, then echo the
        profile's PLC code back to the machine-model-select register.

        Never persists camera.json/detection.json, and never writes a new row
        to the calibration database. Camera and calibration application are
        both best-effort — a camera index the profile mentions that no longer
        exists on this station (or rejects its settings/calibration) is
        skipped, not fatal; likewise a saved capture position that the PLC
        rejects (link down, jog registers not configured) is a warning, not
        fatal — a missing physical reposition shouldn't block the optical
        settings from applying. Detection is all-or-nothing: a malformed
        detection block raises, since it is one atomic hot-swap for every
        camera's strategy at once (see ``VisionEngine.apply_config``). A
        profile captured before per-camera detection existed (no "cameras"
        key under "detection") is transparently upgraded on the fly — see
        :func:`migrate_legacy_detection_config` — so old profiles keep
        applying their one shared block to every camera, unchanged.

        The final model_select write is also best-effort: it runs whether
        this profile was applied because the PLC raised the register (in
        which case it echoes back the value just read — harmless) or an
        operator picked it manually via "Apply Now" — in which case it is
        what stops the PLC's poll worker from re-reading its own stale value
        and silently reverting the operator's choice (see
        ``workers.plc_poll_worker``, which treats every value seen right
        after a reconnect as a change, not just an edge).

        Returns:
            Warning strings for any camera (settings, calibration, or
            position) or the model_select write that could not be applied.

        Raises:
            VisionSystemError: the detection block was rejected.
        """
        warnings: list[str] = []
        current_cameras = {int(cfg["index"]): cfg for cfg in self._cameras.get_configs()}

        for index_str, overrides in profile.get("cameras", {}).items():
            index = int(index_str)
            current = current_cameras.get(index)
            if current is None:
                warnings.append(f"camera {index}: not configured on this station, skipped")
                continue
            merged = dict(current)
            for key in _TUNABLE_CAMERA_FIELDS:
                if key in overrides:
                    merged[key] = overrides[key]
            try:
                self._cameras.apply_live(merged)
            except VisionSystemError as exc:
                warnings.append(f"camera {index}: {exc}")

        for index_str, position in profile.get("jog_positions", {}).items():
            index = int(index_str)
            try:
                self._plc.set_camera_position(
                    index, int(position["x"]), int(position["y"]), int(position.get("z", 0))
                )
            except VisionSystemError as exc:
                warnings.append(f"camera {index}: image capture position not applied ({exc})")

        for index_str, calibration_data in profile.get("calibration", {}).items():
            index = int(index_str)
            try:
                self._calibration.apply_live(CameraCalibration.from_dict(index, calibration_data))
            except VisionSystemError as exc:
                warnings.append(f"camera {index}: calibration not applied ({exc})")

        detection = profile.get("detection")
        if detection:
            camera_indices = current_cameras.keys()
            self._engine.apply_config(migrate_legacy_detection_config(detection, camera_indices))

        try:
            self._plc.set_model_select(int(profile["plc_code"]))
        except VisionSystemError as exc:
            warnings.append(f"model_select register not written: {exc}")

        return warnings

    # -------------------------------------------------------------- internal
    def _snapshot_cameras(self) -> dict[str, dict[str, Any]]:
        snapshot: dict[str, dict[str, Any]] = {}
        for cfg in self._cameras.get_configs():
            index = int(cfg["index"])
            snapshot[str(index)] = {key: cfg[key] for key in _TUNABLE_CAMERA_FIELDS if key in cfg}
        return snapshot

    def _snapshot_calibrations(self) -> dict[str, dict[str, Any]]:
        """Every configured camera's active calibration, JSON-serialised.

        A camera with no active calibration (identity fallback) is simply
        omitted, matching ``jog_positions``'s sparse-by-default convention.
        """
        snapshot: dict[str, dict[str, Any]] = {}
        for cfg in self._cameras.get_configs():
            index = int(cfg["index"])
            calibration = self._calibration.get(index)
            if calibration is not None:
                snapshot[str(index)] = calibration.to_dict()
        return snapshot

    @staticmethod
    def _find(profiles: list[dict[str, Any]], profile_id: int) -> dict[str, Any]:
        for profile in profiles:
            if int(profile.get("id", -1)) == profile_id:
                return profile
        raise ConfigurationError(f"No machine model profile with id {profile_id}")

    @staticmethod
    def _validate_name_and_code(
        name: str, plc_code: int, profiles: list[dict[str, Any]], *, exclude_id: int | None
    ) -> None:
        if not name.strip():
            raise ConfigurationError("Machine model name must not be blank")
        for profile in profiles:
            if exclude_id is not None and int(profile.get("id", -1)) == exclude_id:
                continue
            if int(profile.get("plc_code", -1)) == plc_code:
                raise ConfigurationError(
                    f"PLC code {plc_code} is already used by machine model "
                    f"{profile.get('name', '?')!r}"
                )

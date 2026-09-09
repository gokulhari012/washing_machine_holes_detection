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

The traffic in the other direction is :meth:`MachineModelService.sync_active_profile`:
once a profile is live, saving on the Camera, Detection or Calibration page
folds that one domain straight back into it, so the profile never drifts from
the settings the station is actually running and nobody has to remember to
press "Update Selected from Current" afterwards. The composition root wires
the three save notifications to it; the Machine Models page repaints on the
:meth:`subscribe` fan-out.

Everything a profile captures is therefore read from the **live** objects,
never from ``camera.json``/``detection.json``: applying a profile does not
write those files, so after a model switch the files describe the *previous*
model and snapshotting them would quietly undo the switch (CLAUDE.md §8).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

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


# Profile keys :meth:`MachineModelService.sync_active_profile` can re-capture
# on its own — one settings domain per engineering page, matching the three
# blocks "Update Selected from Current" writes together.
_SYNCABLE_DOMAINS = ("cameras", "detection", "calibration")

ProfileCallback = Callable[[dict[str, Any], str], None]


def _default_screw_compensation() -> dict[str, Any]:
    """A fresh (unshared) default block: compensation off, one zeroed
    (x_mm, y_mm) slot per camera 1-4, edited directly on the Machine Models
    page rather than captured from a live value (see ``set_screw_compensation``)."""
    return {
        "enabled": False,
        "positions": {str(index): {"x_mm": 0.0, "y_mm": 0.0} for index in range(1, 5)},
    }


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
        self._active_profile_id: int | None = None
        self._observers: list[ProfileCallback] = []

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
            "detection": self._snapshot_detection(),
            "calibration": self._snapshot_calibrations(),
            "screw_driver_compensation": _default_screw_compensation(),
            "created_by": created_by,
            "created_at": now,
            "updated_at": now,
        }
        profiles.append(profile)
        self._config.save("machine_models", document)
        self._adopt_if_unclaimed(int(profile["id"]))
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
        profile["detection"] = self._snapshot_detection()
        profile["calibration"] = self._snapshot_calibrations()
        profile["updated_at"] = datetime.now().isoformat(timespec="seconds")
        profile["updated_by"] = updated_by
        self._config.save("machine_models", document)
        self._adopt_if_unclaimed(profile_id)
        logger.info("Machine model profile %r updated from current settings", profile["name"])
        return profile

    def set_screw_compensation(
        self,
        profile_id: int,
        enabled: bool,
        positions: dict[int, tuple[float, float]],
        updated_by: str,
    ) -> dict[str, Any]:
        """Save the 4 screw-driver position offsets directly onto a profile.

        Unlike ``cameras``/``detection``/``calibration``, this block is
        edited by hand on the Machine Models page rather than captured from
        a currently-live value — nothing elsewhere in the application holds
        a "current" screw driver offset to snapshot. Applying the profile
        (``apply_profile``) pushes it into ``CalibrationManager``, which adds
        it to the position written to the PLC only (see
        ``InspectionService._plc_position``) — never to the measured
        position shown on the dashboard or stored in the database.

        Raises:
            ConfigurationError: no profile with that id.
        """
        document = self._config.load("machine_models")
        profiles = document.setdefault("profiles", [])
        profile = self._find(profiles, profile_id)
        profile["screw_driver_compensation"] = {
            "enabled": bool(enabled),
            "positions": {
                str(index): {"x_mm": float(x_mm), "y_mm": float(y_mm)}
                for index, (x_mm, y_mm) in positions.items()
            },
        }
        profile["updated_at"] = datetime.now().isoformat(timespec="seconds")
        profile["updated_by"] = updated_by
        self._config.save("machine_models", document)
        logger.info(
            "Screw driver compensation saved for machine model profile %r", profile["name"]
        )
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

    def delete(self, profile_id: int) -> None:
        """Raises ConfigurationError: no profile with that id."""
        document = self._config.load("machine_models")
        profiles = document.get("profiles", [])
        remaining = [p for p in profiles if int(p.get("id", -1)) != profile_id]
        if len(remaining) == len(profiles):
            raise ConfigurationError(f"No machine model profile with id {profile_id}")
        document["profiles"] = remaining
        self._config.save("machine_models", document)
        if self._active_profile_id == profile_id:
            self._active_profile_id = None  # nothing left to sync saves into
        logger.info("Machine model profile %d deleted", profile_id)

    # --------------------------------------------------------------- apply
    def apply_profile(self, profile: dict[str, Any]) -> list[str]:
        """Push *profile* live: per-camera ROI/exposure, per-camera
        detection, per-camera calibration and the profile's screw-driver
        compensation offsets, then echo the profile's PLC code back to the
        machine-model-select register.

        Also claims *profile* as the target a later page Save syncs into —
        see :attr:`active_profile_id`.

        Never persists camera.json/detection.json, and never writes a new row
        to the calibration database. Camera and calibration application are
        both best-effort — a camera index the profile mentions that no longer
        exists on this station (or rejects its settings/calibration) is
        skipped, not fatal. Detection is all-or-nothing: a malformed
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
            Warning strings for any camera (settings or calibration) or the
            model_select write that could not be applied.

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

        compensation = profile.get("screw_driver_compensation", {})
        screw_positions = {
            int(index): (float(pos.get("x_mm", 0.0)), float(pos.get("y_mm", 0.0)))
            for index, pos in compensation.get("positions", {}).items()
        }
        self._calibration.apply_screw_compensation(
            bool(compensation.get("enabled", False)), screw_positions
        )

        try:
            self._plc.set_model_select(int(profile["plc_code"]))
        except VisionSystemError as exc:
            warnings.append(f"model_select register not written: {exc}")

        # This profile now describes the running settings, so it is where a
        # subsequent page Save is folded back (see :attr:`active_profile_id`).
        # An id-less profile — only a hand-built one in a test — releases the
        # claim rather than leaving saves pointed at the outgoing model.
        self._active_profile_id = int(profile["id"]) if "id" in profile else None
        return warnings

    # ------------------------------------------------------ active profile
    @property
    def active_profile_id(self) -> int | None:
        """Id of the profile the station's live settings belong to — the one
        :meth:`sync_active_profile` writes a saved settings domain back into.

        Claimed by :meth:`apply_profile`: a model that has been pushed live
        (by the PLC's model_select register or by "Apply Now") *is* what the
        running settings describe, and it is what the "Currently active"
        line on the Machine Models page names.

        ``None`` until that has happened, and again once the active profile
        is deleted — which is what stops a station that has never selected a
        model from quietly growing every settings edit into an arbitrary
        profile. Only in that unclaimed state do
        :meth:`capture_current`/:meth:`update_from_current` adopt the profile
        they just wrote (see :meth:`_adopt_if_unclaimed`), so commissioning a
        station with "New from Current" works before any model has been
        applied.

        Deliberately *not* the Machine Models page's selected row: browsing
        profiles — or copying the current settings into a second profile with
        "Update Selected from Current" — must not redirect where the next
        Camera-page Save lands, or the model actually running would start
        drifting again, which is the whole problem this exists to stop.
        """
        return self._active_profile_id

    def _adopt_if_unclaimed(self, profile_id: int) -> None:
        """Make *profile_id* the sync target only while no model is live.

        See :attr:`active_profile_id` for why a claimed target is never
        stolen by a capture.
        """
        if self._active_profile_id is None:
            self._active_profile_id = profile_id

    def sync_active_profile(
        self, domain: str, *, updated_by: str = ""
    ) -> dict[str, Any] | None:
        """Re-capture one live settings domain into the active profile.

        This is what makes a Camera / Detection / Calibration page's **Save**
        show up in the machine model immediately, instead of waiting for
        someone to return to the Machine Models page and press "Update
        Selected from Current". A profile that has been applied describes the
        station's running configuration; letting it drift from the settings
        actually in force is how a model switch later silently reverts a
        tuning session.

        One domain per call, and each snapshot is read from the live objects
        rather than the config files (see :meth:`_snapshot_cameras`): saving
        on the Detection page must not drag the camera and calibration blocks
        back to whatever ``camera.json`` and the calibration table hold, which
        after a model switch is the *previous* model.

        Returns the rewritten profile, or ``None`` when there is nothing to
        do — no active profile, the active profile has since been deleted, or
        the snapshot is identical to what the profile already holds. That last
        case matters: a Save that changed nothing must not rewrite
        machine_models.json or bump ``updated_at``.

        Raises:
            ValueError: *domain* is not one of ``cameras``/``detection``/
                ``calibration`` (a programming error, so deliberately outside
                the ``VisionSystemError`` tree).
            ConfigurationError: machine_models.json could not be read/written.
        """
        if domain not in _SYNCABLE_DOMAINS:
            raise ValueError(
                f"Unknown machine-model domain {domain!r}; expected one of {_SYNCABLE_DOMAINS}"
            )
        profile_id = self._active_profile_id
        if profile_id is None:
            return None

        document = self._config.load("machine_models")
        profiles = document.setdefault("profiles", [])
        try:
            profile = self._find(profiles, profile_id)
        except ConfigurationError:
            # Deleted from under us by another page; stop trying to sync it.
            self._active_profile_id = None
            return None

        snapshot = self._snapshot(domain)
        if profile.get(domain) == snapshot:
            return None

        profile[domain] = snapshot
        profile["updated_at"] = datetime.now().isoformat(timespec="seconds")
        if updated_by:
            profile["updated_by"] = updated_by
        self._config.save("machine_models", document)
        logger.info(
            "Machine model profile %r synced from saved %s settings", profile["name"], domain
        )
        self._notify(profile, domain)
        return profile

    # ----------------------------------------------------------- observers
    def subscribe(self, callback: ProfileCallback) -> None:
        """Register ``callback(profile, domain)``, invoked after
        :meth:`sync_active_profile` actually rewrote a profile.

        Plain callables rather than Qt signals, matching ``ConfigManager``:
        this service stays Qt-free. The Machine Models page uses it to
        repaint, since an automatic sync is the one profile change no button
        on that page initiated. Callbacks run on the saving thread — in
        practice the GUI thread, where the engineering pages' Save buttons
        live — and an exception in one is logged, never propagated.
        """
        self._observers.append(callback)

    def unsubscribe(self, callback: ProfileCallback) -> None:
        """Remove a previously registered callback (no-op if absent)."""
        try:
            self._observers.remove(callback)
        except ValueError:
            pass

    def _notify(self, profile: dict[str, Any], domain: str) -> None:
        for callback in list(self._observers):
            try:
                callback(profile, domain)
            except Exception:  # an observer must never break a save
                logger.exception("Machine model observer failed")

    # -------------------------------------------------------------- internal
    def _snapshot(self, domain: str) -> Any:
        return {
            "cameras": self._snapshot_cameras,
            "detection": self._snapshot_detection,
            "calibration": self._snapshot_calibrations,
        }[domain]()

    def _snapshot_cameras(self) -> dict[str, dict[str, Any]]:
        """Every camera's tunable fields as the station is *running* them.

        Reads ``CameraService.get_effective_configs()`` — camera.json with the
        live ``CameraManager`` settings merged over it — not the persisted
        baseline. Applying a profile (and the Camera page's "Apply Live")
        changes the running camera without writing camera.json, so the file is
        the wrong answer to "what is current": snapshotting it would capture
        the *previous* model's ROI/exposure into the one now live. Same rule
        the Camera page itself follows (CLAUDE.md §8).
        """
        snapshot: dict[str, dict[str, Any]] = {}
        for cfg in self._cameras.get_effective_configs():
            index = int(cfg["index"])
            snapshot[str(index)] = {key: cfg[key] for key in _TUNABLE_CAMERA_FIELDS if key in cfg}
        return snapshot

    def _snapshot_detection(self) -> dict[str, Any]:
        """Every camera's *live* detection block, in detection.json's shape.

        ``VisionEngine.camera_config`` rather than ``detection.json``, for the
        same reason :meth:`_snapshot_cameras` reads the effective camera
        config: ``apply_config``/``apply_camera_config`` are "preview, don't
        persist", so after a model switch (or a Detection-page "Test") the
        file is the older document. A camera the engine has no strategy for
        falls back to the file's block, so a profile never loses a camera
        outright; if that leaves nothing at all, the whole file is kept, since
        an empty ``cameras`` map is not a document ``apply_config`` accepts.
        """
        document = self._config.load("detection")
        from_file = document.get("cameras", {})
        indices = {int(cfg["index"]) for cfg in self._cameras.get_effective_configs()}
        indices.update(int(index) for index in from_file)

        snapshot: dict[str, Any] = {}
        for index in sorted(indices):
            block = self._engine.camera_config(index) or from_file.get(str(index))
            if block is not None:
                snapshot[str(index)] = block
        return {"cameras": snapshot} if snapshot else document

    def _snapshot_calibrations(self) -> dict[str, dict[str, Any]]:
        """Every configured camera's active calibration, JSON-serialised.

        Already live by construction — ``CalibrationManager.get`` reads the
        same cache ``apply_live`` writes, not the calibration table.

        A camera with no active calibration (identity fallback) is simply
        omitted — the map is sparse by convention, never padded with blanks.
        """
        snapshot: dict[str, dict[str, Any]] = {}
        for cfg in self._cameras.get_effective_configs():
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

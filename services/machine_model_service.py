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
once a profile is the *selected* one, saving on the Camera, Detection or
Calibration page folds the saved camera's settings straight back into it, so
nobody has to remember to press "Update Selected from Current" afterwards.
A profile becomes selected when it is picked on the Machine Models page
(:meth:`select_profile`) or applied (:meth:`apply_profile` — the PLC's
model_select register or "Apply Now"), and the choice is persisted in
machine_models.json (``selected_profile_id``) so it survives a restart — a
station whose PLC is offline never applies a model, and used to drop every
Detection-page Save on the floor for exactly that reason. The composition
root wires the three save notifications to it; the Machine Models page
repaints on the :meth:`subscribe` fan-out.

Each sync replaces **only the camera that was saved** (``camera_indices``):
the selected profile need not be the one running, so re-capturing every
camera's live block would overwrite the profile's other cameras with
whatever the station happens to be running.

Everything a profile captures is therefore read from the **live** objects,
never from ``camera.json``/``detection.json``: applying a profile does not
write those files, so after a model switch the files describe the *previous*
model and snapshotting them would quietly undo the switch (CLAUDE.md §8).
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Callable, Iterable

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

# Keys of a camera's detection block that are not a strategy's parameters.
_NON_STRATEGY_KEYS = ("active_detector", "common")

ProfileCallback = Callable[[dict[str, Any], str], None]


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
        self._observers: list[ProfileCallback] = []
        self._active_profile_id: int | None = self._load_selected_id()
        self._backfill_profiles()

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
        if self._active_profile_id == profile_id:
            self._active_profile_id = None  # nothing left to sync saves into
            document["selected_profile_id"] = None
        self._config.save("machine_models", document)
        logger.info("Machine model profile %d deleted", profile_id)

    # --------------------------------------------------------------- apply
    def apply_profile(self, profile: dict[str, Any]) -> list[str]:
        """Push *profile* live: per-camera ROI/exposure, per-camera
        detection and per-camera calibration, then echo the profile's PLC
        code back to the machine-model-select register.

        Screw-driver position compensation is deliberately **not** part of a
        profile: it rides the camera's calibration instead (Calibration page,
        Step 5), because where the screw driver sits relative to the camera is
        a fact about the rig, not the part — the same reason the axis signs
        stay out (``CameraCalibration.to_dict``).

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
            detection = copy.deepcopy(migrate_legacy_detection_config(detection, camera_indices))
            self._fill_missing_strategy_blocks(detection)
            self._engine.apply_config(detection)

        try:
            self._plc.set_model_select(int(profile["plc_code"]))
        except VisionSystemError as exc:
            warnings.append(f"model_select register not written: {exc}")

        # This profile now describes the running settings, so it is where a
        # subsequent page Save is folded back (see :attr:`active_profile_id`).
        # An id-less profile — only a hand-built one in a test — releases the
        # claim rather than leaving saves pointed at the outgoing model.
        self._set_target(int(profile["id"]) if "id" in profile else None)
        return warnings

    # ------------------------------------------------------ active profile
    @property
    def active_profile_id(self) -> int | None:
        """Id of the *selected* profile — the one :meth:`sync_active_profile`
        writes a saved settings domain back into.

        Set by picking a profile on the Machine Models page
        (:meth:`select_profile`) and by :meth:`apply_profile` (the PLC's
        model_select register or "Apply Now"), whichever happened last, and
        persisted as ``selected_profile_id`` in machine_models.json so a
        restart keeps it. It used to be claimed by applying alone and held
        only in memory, which meant a station whose PLC was offline never had
        a target and every Detection-page Save silently updated no model.

        ``None`` until a profile has been selected, and again once the
        selected profile is deleted — which is what stops a station that has
        never chosen a model from quietly growing every settings edit into an
        arbitrary profile. Only in that unclaimed state do
        :meth:`capture_current`/:meth:`update_from_current` adopt the profile
        they just wrote (see :meth:`_adopt_if_unclaimed`).
        """
        return self._active_profile_id

    def select_profile(self, profile_id: int | None) -> None:
        """Make *profile_id* the profile page Saves update (``None`` clears).

        The Machine Models page calls this when the operator picks a row.
        Nothing is applied — the running cameras/detectors/calibrations are
        untouched; only where the next Camera/Detection/Calibration Save
        lands changes.

        Raises:
            ConfigurationError: no profile with that id.
        """
        if profile_id is not None and self.get_by_id(profile_id) is None:
            raise ConfigurationError(f"No machine model profile with id {profile_id}")
        self._set_target(profile_id)

    def _set_target(self, profile_id: int | None) -> None:
        """Point syncs at *profile_id* and persist it, writing only on change."""
        if profile_id == self._active_profile_id:
            return
        self._active_profile_id = profile_id
        try:
            document = self._config.load("machine_models")
            if document.get("selected_profile_id") != profile_id:
                document["selected_profile_id"] = profile_id
                self._config.save("machine_models", document)
        except VisionSystemError as exc:
            # The in-memory target still holds for this session.
            logger.warning("Selected machine model not persisted: %s", exc)

    def _load_selected_id(self) -> int | None:
        """The persisted ``selected_profile_id``, if it still names a profile."""
        try:
            selected = self._config.load("machine_models").get("selected_profile_id")
        except VisionSystemError as exc:
            logger.warning("Selected machine model not read: %s", exc)
            return None
        if selected is None:
            return None
        try:
            selected = int(selected)
        except (TypeError, ValueError):
            return None
        return selected if self.get_by_id(selected) is not None else None

    def _adopt_if_unclaimed(self, profile_id: int) -> None:
        """Make *profile_id* the sync target only while none is selected.

        See :attr:`active_profile_id` for why a claimed target is never
        stolen by a capture.
        """
        if self._active_profile_id is None:
            self._set_target(profile_id)

    def sync_active_profile(
        self,
        domain: str,
        *,
        camera_indices: Iterable[int] | None = None,
        updated_by: str = "",
    ) -> dict[str, Any] | None:
        """Re-capture one live settings domain into the selected profile.

        This is what makes a Camera / Detection / Calibration page's **Save**
        show up in the machine model immediately, instead of waiting for
        someone to return to the Machine Models page and press "Update
        Selected from Current".

        One domain per call, and each snapshot is read from the live objects
        rather than the config files (see :meth:`_snapshot_cameras`): saving
        on the Detection page must not drag the camera and calibration blocks
        back to whatever ``camera.json`` and the calibration table hold, which
        after a model switch is the *previous* model.

        *camera_indices* narrows the re-capture to those cameras (every page
        saves one camera at a time); the profile's other cameras keep what
        they hold. ``None`` re-captures every camera. Narrowing matters
        because the selected profile need not be the one running: copying all
        four live blocks would overwrite the profile's other cameras with
        another model's settings.

        Returns the rewritten profile, or ``None`` when there is nothing to
        do — no selected profile, the selected profile has since been
        deleted, or the snapshot is identical to what the profile already
        holds. That last case matters: a Save that changed nothing must not
        rewrite machine_models.json or bump ``updated_at``.

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
            self._set_target(None)
            return None

        snapshot = self._snapshot(domain)
        if camera_indices is not None:
            snapshot = self._merge_cameras(domain, profile.get(domain), snapshot, camera_indices)
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
    def _backfill_profiles(self) -> None:
        """Give every stored profile the strategy blocks it predates.

        A profile carries every strategy's block and every parameter, but
        only the ones that existed when it was captured — this station's
        profiles from before ``blob_analysis``/``caliper_based_circle_fitting``
        (and their ``template_*`` variants) had no block for them, and older
        ones lacked newer keys inside existing blocks (``template_matching``
        without ``scales``/``max_matches``/the diameter gates, ``common``
        without downscale/NG re-run). Applying one left those Detection-page
        fields on hard-coded fallbacks, which the next Save then wrote into
        the model. Persisted once here (no ``updated_at`` bump: nothing the
        model runs changes), and repeated in :meth:`apply_profile` for any
        strategy or parameter added later.
        """
        try:
            document = self._config.load("machine_models")
        except VisionSystemError as exc:
            logger.warning("Machine model profiles not checked for missing settings: %s", exc)
            return
        filled: dict[str, list[str]] = {}
        for profile in document.get("profiles", []):
            detection = profile.get("detection")
            if isinstance(detection, dict) and "cameras" in detection:
                added = self._fill_missing_strategy_blocks(detection)
                if added:
                    filled[str(profile.get("name", profile.get("id")))] = added
        if not filled:
            return
        try:
            self._config.save("machine_models", document)
        except VisionSystemError as exc:
            logger.warning("Machine model settings backfill not saved: %s", exc)
            return
        for name, added in filled.items():
            settings = sorted({entry.split(": ", 1)[1] for entry in added})
            logger.info(
                "Machine model profile %r: added missing detection settings (%s)",
                name, ", ".join(settings),
            )

    def _fill_missing_strategy_blocks(self, detection: dict[str, Any]) -> list[str]:
        """Complete a per-camera *detection* document in place; returns
        ``"camera N: name"`` per block added and ``"camera N: name.key"`` per
        parameter added.

        Two sources, on purpose:

        - a **missing block** is copied whole from that camera's entry in
          ``detection.json`` — the station's own tuning, and what the
          Detection page shows when no model is applied — or failing that
          the shipped defaults. A block the model never had is not one it
          runs, so its values cannot move a verdict;
        - a **missing key inside an existing block** (``common`` included)
          comes from the **shipped defaults** only. The model has been
          running without that key, i.e. on the detector's own fallback,
          which the defaults file mirrors — so recording it changes nothing,
          where another model's tuned value mixed into this model's block
          would.

        Keys the model already has are never touched.
        """
        sources = []
        for load in (self._config.load, self._config.load_defaults):
            try:
                sources.append(load("detection").get("cameras", {}))
            except VisionSystemError:
                sources.append({})
        defaults = sources[1]
        added: list[str] = []
        for index, block in detection.get("cameras", {}).items():
            if not isinstance(block, dict):
                continue
            original = set(block)
            for source in sources:
                for name, params in source.get(str(index), {}).items():
                    if name in _NON_STRATEGY_KEYS or name in block or not isinstance(params, dict):
                        continue
                    block[name] = copy.deepcopy(params)
                    added.append(f"camera {index}: {name}")
            for name in original:
                params = block[name]
                reference = defaults.get(str(index), {}).get(name)
                if not isinstance(params, dict) or not isinstance(reference, dict):
                    continue
                for key, value in reference.items():
                    if key not in params:
                        params[key] = copy.deepcopy(value)
                        added.append(f"camera {index}: {name}.{key}")
        return added

    def _merge_cameras(
        self, domain: str, existing: Any, snapshot: Any, camera_indices: Iterable[int]
    ) -> Any:
        """*existing* with only *camera_indices* replaced from *snapshot*.

        Every syncable domain is a ``{"<index>": block}`` map — detection's
        under its ``cameras`` key. A camera the snapshot no longer has (a
        calibration cleared back to the identity fallback) is dropped, so the
        sparse calibration map stays sparse. A profile holding nothing for the
        domain yet takes the whole snapshot: a partial detection document is
        not one :meth:`apply_profile` can hot-swap cleanly.
        """
        if not existing:
            return snapshot
        keys = {str(int(index)) for index in camera_indices}
        if domain == "detection":
            indices = [int(cfg["index"]) for cfg in self._cameras.get_effective_configs()]
            merged = copy.deepcopy(migrate_legacy_detection_config(existing, indices))
            target = merged.setdefault("cameras", {})
            fresh = snapshot.get("cameras", {})
        else:
            merged = copy.deepcopy(existing)
            target = merged
            fresh = snapshot
        for key in keys:
            if key in fresh:
                target[key] = fresh[key]
            else:
                target.pop(key, None)
        return merged

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

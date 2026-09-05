# CLAUDE.md — working reference

Orientation notes for this repo so a session can start changing code without
re-reading everything. Companion docs: [README.md](README.md) (operator/commissioning
guide) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (design intent — **partly stale**, see
[Doc drift](#doc-drift)).

---

## 1. What this is

Windows PySide6 desktop HMI for an industrial machine-vision station. Four fixed
cameras photograph the bottom of a washing machine on a conveyor; a detector finds
the drain-hole in each frame; hole X/Y (mm) + GOOD/NG/ERROR go back to the line PLC
over Ethernet; every cycle is stored in SQLite with an annotated PNG.

~16 k lines of Python, no build step for the app itself. Python 3.12+ (uses `StrEnum`).

### Commands

```bash
python main.py                # normal start
python main.py --selftest 8   # start hidden, run 8 s, save logs/selftest.png, exit 0
pytest                        # 109 tests, ~8 s, all passing as of 2026-09-02
python tools/hole_debug.py path/to/images/   # detector tuner; --camera N applies that camera's ROI
python scripts/build_exe.py   # PyInstaller onedir -> dist/WMHoleDetection/
```

`--selftest` is the fastest end-to-end smoke check after a change — it builds the
whole object graph, runs live cycles and exits non-zero on a startup crash.

---

## 2. Layering — the one rule that matters

```
ui/  →  models/  →  services/  →  workers/ + core/
```

**Dependency direction is strictly downward.** `core/` imports nothing from
`services/`, `workers/`, or `ui/`. `core/` is Qt-free except that `workers/` (which
is not core) uses QThread. `ui/` never touches hardware directly.

[main.py](main.py) is the **only** composition root — the single place that constructs
anything. No globals, no singletons, no service locators. If you need a new
dependency in a page or service, add a constructor parameter and wire it in
`Application.__init__`.

Workers own **threads only** — zero business logic. Business logic lives in services.

---

## 3. Directory map — what to touch

| Path | Contents | Touch when |
|---|---|---|
| [main.py](main.py) | composition root, startup/shutdown order, global excepthook | adding any new service/page/worker |
| [config/](config/) | live JSON: `app_config`, `plc`, `camera`, `detection`, `machine_models` | changing runtime settings |
| [config/defaults/](config/defaults/) | pristine copies for "Restore Defaults" | **must mirror any new config key** |
| [core/utilities/](core/utilities/) | `ConfigManager`, enums, exception hierarchy | adding a config domain, enum value, error type |
| [core/plc/](core/plc/) | `PlcClientBase`, Modbus/SLMP/Simulated adapters, `RegisterMap`, `PlcManager` | protocol or register work |
| [core/camera/](core/camera/) | `CameraBase`, 5 drivers, `CameraManager` | camera driver work |
| [core/vision/](core/vision/) | `HoleDetector` ABC, 4 detectors, `VisionEngine`, overlay drawing | detection algorithm work |
| [core/calibration/](core/calibration/) | px→mm scale / homography, reference point | coordinate math |
| [core/database/](core/database/) | SQLAlchemy engine (WAL), ORM models, repositories | schema/query work |
| [models/](models/) | `AppState` (observable QObject) + frozen DTOs | new cross-thread signal or payload field |
| [services/](services/) | orchestration; `InspectionService` is the pipeline | business rules |
| [workers/](workers/) | 4 thread hosts (see [§7](#7-threading-model)) | cadence/lifecycle work |
| [ui/](ui/) | `main_window` + 9 pages + `widgets/` | any screen change |
| [resources/styles/](resources/) | dark QSS theme | styling |
| [tests/](tests/) | pytest — vision, PLC, repositories, calibration | always |
| [tools/](tools/) | `hole_debug.py` tuner, `basler_probe.py` | detector tuning |
| [scripts/](scripts/), [development_files/](development_files/) | build script + throwaway lab scripts | rarely — see [gotchas](#9-gotchas--traps) |

Runtime artifacts are created **beside the app** (`os.chdir(BASE_DIR)` at
[main.py:29](main.py#L29); frozen builds anchor on `sys.executable`, not `__file__`):
`data/` (SQLite+WAL), `images/` (per-day annotated PNGs),
`logs/`, `backups/`, `exports/`. All gitignored.

---

## 4. Extension points — exact registration sites

The three strategy/adapter families are the designed way to extend. Each needs
**two** edits: implement the class, then register it.

### Add a camera driver
1. Subclass `CameraBase` in `core/camera/<name>_camera.py`, implementing the four
   hooks: `_connect_device`, `_disconnect_device`, `_grab`, `_apply_to_device`
   (optionally `_detect_resolution`).
2. Add the value to `CameraDriver` in [core/utilities/enums.py](core/utilities/enums.py).
3. Register in the `create_camera` factory at [core/camera/__init__.py](core/camera/__init__.py) —
   **import it lazily inside the branch** so stations without that vendor SDK still boot.
4. Add to `hiddenimports` in [wmhd.spec](wmhd.spec) (lazy imports are invisible to PyInstaller).

`CameraBase` gives you locking, ROI cropping, timing and error translation for free —
`capture()` serialises on an internal lock because the preview worker and the
inspection pipeline both grab the same device.

### Add a detector strategy
1. Subclass `HoleDetector` in `core/vision/<name>_detector.py`; implement `detect()`,
   optionally `debug_stages()` (returns `{"mask": ..., "edges": ...}` for the debug overlay).
   Set `thread_safe = False` if inference is not re-entrant — `VisionEngine` then
   serialises it for you.
2. Add the value to `DetectorType` in [core/utilities/enums.py](core/utilities/enums.py).
3. Register in `_REGISTRY` at [core/vision/vision_engine.py](core/vision/vision_engine.py).
4. Add a parameter block named exactly like the enum value to **both**
   `config/detection.json` and `config/defaults/detection.json`.

No UI or PLC changes needed. `detect()` raising `DetectionError` means *algorithm
failure*; returning an empty `holes` list means *no hole present* (a legitimate NG).

### Add a PLC protocol
1. Subclass `PlcClientBase` in `core/plc/<name>_client.py` — 16-bit unsigned holding
   registers addressed by plain int. Raise the `PlcError` subtypes on failure.
2. Register in the `create_plc_client` factory at [core/plc/__init__.py](core/plc/__init__.py),
   keyed on `connection.protocol`, importing lazily.
3. Add to `hiddenimports` in [wmhd.spec](wmhd.spec).

### Add a UI page
Construct it in `Application.__init__` and call
`self.window.add_page(title, glyph, page, admin_only=...)`. `admin_only=True` hides it
from the nav rail until an admin logs in (page stays in the stack — see
`MainWindow._refresh_nav_visibility`).

---

## 5. The inspection cycle (end to end)

Entry point: [services/inspection_service.py](services/inspection_service.py) →
`run_inspection(machine_number)`. **Never raises** — every fault degrades the result.

```
PlcPollWorker sees trigger 0→1  (≤50 ms)
  → reads machine_number
  → emits trigger_detected  ──queued──►  InspectionWorker.on_trigger
                                          │
     InspectionService.run_inspection:    │
       1/2. capture + detect per camera   │  sequential (default) or parallel
       3.   overall result                │  ERROR > NG > GOOD (worst wins)
       4.   write PLC: positions, per-cam results, overall, then vision_complete=1
       5.   save annotated PNGs           │  per app_config.storage policy
       6.   persist to SQLite             │  synchronous, WAL makes it ms-fast
       7.   publish to AppState           │  dashboard updates
```

**Capture modes** (`app_config.inspection.capture_mode`):
- `sequential` (**current setting**) — one camera at a time, each picture published to
  the dashboard the moment it is taken, `camera_delay_ms` between cameras. Right for
  four high-res GigE cameras sharing one NIC.
- `parallel` — all four grab and detect at once via `ThreadPoolExecutor`. Shortest cycle.

**Judgement per camera** (`_inspect_one`):
- no frame → `ERROR`
- `DetectionError` → `ERROR`
- `best is None` **or** `len(holes) < expected_hole_count` → `NG`, `hole_found=False`
- deviation > `position_tolerance_mm` (and tolerance > 0 and camera calibrated) → `NG`
- else `GOOD`

**Single-camera cycles** (`run_camera_inspection`): triggered by PLC register 132+N or
the dashboard panel's ▶ button. Captures and judges one camera, writes only that
camera's registers, and marks the cycle `partial=True` — it is stored and shown in
history like any other inspection but **does not increment the product counters**
(`AppState.publish_inspection` returns early), because one camera is not a finished
machine. The inspection worker shares one busy flag across both trigger kinds, so a
per-camera and a full cycle can never overlap.

**Hole selection** (`_select_hole`): an *uncalibrated* camera keeps the
highest-confidence candidate. A *calibrated* camera instead picks the candidate
nearest the reference point — the highest-confidence hole is not necessarily the
right hole when several real holes are in frame.

**Coordinate convention (important):** the `x_mm`/`y_mm` reported to PLC, dashboard
and DB are relative to the **analysed image's own centre** — a hole exactly centred
in frame always reports `(0, 0)`, regardless of the calibration's coordinate frame.
The tolerance judgement (`deviation_mm`) is computed *before* re-basing, so re-basing
never shifts the GOOD/NG verdict. See `CalibrationManager.evaluate`.

**Uncalibrated fallback:** identity mapping, 1 px = 1 mm, `deviation_mm=None`,
tolerance check disabled. The system runs out of the box.

---

## 6. PLC contract

### Position encoding (`core/plc/register_map.py`)

A holding register is unsigned, so each coordinate takes **two** registers — an
unsigned magnitude and a separate sign register:

```
magnitude = round(abs(mm) × position_scale)     # default ×10, clamped to uint16
sign      = 1 (negative) | 2 (positive)
```
Range with the default scale: 0.0 … 6553.5 mm either side of zero. There is no
position offset — it was removed so the PLC reads a plain millimetre value with no
arithmetic to undo.

**Sign `0` (`SIGN_NONE`) is the no-hole sentinel**, written to both sign registers
with the magnitudes zeroed. Zero is a legitimate magnitude (a hole exactly on
centre), so the sign register is the *only* unambiguous place to say "no
measurement". A station with no sign registers wired up therefore reports no-hole
as a plain 0/0, indistinguishable from a centred hole — wire the sign registers.

### Register map — live `config/plc.json`

| Addr | Dir | Purpose |
|---|---|---|
| 100 | PLC→PC | Trigger (0→1 rising edge starts a cycle) |
| 101 | PLC→PC | Machine number |
| 102 | PC→PLC | Heartbeat (toggles every 500 ms — PLC watchdogs the PC) |
| 103 | PLC→PC | **`model_select`** — machine-model code, polled every 1000 ms |
| 110–117 | PC→PLC | Camera 1–4 hole X/Y — unsigned magnitude (encoded as above) |
| 118 | PC→PLC | Overall result: 1=GOOD, 2=NG, 3=ERROR |
| 119 | PC→PLC | Vision complete (PC sets 1; PLC reads, resets 119 + trigger) |
| 120–127 | PC→PLC | **`camera_jog`** — physical camera *mount* X/Y (actuators) |
| 128–131 | PC→PLC | **`camera_results`** — per-camera GOOD/NG/ERROR |
| 132–135 | PLC→PC | **`camera_triggers`** — inspect camera N alone (0→1 edge) |
| 136–139 | PC→PLC | **`camera_vision_complete`** — camera N's own completion handshake |
| 140–143 | PC→PLC | **`camera_status`** — 1 = camera N usable, 0 = disconnected/failing |
| 144–151 | PC→PLC | **`camera_position_signs`** — sign of 110–117: 1=neg, 2=pos, 0=no hole |

Bolded rows are **newer than [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**, which documents only 100–119.

**Camera status (140–143)** is pushed by `PlcPollWorker._publish_camera_status`, not by
the camera state callbacks — that keeps all PLC I/O on the PLC thread, and a change
that happens while the link is down is not lost (the cache clears on link loss, so the
next healthy tick re-publishes every camera). Written only on change. The value comes
from `CameraHealth.healthy` via `Application._camera_availability`, so **connected but
failing to grab reports 0**, not 1 — the safe direction for a PLC gating the station.

**Two independent handshakes.** Register 100 runs all four cameras and answers on
118/119. Register 132+N runs *only* camera N and answers on 136+N — the other
cameras' registers and the overall 118/119 are deliberately left untouched, because
those cameras were not inspected and their last values still stand. Per-camera
triggers are polled on every 50 ms tick with the same baseline-after-reconnect rule as
the global trigger.

`camera_position_signs` (144–151) pairs one-to-one with `camera_positions`: 144/145
carry the sign of camera 1's X/Y, 146/147 camera 2's, and so on. It is optional per
camera — `RegisterMap.camera_position_signs` simply has no entry, and
`PlcManager._write_position` skips the sign write — so an older `plc.json` keeps
loading. It is written in the same transaction as the magnitudes when the pair is
contiguous, immediately after them.

Do not confuse `camera_positions` (110–117, the *detected hole* coordinate, an
inspection output) with `camera_jog` (120–127, the *camera mount's* physical
position, driven by PLC actuators). Both are per-camera X/Y pairs; they mean
completely different things.

Write order in `write_inspection_output` matters: positions (magnitudes then signs)
→ per-camera results → overall result → `vision_complete=1` last, because the PLC may
read the moment `vision_complete` goes high. Contiguous X/Y pairs are written in one
transaction.

### Fault policy
Heartbeat loss lets the PLC stop the line. On any vision fault the PC writes
result 3 (ERROR) so the PLC never dead-waits. `PlcManager` auto-reconnects with
escalating backoff (1s/2s/5s/10s), driven by `ensure_connected()` on each poll tick.

### Edge detection subtlety
After a (re)connect, `PlcPollWorker` takes the first trigger value as a **baseline,
never an edge** — a trigger frozen high cannot re-fire. `model_select` behaves the
*opposite* way on purpose: the first read after reconnect **does** fire, so whichever
model is already selected loads immediately.

### Protocols
`simulated` (**current**), `modbus_tcp` (pymodbus, lazy import), `slmp`
(MELSEC 3E binary over raw sockets, no vendor lib — talks to an iQ-R/Q CPU's built-in
Ethernet port, which does not speak Modbus natively). SLMP maps register N → `D<N>`.
SLMP frame variant selected by `connection.slmp_frame`: `iq_r` (default) or `q`.

---

## 7. Threading model

| Thread | Cadence | Job |
|---|---|---|
| Main (Qt) | event loop | rendering + input only; never blocks |
| `PlcPollWorker` (QThread) | 50 ms | trigger edge detection, heartbeat, model poll, reconnect driving |
| `InspectionWorker` (QObject on QThread) | per trigger | runs the pipeline; re-entrant triggers dropped with a loud warning |
| `AcquisitionWorker` ×N (QThread) | `ui.live_preview_fps` | live preview grabs; 2 s backoff on fault |
| `DatabaseWorker` (QThread) | batched | log persistence only |
| detection pool | per cycle | `ThreadPoolExecutor`, parallel mode only |

**All cross-thread traffic goes through `AppState` Qt signals** (queued delivery).
`AppState` also keeps a lock-protected snapshot so a newly-opened page renders current
values immediately instead of waiting for the next event.

The poll loop must never block longer than one tick or the heartbeat jitters — that's
why the pipeline lives on its own thread. PLC writes from the inspection thread
interleave safely with polling via the client's internal per-transaction lock; a
*sequence* of writes is not atomic across threads, so multi-register workflows stay
on one thread.

Inspections persist **synchronously** on the inspection thread (WAL makes it
ms-fast and keeps ordering strict). Only *logs* are batched asynchronously.
`DatabaseWorker` has a feedback-loop guard: a failed flush logs with
`extra={"no_db": True}`, which it refuses to enqueue.

Shutdown order is deliberate ([main.py](main.py) `shutdown()`, idempotent):
poll (no new triggers) → acquisition → inspection (lets in-flight cycle finish) →
PLC disconnect → db worker (final flush) → engine dispose → log manager.

---

## 8. Config system

`ConfigManager` ([core/utilities/config_manager.py](core/utilities/config_manager.py)) — thread-safe, caching,
deep-copy isolation, **atomic saves** (temp file + `os.replace`), dot-path access
(`get_value("plc", "connection.ip")`), restore-from-`defaults/`, and change
notification via plain callables (Qt-free).

Five domains in `KNOWN_CONFIGS`: `app_config`, `plc`, `camera`, `detection`,
`machine_models`.

**JSON is the boot source; the DB is an audit mirror** — and only for camera and PLC.
`detection.json` and `machine_models.json` are JSON-only by deliberate precedent.

### Hot-reload wiring — asymmetric, know this before changing settings code

| Domain | Runtime effect of a save |
|---|---|
| `camera` | **Subscribed** ([main.py:215](main.py#L215)) → rebuilds `CameraManager` + acquisition workers |
| `detection` | Hot-swapped **manually** by the page, which calls `VisionEngine.apply_config(cfg)` *before* `save()` so validation happens first ([ui/detection/detection_page.py:461](ui/detection/detection_page.py#L461)) |
| `plc` | **No subscriber — requires an app restart** (see [gotchas](#9-gotchas--traps)) |
| `machine_models` | Applied live via `MachineModelService.apply_profile` |
| `app_config` | Read per-cycle in the pipeline; other keys read at startup |

### Machine-model profiles
[services/machine_model_service.py](services/machine_model_service.py) snapshots per-camera ROI/exposure +
the whole detection block into a named, `plc_code`-tagged profile. When the PLC
changes register 103, `Application._on_machine_model_changed` looks up the profile and
applies it.

Applying **never persists** `camera.json`/`detection.json` — it goes through the same
"preview, don't persist" entry points the Camera/Detection pages use
(`CameraService.apply_live`, `VisionEngine.apply_config`), so the manually maintained
baseline survives an automatic switch.

Only `_TUNABLE_CAMERA_FIELDS` are overridable (`roi`, `exposure_us`, `gain_db`,
`gamma`, `brightness`, `width`, `height`, `trigger_mode`). Identity/wiring fields
(`driver`, `connection_id`, `name`, `enabled`) describe the physical rig, not the part
— `CameraManager.apply_settings()` doesn't re-instantiate the driver anyway, so
overriding them would silently do nothing.

Camera application is **best-effort** (missing camera → warning, skipped); detection is
**all-or-nothing** (malformed block raises — it's one atomic hot-swap for the station).

---

## 9. Gotchas & traps

**PLC config saves need a restart.** `plc_service.save_config` claims the rebuild is
"wired in the composition root via `ConfigManager.subscribe("plc", ...)`"
([services/plc_service.py:56](services/plc_service.py#L56)) — **that subscription does not
exist.** [main.py:215](main.py#L215) subscribes `camera` only. The `RegisterMap`,
client and `PlcManager` are all built once at startup from the boot-time dict. Editing
the PLC page writes JSON + DB but does not change the running client. Either add the
subscriber (and handle rebuilding the client under the running poll worker) or leave
it and fix the docstring — don't assume it works.

**`ui.live_preview_fps` is `0` — live preview is deliberately off.** `fps <= 0` makes
`create_acquisition_workers` return an **empty list**: no preview threads exist at all,
cameras are touched only when an inspection triggers, and the dashboard shows each
cycle's captured pictures instead of a stream. Don't "fix" a blank live view without
checking this. (`AcquisitionWorker` itself clamps `fps` to ≥1.0 internally, so the
zero-check in the factory is the only thing that disables preview.)

**Live config ≠ shipped defaults.** The station has been tuned away from defaults:

| Key | `config/` (live) | `config/defaults/` |
|---|---|---|
| `detection.active_detector` | `opencv` | `dark_hole` |
| camera `driver` (all 4) | `image_file` | `simulated` |
| `plc.connection.port` / `ip` | 5007 / 192.168.3.20 | 502 / 192.168.0.10 |

README describes `dark_hole` as "the default and the one to use on real parts" — that
describes the *defaults file*, not the current live setting. Both statements are
correct; don't "correct" either one.

**Camera 1 has a machine-specific absolute path.** `config/camera.json` camera 1 carries
`"image_source": "D:/washing_machine_holes_detection-main/.../test images"`, which won't
exist on another machine. Expect camera 1 to fail to connect on a fresh checkout.

**Detection size gates are in pixels of the *analysed* frame** — after ROI crop and
resolution fit. Cameras with different fields of view need different numbers, but the
detection parameters are **global** (one block for all four cameras). Tune for the
optics actually inspected with and keep the four stations comparable.

**`config/defaults/` must be updated in lockstep.** Adding a config key without adding
it to `defaults/` means "Restore Defaults" silently drops the feature. (Defaults are
currently in sync, including `model_select`, `camera_results` and `camera_jog`.)

**The PLC DB audit mirror is incomplete.** `PlcService._mirror_to_database` and the
`plc_configurations` table cover only the original 100–119 registers — `model_select`,
`camera_results` and `camera_jog` are **not** mirrored. JSON remains the source of
truth; the table is audit-only, so this is cosmetic unless you start reading from it.

**Duplicate lab scripts.** `scripts/dark_contour_lab.py` and
`development_files/dark_contour_lab.py` are **byte-identical**; the two
`dark_contour_test.py` copies have **diverged**. These are throwaway tuning labs, not
app code — the contour/shape stage was already ported into `OpenCVHoleDetector`. Don't
import from them; prefer `tools/hole_debug.py` for tuning.

**Basler GigE packet size.** Every camera ships `packet_size: 1500` (safe default).
Raising it to `8192` requires jumbo frames (MTU 9014) enabled on the shared NIC first,
verified with `ping -f -l 8000 <camera IP>`. Raising it without jumbo frames causes
dropped frames and grab timeouts, not a speedup.

**`from_config` swallows unknown keys into `extra`.** `CameraSettings.extra` is
`dict(cfg)` — the *whole* raw entry, so driver-specific blocks (`basler`, `simulation`,
`image_source`) reach the driver through it.

### Doc drift

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) is design-intent from initial generation and predates
several shipped features. It is still the best explanation of *why* the layering is
what it is, but treat these as stale:
- names a `ProcessingWorker` — the class is `InspectionWorker`
- no mention of `machine_models` (service, page, config, PLC register 103)
- no mention of the `dark_hole` detector, the `slmp` and `simulated` protocols, or the
  `image_file` camera driver
- register table stops at 119 (missing jog 120–127, per-camera results 128–131)
- §6 "Generation Plan" is a historical build checklist, not current state

---

## 10. Conventions worth matching

- **Exceptions:** everything domain-specific derives from `VisionSystemError`
  ([core/utilities/exceptions.py](core/utilities/exceptions.py)). Workers catch `VisionSystemError`,
  log, raise a UI alarm and **keep running** — a production line must not stop on a
  recoverable fault. Programming errors (`TypeError`, `AttributeError`) deliberately
  stay outside the tree and propagate to the global excepthook in `main.py`.
- **Observer callbacks never break the caller** — every fan-out site wraps callbacks in
  `try/except` + `logger.exception`. Match that in new observers.
- **Enums:** `StrEnum` for anything persisted to JSON/SQLite (stays human-readable);
  `IntEnum` only for `PlcResultCode`, which goes straight into a register.
- **Logging:** `get_logger(LogSource.X)` → `wmhd.<source>`. The source tag is parsed
  back out of the logger name in `LogEvent.from_record`, so keep the naming scheme.
- **Docstrings carry the design rationale** — this codebase explains *why* at module
  level, not just *what*. When changing behaviour, update the module docstring; it is
  frequently the only place a constraint is recorded.
- **Type hints throughout**, `from __future__ import annotations` in every module.
- DTOs are frozen/plain dataclasses, treated as immutable after emit; `frame` arrays
  are display-only and never written to downstream.

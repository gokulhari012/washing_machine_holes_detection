# Washing Machine Bottom Hole Detection System — Architecture

Industrial machine-vision desktop application (Windows) that inspects the bottom
of washing machines with 4 fixed cameras, detects hole positions, and exchanges
results with a PLC over Ethernet (Modbus TCP).

---

## 1. Layered Architecture (MVVM + Services)

```
┌─────────────────────────────────────────────────────────────────┐
│  UI LAYER (ui/)                    PySide6 views, main thread    │
│  MainWindow, Dashboard, Config pages, reusable widgets          │
└──────────────▲──────────────────────────────┬───────────────────┘
               │ Qt signals (queued)          │ slot calls
┌──────────────┴──────────────────────────────▼───────────────────┐
│  VIEW-MODEL LAYER (models/)                                     │
│  AppState (observable QObject), DTO dataclasses                 │
└──────────────▲──────────────────────────────┬───────────────────┘
               │                              │
┌──────────────┴──────────────────────────────▼───────────────────┐
│  SERVICE LAYER (services/)        orchestration, business rules │
│  InspectionService, PlcService, CameraService, DatabaseService, │
│  ExportService, BackupService, AuthService                      │
└──────────────▲──────────────────────────────┬───────────────────┘
               │                              │
┌──────────────┴───────────┐  ┌───────────────▼───────────────────┐
│  WORKER LAYER (workers/) │  │  CORE LAYER (core/)  no Qt-UI dep │
│  QThreads / QThreadPool  │  │  plc/ camera/ vision/ calibration │
│  PlcPollWorker,          │  │  database/ logging/ utilities/    │
│  AcquisitionWorker(x4),  │  │  Pure Python + OpenCV + pymodbus  │
│  ProcessingWorker,       │  │  + SQLAlchemy. Hardware drivers   │
│  DatabaseWorker          │  │  behind abstract interfaces.      │
└──────────────────────────┘  └───────────────────────────────────┘
```

Rules:

- **Dependency direction is strictly downward.** `core/` imports nothing from
  `services/`, `workers/`, or `ui/`. `ui/` never touches hardware directly.
- **Composition root** is `main.py`: it constructs core drivers, injects them
  into services (constructor injection), injects services into workers and
  views. No globals, no singletons, no service locators.
- **Strategy pattern** for detection (`HoleDetector` ABC →
  `OpenCVHoleDetector`, `TemplateMatchingDetector`, `YOLOHoleDetector`).
  Swapping the algorithm touches zero UI/PLC code.
- **Abstract driver interfaces** for cameras (`CameraBase`) and PLC
  (`PlcClientBase`) so HikRobot/Basler/Daheng/IDS SDKs or pycomm3 can be added
  as new adapter classes only.
- **Repository pattern** over SQLAlchemy; UI never writes SQL.
- All cross-thread traffic uses Qt signals/slots with queued connections
  carrying immutable DTOs (frozen dataclasses + numpy frames).

---

## 2. Module Responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | Composition root: load configs, build object graph, start workers, show MainWindow. |
| `config/*.json` | Persisted user-editable configuration (app, plc, camera, detection). |
| `core/utilities/` | `ConfigManager` (JSON load/save/validate), custom exception hierarchy, shared enums. |
| `core/logging/` | Central `LogManager`: rotating file handler, DB handler, Qt-signal handler feeding the Logs page. |
| `core/database/` | SQLAlchemy engine/session factory, ORM models, repositories. |
| `core/plc/` | `PlcClientBase` ABC, `ModbusTcpClient` adapter (pymodbus), `RegisterMap`, `PlcManager` (connect/reconnect/heartbeat). |
| `core/camera/` | `CameraBase` ABC, `SimulatedCamera`, `UsbCamera` (OpenCV), `HikRobotCamera` stub, `CameraManager` (4-camera lifecycle, health). |
| `core/vision/` | `HoleDetector` strategy ABC, concrete detectors, `DetectionResult` dataclasses, `VisionEngine` (strategy context + factory). |
| `core/calibration/` | Pixel→mm scaling, perspective (homography) correction, reference-point offset, persistence. |
| `models/` | `AppState` (QObject with signals: counters, statuses, last result), DTOs crossing thread boundaries. |
| `services/` | `InspectionService` = the trigger→capture→detect→write→store pipeline; plus per-domain services wrapping core managers for the UI. |
| `workers/` | Thread ownership only — no business logic. Each worker hosts one service loop in its own thread. |
| `ui/` | Views only: render state, forward user actions to services. One package per page + `widgets/` for reusable controls (LED indicator, zoomable image view, ROI editor, alarm banner). |
| `resources/styles/` | Dark industrial QSS theme. |
| `tests/` | pytest unit tests for core (vision on synthetic images, register map, repositories, calibration math). |

---

## 3. Database Schema (SQLite via SQLAlchemy)

```
users                    inspections                    inspection_details
─────                    ───────────                    ──────────────────
id            PK         id             PK              id             PK
username      UQ         created_at     IX (local)        inspection_id  FK→inspections IX
password_hash            machine_number IX              camera_index   1..4
role                     serial_number                  camera_name
created_at               overall_result IX (GOOD/NG/ERROR)  hole_found  bool
last_login               plc_cycle_time_ms              x_px, y_px     float
                         detection_time_ms              x_mm, y_mm     float
                         operator                       deviation_mm   float
                         shift                          confidence     float
                                                        result         (GOOD/NG/ERROR)
                                                        image_path

camera_configurations    plc_configurations             calibrations
─────────────────────    ──────────────────             ────────────
id, camera_index UQ      id (single row)                id
name, driver             ip, port, protocol, unit_id    camera_index IX
connection_id            poll_interval_ms, timeout_ms   pixels_per_mm_x / _y
exposure_us, gain_db     trigger_register               homography_json (3x3)
gamma, brightness        machine_number_register        ref_point_x_mm / _y_mm
width, height            heartbeat_register             rms_error
roi_x/y/w/h              result_register                is_active
trigger_mode             ack_register                   calibrated_by
enabled, updated_at      cam{1..4}_x/_y_register        calibrated_at
                         updated_at

system_configurations    logs
─────────────────────    ────
key PK, value            id PK, created_at IX
description, updated_at  level IX, source IX (PLC/CAMERA/VISION/DB/SYSTEM)
                         message
```

- One `inspections` row per trigger cycle; exactly four `inspection_details`
  children (cascade delete).
- Config tables mirror the JSON files (JSON = boot source, DB = audit/history).
- Retention job deletes `inspections`/`logs` older than N days (Settings page).

---

## 4. PLC Communication Flow (Modbus TCP holding registers)

Default register map (all editable on the PLC Configuration page):

| Addr | Dir | Purpose |
|---|---|---|
| 100 | PLC→PC | Trigger (0→1 edge starts inspection) |
| 101 | PLC→PC | Machine number |
| 102 | PC→PLC | Heartbeat (toggles every 500 ms) |
| 110/111 | PC→PLC | Camera 1 X / Y (mm × 10, offset +10000 for sign) |
| 112/113 | PC→PLC | Camera 2 X / Y |
| 114/115 | PC→PLC | Camera 3 X / Y |
| 116/117 | PC→PLC | Camera 4 X / Y |
| 118 | PC→PLC | Overall result (1=GOOD, 2=NG, 3=ERROR) |
| 119 | PC→PLC | Vision complete handshake (PC sets 1, PLC resets) |

Cycle:

```
PLC                       PC (this software)
 │  trigger reg 0→1        │
 │─────────────────────────▶ PlcPollWorker detects rising edge (poll ≤50 ms)
 │                          │ read machine number (reg 101)
 │                          │ capture 4 frames (parallel)
 │                          │ detect holes (QThreadPool, 4 jobs)
 │                          │ px→mm via calibration, scale to registers
 │ ◀────────────────────────│ write regs 110–118, then set 119=1
 │  PLC reads results,      │
 │  resets trigger + 119    │ queue DB write, emit dashboard update
 │─────────────────────────▶ edge detector re-arms; wait next cycle
```

Fault handling: heartbeat loss → PLC can stop the line; PC side raises an
alarm banner, `PlcManager` auto-reconnects with backoff; if inspection fails
(camera/vision error) PC writes result=3 (ERROR) so the PLC never waits forever.

---

## 5. Threading Model

| Thread | Owner | Work |
|---|---|---|
| Main (UI) | Qt event loop | Rendering + user input only. Never blocks. |
| `PlcPollWorker` (QThread) | poll cadence | Trigger **rising-edge** detection (baseline after reconnect, never a false edge), heartbeat toggle, reconnect driving. Never blocked longer than one poll tick. |
| `InspectionWorker` (QThread, worker-object) | the pipeline | Runs InspectionService per trigger — including PLC output writes, which interleave safely with polling via the client's internal lock. `trigger_requested` signal = manual/simulated cycles from the UI. |
| `AcquisitionWorker` ×4 (QThread) | one per camera | Throttled live-preview grabs (`ui.live_preview_fps`); the camera capture lock serialises with inspection grabs. Backs off on faults. |
| detection pool | inside InspectionService | ThreadPoolExecutor: 4 detections in parallel per cycle (engine serialises non-thread-safe strategies). |
| `DatabaseWorker` (QThread) | log persistence | Batched INFO+ log writes fed by the logging fan-out, with a feedback-loop guard (`no_db`). Inspections persist synchronously on the inspection worker — WAL makes it ms-fast and keeps ordering strict. |

- Frames move as numpy arrays inside frozen DTOs through queued signals
  (Qt copies the reference safely; DTOs are never mutated after emit).
- Live preview is throttled (~15 fps to UI) independent of acquisition rate.
- Every worker has `start()/stop()` with graceful shutdown joined in
  `MainWindow.closeEvent`.

---

## 6. Generation Plan

| Step | Content |
|---|---|
| 1 ✅ | Folder skeleton + this document |
| 2 | `config/` JSONs, `core/utilities/`, `core/logging/`, `requirements.txt` |
| 3 | `core/database/` (ORM models, engine, repositories) |
| 4 | `core/plc/` |
| 5 | `core/camera/` |
| 6 | `core/vision/` + `core/calibration/` |
| 7 | `models/` + `services/` |
| 8 | `workers/` |
| 9 | `ui/widgets/` + dark theme QSS |
| 10 | `ui/main_window.py` + `ui/dashboard/` |
| 11 | Remaining UI pages (camera, plc, detection, calibration, database, logs, settings) |
| 12 | `main.py`, `README.md`, `tests/` |

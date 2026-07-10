# Washing Machine Bottom Hole Detection System

Industrial machine-vision desktop application (Windows, PySide6) that inspects
the bottom of washing machines on a conveyor: **4 fixed cameras** detect the
drain-hole position in each section, results are exchanged with the line
**PLC over Ethernet (Modbus TCP)**, and every inspection is stored in SQLite.

Dark industrial HMI in the style of VisionPro / In-Sight: dashboard with four
live camera panels, camera/PLC/detection/calibration configuration pages,
searchable database viewer with CSV/Excel/PDF export, live logs, and
password-protected settings.

---

## Quick start (no hardware required)

```bash
pip install -r requirements.txt
python main.py
```

The shipped configuration runs in **simulation mode**: four synthetic cameras
render a brushed-metal machine bottom with a drain hole, and a simulated PLC
raises a trigger every few seconds — you immediately see live inspection
cycles on the dashboard. The **Simulate Trigger** toolbar button runs a
manual cycle at any time.

Default settings login: `admin` / `admin` — **change it in Settings before
production use**.

Smoke test (starts hidden, runs 8 s, saves `logs/selftest.png`, exits):

```bash
python main.py --selftest 8
```

Unit tests:

```bash
pytest
```

## Going to production

1. **PLC** — in `config/plc.json` (or the PLC page) set
   `connection.protocol` to `"modbus_tcp"`, the PLC IP/port/unit id, and the
   register map. Default layout (16-bit holding registers):

   | Addr | Direction | Purpose |
   |------|-----------|---------|
   | 100  | PLC → PC  | Trigger (0→1 rising edge starts an inspection) |
   | 101  | PLC → PC  | Machine number |
   | 102  | PC → PLC  | Heartbeat (toggles every 500 ms) |
   | 110–117 | PC → PLC | Camera 1–4 hole X/Y. `raw = mm × 10 + 10000`; raw `0` = no hole |
   | 118  | PC → PLC  | Result: 1 = GOOD, 2 = NG, 3 = ERROR |
   | 119  | PC → PLC  | Vision complete (PC sets 1; PLC reads results, resets 119 and the trigger) |

   The PC toggles the heartbeat so the PLC can watchdog it; on any vision
   fault the PC writes result 3 (ERROR) so the PLC never dead-waits.

2. **Cameras** — on the Cameras page set each camera's driver:
   `usb` (OpenCV/DirectShow, `connection_id` = device index) or `hikrobot`
   (requires the MVS SDK — see the integration checklist in
   `core/camera/hikrobot_camera.py`). Basler/Daheng/IDS: implement a
   `CameraBase` adapter and register it in `core/camera/__init__.py`.

3. **Calibration** — per camera on the Calibration page: pixel→mm scale (or a
   4-point homography for perspective correction) and the reference (nominal)
   hole position. The position tolerance judgement activates only for
   calibrated cameras.

4. **Detection** — tune thresholds on the Detection page with the live
   test view. Strategies (`opencv`, `template_matching`, `yolo`) are
   hot-swappable; adding one means implementing `HoleDetector` and
   registering it in `core/vision/vision_engine.py` — no UI/PLC changes.

## Architecture

Layered MVVM: `ui/` → `models/` (AppState + DTOs) → `services/` → `workers/`
+ `core/` (Qt-free hardware/domain layer). `main.py` is the only composition
root; hardware is abstracted behind `CameraBase` / `PlcClientBase` adapters;
detection uses the strategy pattern. Threads: PLC poll (50 ms edge detection
+ heartbeat), inspection pipeline, 4 preview grabbers, batched DB log writer —
all UI communication via queued Qt signals. Full details, schema and
communication flow: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
main.py            entry point / composition root
config/            app_config · plc · camera · detection (+ defaults/)
core/              plc/ camera/ vision/ calibration/ database/ logging/ utilities/
models/            AppState (observable) + DTOs
services/          inspection · camera · plc · database · export · backup · auth
workers/           PlcPollWorker · InspectionWorker · AcquisitionWorker×4 · DatabaseWorker
ui/                main_window + dashboard/ camera/ plc/ detection/ calibration/
                   database/ logs/ settings/ widgets/ (dark QSS theme in resources/)
tests/             pytest suite (vision, PLC, repositories, calibration)
```

Runtime artifacts are created beside the app: `data/` (SQLite, WAL),
`images/` (annotated inspection photos, per-day folders), `logs/`,
`backups/` (daily online backups + retention purge, see Settings).

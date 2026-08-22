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

   The `image_file` driver needs no hardware: press **Choose Image…** (or
   **Folder…**) to pick a picture, then **Save Configuration**. That picture
   is streamed as the camera's frames, so it shows up in the live preview and
   is inspected by every trigger exactly like a real feed — the quickest way
   to try detection on your own photos.

3. **Calibration** — per camera on the Calibration page: pixel→mm scale (or a
   4-point homography for perspective correction) and the reference (nominal)
   hole position. The position tolerance judgement activates only for
   calibrated cameras.

4. **Detection** — tune thresholds on the Detection page with the live
   test view. Strategies (`dark_hole`, `opencv`, `template_matching`, `yolo`)
   are hot-swappable; adding one means implementing `HoleDetector` and
   registering it in `core/vision/vision_engine.py` — no UI/PLC changes.

   `dark_hole` is the default and the one to use on real parts. It scores each
   pixel by how much darker it is than its *own surroundings*, so uneven
   lighting, shadowed corners and specular bands do not need a threshold of
   their own, and it fits a circle to whatever arc of the rim is visible — a
   bore half-hidden behind a slot edge is still found, and still reports the
   centre and diameter of the whole bore. `Hole.circularity` carries how much
   of the rim was actually visible (1.0 = all of it, ~0.5 = half).

   To match it to a station, run the sample images through the tuner — it
   prints what was accepted, what was rejected and why, writes a stage-by-stage
   montage to `logs/hole_debug/`, and suggests the size gates:

   ```
   python tools/hole_debug.py path/to/sample_images/
   ```

   `--camera N` runs the pictures through that camera's entry in camera.json
   (resolution fit and ROI crop included), so the tuner judges the same pixels
   the line will. The quickest way to try a photo end to end is the
   `image_file` camera driver (step 2): choose the picture, save, then use
   **Test on Camera** on the Detection page.

   The size gates are in pixels **of the analysed frame** — after the ROI crop
   and any resolution fit. Cameras with different fields of view therefore need
   different numbers, and the detection parameters are global: tune them for
   the optics you actually inspect with, and keep the four stations
   comparable.

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

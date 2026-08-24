"""Bring up a Basler GigE camera and prove the Ethernet link works.

Run it before touching the main application — it uses the very same adapter
and the very same camera.json entry the line uses, so a green result here
means the station will connect too::

    python tools/basler_probe.py --list                 # what is on the network
    python tools/basler_probe.py --camera 1             # grab 5 frames, save them
    python tools/basler_probe.py --camera 1 -n 30 --exposure 4000
    python tools/basler_probe.py --serial 40123456      # ignore camera.json

Frames land in ``logs/basler_probe/`` with the measured grab time printed per
frame, which is the quickest way to tell a packet-size problem (slow, jittery,
occasional timeouts) from an exposure problem (fast but black).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np

from core.camera.basler_camera import BaslerCamera, enumerate_devices
from core.camera.camera_base import CameraSettings
from core.utilities.exceptions import VisionSystemError

CONFIG_PATH = BASE_DIR / "config" / "camera.json"
OUTPUT_DIR = BASE_DIR / "logs" / "basler_probe"


def list_devices(gige_only: bool) -> int:
    """Print every Basler camera the pylon runtime can see."""
    devices = enumerate_devices(gige_only=gige_only)
    if not devices:
        print(
            "No Basler camera found.\n"
            "  - is the Ethernet cable in and the camera powered (PoE or aux)?\n"
            "  - does the camera hold an IP on this PC's subnet? "
            "(open the pylon IP Configurator)\n"
            "  - is the pylon GigE filter driver bound to that NIC, "
            "and the firewall out of the way?"
        )
        return 1
    for position, device in enumerate(devices):
        print(
            f"[{position}] {device['model']}  sn={device['serial']}  "
            f"ip={device['ip'] or 'n/a'}  mac={device['mac'] or 'n/a'}  "
            f"class={device['device_class']}  name={device['user_name'] or '-'}"
        )
    print("\nPut the serial (or the IP) into connection_id in config/camera.json.")
    return 0


def load_entry(camera_index: int) -> dict:
    """One ``cameras[]`` entry of camera.json."""
    entries = json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("cameras", [])
    entry = next((e for e in entries if int(e.get("index", -1)) == camera_index), None)
    if entry is None:
        raise SystemExit(f"No camera {camera_index} in {CONFIG_PATH}")
    return dict(entry)


def build_settings(args: argparse.Namespace) -> CameraSettings:
    """Camera settings from camera.json, with the CLI overrides applied.

    ``--serial`` / ``--ip`` build a standalone entry instead, so an unlisted
    camera can be tested before it is added to the configuration.
    """
    if args.serial or args.ip:
        entry = {
            "index": args.camera,
            "name": f"Basler {args.serial or args.ip}",
            "driver": "basler",
            "connection_id": args.serial or args.ip,
        }
    else:
        entry = load_entry(args.camera)
        entry["driver"] = "basler"  # probe the hardware even if the entry says image_file

    if args.exposure is not None:
        entry["exposure_us"] = args.exposure
    if args.gain is not None:
        entry["gain_db"] = args.gain
    if args.trigger is not None:
        entry["trigger_mode"] = args.trigger
    if args.packet_size is not None:
        entry["basler"] = dict(entry.get("basler", {}), packet_size=args.packet_size)
    if args.full_frame:
        entry["roi"] = {"x": 0, "y": 0, "width": 0, "height": 0}
    return CameraSettings.from_config(entry)


def report(frame: np.ndarray, elapsed_ms: float, position: int) -> None:
    """One line per frame: shape, brightness and how long the grab took."""
    print(
        f"  frame {position:>3}  {frame.shape[1]}x{frame.shape[0]}"
        f"{'' if frame.ndim == 2 else f'x{frame.shape[2]}'}  "
        f"{frame.dtype}  mean={frame.mean():6.1f}  "
        f"min={frame.min():3d}  max={frame.max():3d}  {elapsed_ms:6.1f} ms"
    )


def grab_frames(args: argparse.Namespace) -> int:
    """Connect, grab ``--frames`` frames, save them, report the timing."""
    settings = build_settings(args)
    camera = BaslerCamera(settings)

    print(f"Connecting to {settings.name} (connection_id={settings.connection_id!r}) ...")
    try:
        camera.connect()
    except VisionSystemError as exc:
        print(f"FAILED: {exc}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    durations: list[float] = []
    failures = 0
    try:
        print(f"Connected. Grabbing {args.frames} frame(s) in {settings.trigger_mode.value} mode:")
        for position in range(1, args.frames + 1):
            started = time.perf_counter()
            try:
                frame = camera.capture()
            except VisionSystemError as exc:
                failures += 1
                print(f"  frame {position:>3}  FAILED: {exc}")
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            durations.append(elapsed_ms)
            report(frame, elapsed_ms, position)
            if not args.no_save:
                path = OUTPUT_DIR / f"cam{settings.index}_{stamp}_{position:03d}.png"
                cv2.imwrite(str(path), frame)
    finally:
        camera.disconnect()

    if durations:
        print(
            f"\n{len(durations)} frame(s) ok, {failures} failed — "
            f"grab min/mean/max = {min(durations):.1f}/"
            f"{sum(durations) / len(durations):.1f}/{max(durations):.1f} ms"
        )
        if not args.no_save:
            print(f"Saved to {OUTPUT_DIR}")
    return 1 if failures or not durations else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list visible Basler cameras and exit")
    parser.add_argument("--gige-only", action="store_true", help="--list: hide USB3 cameras")
    parser.add_argument("--camera", type=int, default=1, help="camera.json index (default 1)")
    parser.add_argument("--serial", help="probe this serial number instead of camera.json")
    parser.add_argument("--ip", help="probe this IP address instead of camera.json")
    parser.add_argument("-n", "--frames", type=int, default=5, help="frames to grab (default 5)")
    parser.add_argument("--exposure", type=int, help="override exposure_us")
    parser.add_argument("--gain", type=float, help="override gain_db")
    parser.add_argument(
        "--trigger",
        choices=["software", "hardware", "continuous"],
        help="override trigger_mode",
    )
    parser.add_argument("--packet-size", type=int, help="GigE packet size, e.g. 8192 for jumbo")
    parser.add_argument("--full-frame", action="store_true", help="ignore the configured ROI")
    parser.add_argument("--no-save", action="store_true", help="measure only, write no files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.list:
            return list_devices(args.gige_only)
        return grab_frames(args)
    except VisionSystemError as exc:
        print(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

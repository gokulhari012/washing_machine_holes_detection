"""Tune and troubleshoot the ``dark_hole`` detector against real photos.

Run it on a picture (or a folder of pictures) straight from the line::

    python tools/hole_debug.py samples/                 # whole folder
    python tools/hole_debug.py samples/part_07.png      # one picture
    python tools/hole_debug.py samples/ --min-d 30 --max-d 160   # override a gate

For every image it prints what the detector saw — accepted bores and, just as
important, the blobs it *rejected* and why — then writes a four-panel picture
to ``logs/hole_debug/`` showing the stages: source, black-hat response, mask,
and the result overlay. When nothing is found, that montage tells you in one
glance which stage lost the bore.

It ends with a suggested ``dark_hole`` block. Paste the numbers into the
Detection page (or detection.json) once they look right; the tool runs the
production code path, so agreement here means agreement on the line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np

from core.camera.camera_base import CameraSettings
from core.camera.image_file_camera import IMAGE_PATTERNS, ImageFileCamera, read_image
from core.vision.dark_hole_detector import Candidate, DarkHoleDetector
from core.vision.detection_result import DetectionResult
from core.vision.vision_engine import draw_detection_overlay

PANEL_LABELS = ("1 source", "2 darker-than-surroundings", "3 mask", "4 result")


def camera_block(config_path: Path, camera_index: int | None) -> dict:
    """One camera's block from the per-camera detection.json (``--camera``
    defaults to camera 1 when not given, since there is no longer a single
    global block)."""
    if not config_path.is_file():
        return {}
    document = json.loads(config_path.read_text(encoding="utf-8"))
    return document.get("cameras", {}).get(str(camera_index or 1), {})


def load_params(config_path: Path, camera_index: int | None, overrides: argparse.Namespace) -> dict:
    """``dark_hole`` block for one camera, with any CLI overrides applied."""
    params: dict = dict(camera_block(config_path, camera_index).get("dark_hole", {}))
    if overrides.min_d is not None:
        params["min_hole_diameter_px"] = overrides.min_d
    if overrides.max_d is not None:
        params["max_hole_diameter_px"] = overrides.max_d
    if overrides.min_contrast is not None:
        params["min_contrast"] = overrides.min_contrast
    if overrides.channel is not None:
        params["channel"] = overrides.channel
    return params


def live_loader(camera_index: int | None):
    """Return a reader that produces frames exactly as the pipeline sees them.

    With ``--camera N`` the picture goes through that camera's entry in
    camera.json — resolution fit and ROI crop included — so the tuner works on
    the same pixels the line will judge, not on the raw file.
    """
    if camera_index is None:
        return read_image

    config_path = BASE_DIR / "config" / "camera.json"
    entries = json.loads(config_path.read_text(encoding="utf-8")).get("cameras", [])
    entry = next((e for e in entries if int(e.get("index", -1)) == camera_index), None)
    if entry is None:
        raise SystemExit(f"No camera {camera_index} in {config_path}")

    def read_through_camera(path: Path):
        camera = ImageFileCamera(
            CameraSettings.from_config(dict(entry, image_source=str(path)))
        )
        try:
            camera.connect()
            return camera.capture()
        except Exception as exc:  # unreadable file, unsupported format
            print(f"  {path.name}: {exc}")
            return None

    return read_through_camera


def gather(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for pattern in IMAGE_PATTERNS for p in target.glob(pattern))
    return [target]


def montage(image: np.ndarray, stages: dict, result: DetectionResult) -> np.ndarray:
    """Source | black-hat response | mask | overlay, side by side."""
    panels = [
        image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        cv2.applyColorMap(stages["response"], cv2.COLORMAP_INFERNO),
        cv2.cvtColor(stages["mask"], cv2.COLOR_GRAY2BGR),
        draw_detection_overlay(image, result),
    ]
    height = max(panel.shape[0] for panel in panels)
    scaled = []
    for panel, label in zip(panels, PANEL_LABELS):
        if panel.shape[0] != height:
            scale = height / panel.shape[0]
            panel = cv2.resize(panel, (int(panel.shape[1] * scale), height))
        panel = panel.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 26), (25, 25, 25), -1)
        cv2.putText(
            panel, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1,
            cv2.LINE_AA,
        )
        scaled.append(panel)
    return np.hstack(scaled)


def describe(candidate: Candidate, threshold: float) -> str:
    hole = candidate.hole
    verdict = candidate.rejected_because or (
        "ACCEPTED" if hole.confidence >= threshold
        else f"below confidence threshold ({hole.confidence:.2f} < {threshold:.2f})"
    )
    return (
        f"    ({hole.x_px:7.1f}, {hole.y_px:7.1f})  d={hole.diameter_px:6.1f} px  "
        f"visible {candidate.fill:4.0%}  rim {hole.circularity:4.0%}  "
        f"contrast {candidate.contrast:4.2f}  conf {hole.confidence:4.2f}   {verdict}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=Path, help="image file or folder of images")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config" / "detection.json")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "logs" / "hole_debug")
    parser.add_argument("--confidence", type=float, default=None,
                        help="confidence threshold (default: the one in detection.json)")
    parser.add_argument("--min-d", type=float, help="override min_hole_diameter_px")
    parser.add_argument("--max-d", type=float, help="override max_hole_diameter_px")
    parser.add_argument("--min-contrast", type=float, help="override min_contrast")
    parser.add_argument("--channel", choices=["auto", "gray", "red", "green", "blue"])
    parser.add_argument(
        "--camera", type=int, metavar="N",
        help="feed the pictures through camera N of camera.json (same resize + ROI "
             "crop as the live pipeline) instead of reading them raw",
    )
    args = parser.parse_args()

    images = gather(args.target)
    if not images:
        print(f"No images found at {args.target}")
        return 1

    params = load_params(args.config, args.camera, args)
    threshold = args.confidence
    if threshold is None:
        common = camera_block(args.config, args.camera).get("common", {})
        threshold = float(common.get("confidence_threshold", 0.6))
    threshold = threshold if threshold is not None else 0.6

    detector = DarkHoleDetector(params)
    load = live_loader(args.camera)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.camera is not None:
        print(f"frames taken through camera {args.camera} (resolution fit + ROI crop)")
    else:
        print("no --camera given; using camera 1's dark_hole block")
    print(f"dark_hole parameters: {json.dumps(params, sort_keys=True)}")
    print(f"confidence threshold: {threshold}\n")

    found_diameters: list[float] = []
    hit_count = 0
    for path in images:
        image = load(path)
        if image is None:
            print(f"{path.name}: cannot read")
            continue

        stages, candidates = detector.explain(image)
        result = detector.detect(image)
        accepted = [hole for hole in result.holes if hole.confidence >= threshold]
        hit_count += bool(accepted)
        found_diameters.extend(hole.diameter_px for hole in accepted)

        status = (
            f"{len(accepted)} hole(s)" if accepted
            else ("NOTHING ACCEPTED" if candidates else "NOTHING EVEN CONSIDERED")
        )
        print(f"{path.name}  [{image.shape[1]}x{image.shape[0]}]  ->  {status}")
        for candidate in candidates[:8]:
            print(describe(candidate, threshold))
        if not candidates:
            print("    no blob survived the size pre-gate: check min/max diameter, "
                  "and whether the bore is dark enough (lower min_contrast)")

        destination = args.out / f"{path.stem}_stages.png"
        cv2.imencode(".png", montage(image, stages, result))[1].tofile(str(destination))
    print(f"\nstage montages -> {args.out}")

    print(f"\n{hit_count}/{len(images)} image(s) produced a hole.")
    if found_diameters:
        smallest, largest = min(found_diameters), max(found_diameters)
        print(
            "suggested gates for this set (+/-35% headroom):\n"
            f'  "min_hole_diameter_px": {max(4, int(smallest * 0.65))},\n'
            f'  "max_hole_diameter_px": {int(largest * 1.35)}'
        )
    else:
        print(
            "Nothing detected. In order, try:\n"
            "  1. widen --min-d/--max-d to bracket the bore's pixel diameter\n"
            "     (open the montage and measure it if unsure)\n"
            "  2. lower --min-contrast (default 18) if the bore is only slightly\n"
            "     darker than the metal around it\n"
            "  3. --channel red for a red ring light, if 'auto' guessed wrong"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

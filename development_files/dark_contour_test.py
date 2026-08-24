"""Standalone visual test for dark circular/elliptical contour detection.

Not part of the pytest suite — this is a quick, runnable script for eyeballing
how a threshold + contour pipeline picks out dark, round-ish blobs (screw
holes, mounting holes) on the washing machine panel photos, and for tuning
parameters before they get ported into ``core/vision``.

Usage
-----
    python scripts/dark_contour_test.py
        # runs on every image in "test images/" (or a synthetic frame if
        # that folder is empty/missing) and writes annotated PNGs to
        # scripts/output/

    python scripts/dark_contour_test.py "test images/Image_20260820105712254.bmp"
        # run on one specific image

    python scripts/dark_contour_test.py --threshold 45 --min-diameter 10 --max-diameter 60

Pipeline
--------
1. grayscale -> Gaussian blur
2. inverse binary threshold (fixed or Otsu/adaptive) — holes are darker
   than the surrounding metal
3. morphological close to fuse rim speckle
4. external contours -> area gate -> circularity gate (4*pi*A/P^2)
5. contours with >= 5 points get ``cv2.fitEllipse``; the minor/major axis
   ratio decides "circle" (near 1.0) vs "ellipse" (perspective-skewed hole)
6. results are drawn on the original image and written next to a debug
   view of the threshold mask
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "test images"
DEFAULT_OUTPUT_DIR = BASE_DIR / "scripts" / "output"

_GREEN = (80, 220, 80)     # circle
_CYAN = (220, 200, 60)     # ellipse
_RED = (70, 70, 230)       # rejected (debug overlay only, unused by default)


@dataclass
class Blob:
    x: float
    y: float
    major_axis: float
    minor_axis: float
    angle: float
    circularity: float
    aspect_ratio: float
    shape: str  # "circle" | "ellipse"

    @property
    def diameter(self) -> float:
        return (self.major_axis + self.minor_axis) / 2.0


def find_dark_blobs(
    image: np.ndarray,
    *,
    threshold: int = 60,
    adaptive: bool = False,
    blur_kernel: int = 5,
    morph_kernel: int = 5,
    morph_iterations: int = 1,
    min_diameter: float = 8.0,
    max_diameter: float = 120.0,
    min_circularity: float = 0.55,
    min_aspect_ratio: float = 0.35,
) -> tuple[list[Blob], np.ndarray]:
    """Return (blobs, binary_mask) for dark round contours in *image*."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    k = max(1, blur_kernel | 1)  # force odd
    blurred = cv2.GaussianBlur(gray, (k, k), 0)

    if adaptive:
        block_size = max(3, (min(gray.shape) // 8) | 1)
        mask = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=block_size, C=10,
        )
    else:
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel | 1, morph_kernel | 1)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = math.pi * (min_diameter / 2.0) ** 2 * 0.5  # cheap pre-gate
    blobs: list[Blob] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        if len(contour) >= 5:
            (cx, cy), (minor, major), angle = cv2.fitEllipse(contour)
            if minor > major:
                minor, major = major, minor
        else:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            major = minor = 2.0 * radius
            angle = 0.0

        if not (min_diameter <= (major + minor) / 2.0 <= max_diameter):
            continue

        aspect_ratio = minor / major if major > 0 else 0.0
        if aspect_ratio < min_aspect_ratio:
            continue  # too elongated to be a hole (scratch / shadow streak)

        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

        blobs.append(
            Blob(
                x=float(cx), y=float(cy),
                major_axis=float(major), minor_axis=float(minor),
                angle=float(angle),
                circularity=float(min(1.0, circularity)),
                aspect_ratio=float(aspect_ratio),
                shape="circle" if aspect_ratio >= 0.85 else "ellipse",
            )
        )

    blobs.sort(key=lambda b: b.circularity, reverse=True)
    return blobs, mask


def draw_overlay(image: np.ndarray, blobs: list[Blob]) -> np.ndarray:
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for i, blob in enumerate(blobs):
        color = _GREEN if blob.shape == "circle" else _CYAN
        center = (int(round(blob.x)), int(round(blob.y)))
        axes = (max(1, int(blob.major_axis / 2)), max(1, int(blob.minor_axis / 2)))
        cv2.ellipse(out, center, axes, blob.angle, 0, 360, color, 2)
        cv2.drawMarker(out, center, color, cv2.MARKER_CROSS, 10, 1)
        cv2.putText(
            out, f"{i}:{blob.shape[0]} d={blob.diameter:.0f} c={blob.circularity:.2f}",
            (center[0] + 10, max(14, center[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    cv2.putText(
        out, f"{len(blobs)} dark round contour(s)", (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA,
    )
    return out


def make_synthetic_frame() -> np.ndarray:
    """Fallback test image when no input images are available: a mid-grey
    panel with two round holes, one elliptical (perspective-skewed) hole,
    a bright glare patch, and small bolt-head distractors below the size
    gate — mirrors tests/test_vision.py's synthetic ground truth."""
    image = np.full((480, 640), 110, dtype=np.uint8)
    cv2.circle(image, (160, 140), 22, 15, -1)       # round hole
    cv2.circle(image, (420, 320), 30, 20, -1)        # round hole
    cv2.ellipse(image, (300, 380), (34, 18), 15, 0, 360, 18, -1)  # skewed hole
    for center in (80, 90), (560, 60), (500, 420):
        cv2.circle(image, center, 6, 50, -1)         # bolt distractors
    cv2.circle(image, (540, 160), 60, 230, -1)       # glare patch
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def process_image(path: Path | None, image: np.ndarray, args: argparse.Namespace, outdir: Path) -> None:
    label = path.name if path else "synthetic"
    blobs, mask = find_dark_blobs(
        image,
        threshold=args.threshold,
        adaptive=args.adaptive,
        blur_kernel=args.blur,
        morph_kernel=args.morph_kernel,
        morph_iterations=args.morph_iterations,
        min_diameter=args.min_diameter,
        max_diameter=args.max_diameter,
        min_circularity=args.min_circularity,
        min_aspect_ratio=args.min_aspect_ratio,
    )

    print(f"\n{label}: {len(blobs)} dark round contour(s)")
    for i, blob in enumerate(blobs):
        print(
            f"  [{i}] {blob.shape:7s} center=({blob.x:.1f},{blob.y:.1f}) "
            f"diameter={blob.diameter:.1f}px axes=({blob.major_axis:.1f},{blob.minor_axis:.1f}) "
            f"circularity={blob.circularity:.2f} aspect={blob.aspect_ratio:.2f}"
        )

    overlay = draw_overlay(image, blobs)
    stem = Path(label).stem
    cv2.imwrite(str(outdir / f"{stem}_overlay.png"), overlay)
    cv2.imwrite(str(outdir / f"{stem}_mask.png"), mask)

    if args.show:
        cv2.imshow(f"{label} — overlay", overlay)
        cv2.imshow(f"{label} — mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path, help="image file(s) to scan; default: all of 'test images/'")
    parser.add_argument("--threshold", type=int, default=60, help="fixed inverse-threshold cut (default: 60)")
    parser.add_argument("--adaptive", action="store_true", help="use adaptive threshold instead of a fixed cut")
    parser.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel size (default: 5)")
    parser.add_argument("--morph-kernel", type=int, default=5, help="morphology close kernel size (default: 5)")
    parser.add_argument("--morph-iterations", type=int, default=1)
    parser.add_argument("--min-diameter", type=float, default=8.0, help="px (default: 8)")
    parser.add_argument("--max-diameter", type=float, default=120.0, help="px (default: 120)")
    parser.add_argument("--min-circularity", type=float, default=0.55, help="4*pi*A/P^2 gate (default: 0.55)")
    parser.add_argument("--min-aspect-ratio", type=float, default=0.35, help="minor/major axis gate (default: 0.35)")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show", action="store_true", help="also pop up cv2.imshow windows")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    paths = args.images
    if not paths and DEFAULT_INPUT_DIR.is_dir():
        paths = sorted(DEFAULT_INPUT_DIR.glob("*.bmp")) + sorted(DEFAULT_INPUT_DIR.glob("*.png")) \
            + sorted(DEFAULT_INPUT_DIR.glob("*.jpg"))

    if not paths:
        print("No input images found — running on a synthetic test frame instead.")
        process_image(None, make_synthetic_frame(), args, args.outdir)
        print(f"\nWrote overlay/mask PNGs to {args.outdir}")
        return 0

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  ! could not read {path}")
            continue
        process_image(path, image, args, args.outdir)

    print(f"\nWrote overlay/mask PNGs to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
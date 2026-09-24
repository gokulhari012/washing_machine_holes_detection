"""Single YOLO training run, driven by the Detection page (or by hand).

Companion to ``training_all_folder.py``, which loops Camera_1..Camera_6 with
hard-coded paths for the brush project. This one trains **one** dataset with
everything passed in, because the Detection page's "Dataset & Training" dialog
has to hand it operator-chosen folders that may contain spaces.

Why a script rather than the ``yolo`` CLI the help.txt describes: the app
launches this with ``sys.executable``, so it always runs in the *same*
interpreter the station's app runs in — no dependence on ``yolo.exe`` being on
PATH, and no shell quoting to get wrong when a dataset folder is called
``C:/My Parts/cam 2``. Everything is passed as separate argv entries.

It prints one machine-readable line when it succeeds::

    WMHD_BEST_WEIGHTS=<absolute path to best.pt>

which is how the dialog offers to load the freshly trained weights straight
into the camera's ``model_path``. Run by hand:

    python "run_training.py" --dataset D:/data/cam2 --output D:/runs \\
        --model yolo11n.pt --epochs 100 --imgsz 640 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: Parsed by the training worker to recover the trained weights.
BEST_WEIGHTS_PREFIX = "WMHD_BEST_WEIGHTS="


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one YOLO detection model")
    parser.add_argument("--data", required=True, help="Path to the generated data.yaml")
    parser.add_argument("--output", required=True, help="Directory training results go into")
    parser.add_argument("--name", default="train", help="Run name inside --output")
    parser.add_argument("--model", default="yolo11n.pt", help="Base weights to fine-tune")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        default="",
        help="'' lets ultralytics choose, 'cpu' forces CPU, '0' is the first GPU",
    )
    parser.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ERROR: the 'ultralytics' package is not installed in this "
            f"interpreter ({sys.executable}).\n"
            "Install it with:  pip install ultralytics",
            file=sys.stderr,
        )
        return 2

    overrides = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "project": args.output,
        "name": args.name,
        "exist_ok": True,
    }
    if args.device:  # "" means "let ultralytics decide"; passing it would not
        overrides["device"] = args.device
    if args.batch and args.batch > 0:  # -1 is ultralytics' own auto-batch
        overrides["batch"] = args.batch

    print(f"Base weights : {args.model}")
    print(f"Dataset yaml : {args.data}")
    print(f"Output       : {args.output}/{args.name}")
    print(f"Epochs {args.epochs} | imgsz {args.imgsz} | device {args.device or 'auto'}")
    print("-" * 70, flush=True)

    model = YOLO(args.model)
    results = model.train(**overrides)

    # save_dir is where ultralytics actually wrote, which is not always
    # output/name — it appends a suffix when a run of that name already exists
    # and exist_ok was not honoured by an older version.
    save_dir = Path(getattr(results, "save_dir", Path(args.output) / args.name))
    best = save_dir / "weights" / "best.pt"
    print("-" * 70)
    if best.exists():
        print(f"{BEST_WEIGHTS_PREFIX}{best.resolve()}", flush=True)
    else:
        print(f"WARNING: no best.pt under {save_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

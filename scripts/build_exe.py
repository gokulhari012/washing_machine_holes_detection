"""Build the distributable Windows folder for the hole-detection station.

    python scripts/build_exe.py            # clean build into dist/WMHoleDetection
    python scripts/build_exe.py --no-clean # incremental (reuses build/ cache)

Runs PyInstaller against wmhd.spec, then stages the parts that must stay
*outside* the bundle: config/ (operators edit it, the app rewrites it) and the
empty runtime directories the app fills at run time. Ship the whole
dist/WMHoleDetection folder — the .exe does not work on its own.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SPEC = BASE_DIR / "wmhd.spec"
DIST_DIR = BASE_DIR / "dist" / "WMHoleDetection"
RUNTIME_DIRS = ("data", "logs", "images", "backups", "exports")


def run_pyinstaller(clean: bool) -> None:
    cmd = [sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm"]
    if clean:
        cmd.append("--clean")
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=BASE_DIR, check=True)


def stage_config() -> None:
    """Copy config/ next to the .exe — it is read AND written at run time."""
    target = DIST_DIR / "config"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(BASE_DIR / "config", target)
    print(f"staged {target.relative_to(BASE_DIR)}")


def stage_runtime_dirs() -> None:
    for name in RUNTIME_DIRS:
        (DIST_DIR / name).mkdir(parents=True, exist_ok=True)
    print(f"created runtime dirs: {', '.join(RUNTIME_DIRS)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-clean", action="store_true", help="reuse the build/ cache")
    args = parser.parse_args()

    try:
        run_pyinstaller(clean=not args.no_clean)
    except FileNotFoundError:
        print("PyInstaller is not installed — run: pip install pyinstaller", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"PyInstaller failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    stage_config()
    stage_runtime_dirs()
    print(f"\nBuild complete: {DIST_DIR}")
    print("Smoke test it with:")
    print(f'  "{DIST_DIR / "WMHoleDetection.exe"}" --selftest 8')
    return 0


if __name__ == "__main__":
    sys.exit(main())

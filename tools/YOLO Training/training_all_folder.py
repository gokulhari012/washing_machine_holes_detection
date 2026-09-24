"""
train_all_cameras.py

Usage:
    python train_all_cameras.py [--epochs EPOCHS] [--model MODEL] [--dataset DATASET] [--output OUTPUT]

What it does:
 - iterates camera_id = 1..6 (Camera_1 .. Camera_6)
 - creates a camera-specific data YAML file (data_camera_1.yaml etc.)
 - runs: yolo detect train data="path_to_yaml" model=... epochs=... imgsz=...
 - stores results in project/<camera_project> with separate name per camera
"""

import subprocess
from pathlib import Path
import textwrap
import argparse
import os
from datetime import datetime

# folder_name = "1"  # Default folder name
project_base_path = os.getcwd()

def parse_arguments():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train YOLO models for all cameras")
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100, 
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolo11m.pt", 
        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
        help="YOLO model to use (default: yolo11m.pt)"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=project_base_path + r"\brush_knowledge\dataset\normal_camera_images", 
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--folder_name", 
        type=str, 
        default="1", 
        help="Name of the folder containing the dataset (default: '1')"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Output directory for training results. If None, creates timestamped folder."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="0", 
        help="CUDA device ID or 'cpu' (default: 0)"
    )
    
    return parser.parse_args()

# ---------- USER CONFIG ----------
# Base folder that contains Camera_1 .. Camera_6 directories
BASE_IMAGES_DIR = Path(project_base_path + r"\brush_knowledge\dataset\normal_camera_images")
# Where to store training outputs (each camera gets its own name)
PROJECT_DIR = Path(project_base_path + r"\brush_knowledge\yolo_runs")
# PROJECT_DIR = "yolo_runs"  # will create ./yolo_runs/Camera_1 etc

# For each camera i the image folder used in your example was Camera_3\3
# If your structure differs, change the format string below accordingly.
# Example result: D:\...\normal_camera_images\Camera_3\3
CAMERA_SUBPATH_FMT = "Camera_{cam}/{folder_name}"  # {cam} replaced by 1..6

# YOLO training parameters (will be overridden by CLI args)
YOLO_MODEL = "yolo11m.pt"
EPOCHS = 100
IMGSZ = 640
DEVICE = "0"  # change to "cpu" or "0" or "0,1" as needed
# DEVICE = ""  # change to "cpu" or "0" or "0,1" as needed

# Where to write generated YAMLs
OUT_YAML_DIR = Path.cwd() / "generated_data_yaml"
OUT_YAML_DIR.mkdir(exist_ok=True)


# Whether to stop on first failure or continue to next camera
STOP_ON_ERROR = False
# ----------------------------------

NC = 2
NAMES = ["cb", "defect"]

def make_data_yaml(path_yaml: Path, train_path: str, val_path: str, nc: int, names: list):
    """Write a small data.yaml file compatible with ultralytics/yolo CLI"""
    content = textwrap.dedent(f"""\
        train: {train_path}
        val: {val_path}

        nc: {nc}
        names: {names}
    """)
    path_yaml.write_text(content, encoding="utf-8")
    return path_yaml

def run_yolo_train(data_yaml_path: Path, model: str, epochs: int, imgsz: int, device: str, project: str, name: str):
    """
    Run the yolo CLI. Adjust command string quoting to match how yolo CLI expects the data param.
    Using shell=True so we can include data="..." exactly like your manual command.
    """
    cmd = f'yolo detect train data="{str(data_yaml_path)}" model={model} epochs={epochs} imgsz={imgsz} device={device} project="{project}" name="{name}"'
    print(f"\nRunning for {name}:\n  {cmd}\n")
    # run and stream output (stdout/stderr will appear in console)
    result = subprocess.run(cmd, shell=True)
    return result.returncode

def main():
    # global folder_name  # to allow override from CLI if needed
    """Main training function with CLI argument support."""
    args = parse_arguments()
    
    # Override global config with CLI arguments
    base_images_dir = Path(args.dataset)
    yolo_model = args.model
    epochs = args.epochs
    device = args.device
    folder_name = args.folder_name
    # Create timestamped output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(project_base_path + r"\brush_knowledge\yolo_runs") / timestamp
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training output directory: {output_dir}")
    print(f"Epochs: {epochs}, Model: {yolo_model}, Device: {device}")
    
    # Copy models to weights folder after training
    weights_dir = Path(project_base_path + r"\brush_knowledge\weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    cameras = list(range(1, 7))  # 1..6
    for cam in cameras:
        cam_str = str(cam)
        # Build train/val paths using the format string
        rel = CAMERA_SUBPATH_FMT.format(cam=cam_str, folder_name=folder_name)
        train_path = base_images_dir / Path(rel)
        val_path = train_path  # same as your example
        if not train_path.exists():
            print(f"WARNING: train path for Camera_{cam} does not exist: {train_path}")
            # continue or create? we skip this camera
            if STOP_ON_ERROR:
                print("Stopping due to missing path.")
                return
            else:
                print("Skipping this camera and continuing.")
                continue

        # create a yaml file for this camera
        yaml_fname = OUT_YAML_DIR / f"data_camera_{cam}.yaml"
        make_data_yaml(yaml_fname, str(train_path), str(val_path), NC, NAMES)
        print(f"Generated YAML: {yaml_fname}")

        # run training; store outputs under project/project_name (separate per camera)
        project = str(output_dir)
        name = f"Camera_{cam}"
        rc = run_yolo_train(yaml_fname, yolo_model, epochs, IMGSZ, device, project, name)

        if rc != 0:
            print(f"Training for Camera_{cam} returned non-zero exit code: {rc}")
            if STOP_ON_ERROR:
                print("Stopping due to error.")
                return
            else:
                print("Continuing to next camera...")

    print("\nAll done.")
    print(f"\nTraining results saved to: {output_dir}")
    return str(output_dir)  # Return the output path

if __name__ == "__main__":
    main()

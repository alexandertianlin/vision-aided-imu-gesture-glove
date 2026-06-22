#!/bin/bash
# Task 1 (GPU 1: 10.128.0.8) ? YOLO-Hand-Pose Setup
set -e
echo "[YOLO-Hand-Pose] Starting setup..."
sudo apt-get update -y && sudo apt-get install -y python3-pip python3-venv git

WORKSPACE=~/yolo_benchmark
mkdir -p "$WORKSPACE" && cd "$WORKSPACE"
python3 -m venv venv && source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics numpy opencv-python tqdm

mkdir -p models data
echo "[YOLO-Hand-Pose] Setup complete!"
echo "  Activate: source $WORKSPACE/venv/bin/activate"

#!/bin/bash
# Task 2 (GPU 2: 10.128.0.9) ? MMPose-Hand Setup
set -e
echo "[MMPose-Hand] Starting setup..."
sudo apt-get update -y && sudo apt-get install -y python3-pip python3-venv git

WORKSPACE=~/mmpose_benchmark
mkdir -p "$WORKSPACE" && cd "$WORKSPACE"
python3 -m venv venv && source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install openmim numpy opencv-python tqdm
mim install mmengine mmcv>=2.0.0 mmdet>=3.0.0 mmpose>=1.0.0

mkdir -p models
cd models
mim download mmpose --config hrnet_w18_coco_wholebody_hand_256x256 --dest .
cd ..
mkdir -p data logs
echo "[MMPose-Hand] Setup complete!"
echo "  Activate: source $WORKSPACE/venv/bin/activate"

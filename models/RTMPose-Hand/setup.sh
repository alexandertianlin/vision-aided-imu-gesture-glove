#!/bin/bash
# ==============================================================
# Subagent 2: RTMPose-Hand Setup Script
# Target: GCP L4 GPU instance (10.128.0.9)
# Purpose: Set up RTMPose-Hand environment for hand pose benchmark
# ==============================================================

set -e

echo "[RTMPose-Hand] Starting setup..."

# System dependencies
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git

# Create workspace
WORKSPACE=~/rtmpose_benchmark
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1 (L4 compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install mmpose ecosystem (OpenMMLab)
pip install openmim
mim install mmengine
mim install mmcv>=2.0.0
mim install mmdet>=3.0.0
mim install mmpose>=1.0.0

# Install other deps
pip install numpy opencv-python tqdm

# Create model directory for pretrained weights
mkdir -p models

# Download RTMPose hand model
echo "[RTMPose-Hand] Downloading pretrained model..."
cd models
mim download mmpose --config rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256 --dest .
echo "[RTMPose-Hand] Model downloaded"

# Create data and log directories
cd "$WORKSPACE"
mkdir -p "$WORKSPACE/data" "$WORKSPACE/logs"

echo ""
echo "[RTMPose-Hand] Setup complete!"
echo "  Activate env: source $WORKSPACE/venv/bin/activate"
echo "  Next: copy benchmark test data and run benchmark.py"

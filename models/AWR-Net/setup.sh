#!/bin/bash
# ==============================================================
# Subagent 1: AWR-Net Setup Script
# Target: GCP L4 GPU instance (10.128.0.8)
# Purpose: Set up AWR-Net environment for hand pose benchmark
# ==============================================================

set -e

echo "[AWR-Net] Starting setup..."

# System dependencies
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git cmake build-essential

# Create workspace
WORKSPACE=~/awrnet_benchmark
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Clone AWR-Net (official implementation)
if [ ! -d "AWR-Net" ]; then
    git clone https://github.com/zhangboshen/AWR-Net.git
fi
cd AWR-Net

# Install PyTorch with CUDA 12.1 (compatible with L4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install numpy opencv-python matplotlib scipy tqdm tensorboard

# Create benchmark directories
mkdir -p data logs

# Download pretrained model (ICCV 2019 release)
PRETRAINED_DIR=./models
mkdir -p "$PRETRAINED_DIR"
echo "[AWR-Net] Please manually download pretrained model to $WORKSPACE/AWR-Net/$PRETRAINED_DIR"
echo "  From: https://github.com/zhangboshen/AWR-Net (check releases)"
echo "  Or:  download from Google Drive link in the official repo README"

echo "[AWR-Net] Setup complete!"
echo "  Activate env: source $WORKSPACE/venv/bin/activate"
echo "  Next: copy benchmark test data and run benchmark.py"

#!/bin/bash
# ==============================================================
# Subagent 3: IPNet Setup Script
# Target: GCP L4 GPU instance (10.128.0.11)
# Purpose: Set up IPNet environment for hand pose benchmark
# ==============================================================

set -e

echo "[IPNet] Starting setup..."

# System dependencies
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git cmake build-essential

# Create workspace
WORKSPACE=~/ipnet_benchmark
mkdir -p $WORKSPACE
cd $WORKSPACE

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1 (L4 compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install numpy opencv-python matplotlib scipy tqdm open3d

# Clone IPNet
if [ ! -d "IPNet" ]; then
    git clone https://github.com/shubham-goel/IPNet.git
fi
cd IPNet

# Install IPNet dependencies
pip install -r requirements.txt 2>/dev/null || echo "No requirements.txt found"

# Create directories
mkdir -p data logs models

# Download pretrained model
echo "[IPNet] Please download pretrained model:"
echo "  From: https://github.com/shubham-goel/IPNet/releases"
echo "  Place in: /IPNet/models/"
echo ""
echo "  Or use cached model if available"

echo ""
echo "[IPNet] Setup complete!"
echo "  Activate env: source $WORKSPACE/venv/bin/activate"
echo "  Next: copy benchmark test data and run benchmark.py"

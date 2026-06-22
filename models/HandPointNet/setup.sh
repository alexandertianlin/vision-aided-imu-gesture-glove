#!/bin/bash
# Task 3 (GPU 3: 10.128.0.11) ? HandPointNet Setup
set -e
echo "[HandPointNet] Starting setup..."
echo "WARNING: HandPointNet may require Python 2 / PyTorch 0.3 / CUDA 8"
echo "Attempting modern env first..."

sudo apt-get update -y && sudo apt-get install -y python3-pip python3-venv git cmake build-essential

WORKSPACE=~/handpointnet_benchmark
mkdir -p $WORKSPACE && cd $WORKSPACE
python3 -m venv venv && source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy opencv-python tqdm open3d

if [ ! -d "HandPointNet" ]; then
    git clone https://github.com/xinghaochen/HandPointNet.git || true
fi

mkdir -p models data logs
echo ""
echo "[HandPointNet] Setup complete!"
echo "  Activate: source $WORKSPACE/venv/bin/activate"
echo "  If modern env fails, try: conda create -n hpn python=2.7 pytorch=0.3 cuda=8"

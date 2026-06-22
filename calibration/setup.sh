#!/bin/bash
# Task 4: Extrinsic Calibration Setup
set -e
echo "[Calibration] Starting setup..."
sudo apt-get update -y && sudo apt-get install -y python3-pip python3-venv git cmake build-essential

WORKSPACE=~/calibration
mkdir -p  && cd 
python3 -m venv venv && source venv/bin/activate

pip install numpy opencv-python matplotlib open3d scipy
mkdir -p data logs
echo "[Calibration] Setup complete!"
echo "  Activate: source /venv/bin/activate"

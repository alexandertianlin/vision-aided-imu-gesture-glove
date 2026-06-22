# Subagent 1: AWR-Net Benchmark Preparation

## Target Instance
- IP: 10.128.0.8
- GPU: L4 (us-central1-a)

## Model Info
- **AWR-Net** (Adaptive Weight Regression Network)
- Input: Depth map (256x256, single channel)
- Output: 21 keypoints in 3D (63 values)
- Paper: "Adaptive Weight Regression for 3D Hand Pose Estimation" (ICCV 2019)

## Quick Start

### 1. SSH into the instance
`
gcloud compute ssh instance-20260612-20260612-20260612-20260612-123731 --zone=us-central1-a
`

### 2. Run setup
`
chmod +x setup.sh
./setup.sh
`

### 3. Run benchmark
`
source ~/awrnet_benchmark/venv/bin/activate
python benchmark.py --test-dir ../../data/rgb_depth_sequences
`

### 4. View results
`
cat results/logs/AWR-Net_benchmark.log
`

## Files
- setup.sh ? environment setup (PyTorch, OpenCV, git clone)
- enchmark.py ? benchmark runner with metrics logging
- config.yaml ? model config

## Pretrained Weights
Download from: https://github.com/zhangboshen/AWR-Net
Place in: models/AWR-Net/pretrained/awrnet_hand.pth

# Subagent 3: IPNet Benchmark Preparation

## Target Instance
- IP: 10.128.0.11
- GPU: L4 (us-central1-a)

## Model Info
- **IPNet** (Implicit Point Network)
- Input: Depth map / Point Cloud (256x256)
- Output: 21 keypoints in 3D
- Paper: "IPNet: Implicit Point Network for 3D Hand Pose and Shape Estimation from Depth" (3DV 2020)

## Quick Start

### 1. SSH into the instance
`
gcloud compute ssh instance-20260612-20260612-20260612-20260612-20260616-015628 --zone=us-central1-a
`

### 2. Run setup
`
chmod +x setup.sh
./setup.sh
`

### 3. Run benchmark
`
source ~/ipnet_benchmark/venv/bin/activate
python benchmark.py --test-dir ../../data/rgb_depth_sequences
`

### 4. View results
`
cat results/logs/IPNet_benchmark.log
`

## Files
- setup.sh ? PyTorch + IPNet setup
- enchmark.py ? benchmark runner
- config.yaml ? model config

## Pretrained Weights
Download from: https://github.com/shubham-goel/IPNet/releases
Place in: models/IPNet/pretrained/ipnet_hand.pth

# Subagent 2: RTMPose-Hand Benchmark Preparation

## Target Instance
- IP: 10.128.0.9
- GPU: L4 (us-central1-a)

## Model Info
- **RTMPose-Hand** (part of MMPose, OpenMMLab)
- Input: RGB image (256x256, 3 channels)
- Output: 21 keypoints in 2D
- Paper: "RTMPose: Real-Time Multi-Person Pose Estimation" (2023)

## Quick Start

### 1. SSH into the instance
`
gcloud compute ssh instance-20260612-20260612-20260612-20260612-20260616-014510 --zone=us-central1-a
`

### 2. Run setup
`
chmod +x setup.sh
./setup.sh
`

### 3. Run benchmark
`
source ~/rtmpose_benchmark/venv/bin/activate
python benchmark.py --test-dir ../../data/rgb_depth_sequences
`

### 4. View results
`
cat results/logs/RTMPose-Hand_benchmark.log
`

## Files
- setup.sh ? MMPose + PyTorch + OpenMMLab setup
- enchmark.py ? benchmark runner
- config.yaml ? model config

## Notes
- RTMPose-Hand output is 2D only (no 3D keypoints)
- Model weights auto-downloaded by mim
- If mmpose import fails, check CUDA version compatibility

#!/bin/bash
# ==============================================================
# Task 1 Runner — AWR-Net + YOLO-Hand-Pose
# Target: GCP L4 GPU instance (10.128.0.8)
# Usage:  SCP to instance, chmod +x, then ./task1_runner.sh
# ==============================================================
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=~/task1_results_$TIMESTAMP
mkdir -p "$RESULTS_DIR"/logs

echo "=========================================="
echo "Task 1: GPU Benchmark — AWR-Net + YOLO-Hand-Pose"
echo "Started: $(date)"
echo "Results: $RESULTS_DIR"
echo "=========================================="
echo ""

# --------------------------------------------------
# Phase 1: AWR-Net
# --------------------------------------------------
echo "=== [Phase 1/2] AWR-Net (Depth) ==="

if [ ! -d ~/awrnet_benchmark/venv ]; then
    echo "[ERROR] AWR-Net environment not found. Run models/AWR-Net/setup.sh first."
    exit 1
fi

source ~/awrnet_benchmark/venv/bin/activate
cd ~/awrnet_benchmark/AWR-Net

# Link test data if available
if [ -d ../../../data/rgb_depth_sequences ]; then
    ln -sfn "$(pwd)/../../../data/rgb_depth_sequences" ./data/rgb_depth_sequences
fi

echo "[AWR-Net] Running benchmark..."
python benchmark.py \
    --test-dir ./data/rgb_depth_sequences \
    --output "$RESULTS_DIR/awrnet"
deactivate

echo "[AWR-Net] Done."
echo ""

# --------------------------------------------------
# Phase 2: YOLO-Hand-Pose
# --------------------------------------------------
echo "=== [Phase 2/2] YOLO-Hand-Pose (RGB) ==="

if [ ! -d ~/yolo_benchmark/venv ]; then
    echo "[ERROR] YOLO-Hand-Pose environment not found. Run models/YOLO-Hand-Pose/setup.sh first."
    exit 1
fi

source ~/yolo_benchmark/venv/bin/activate
cd ~/yolo_benchmark

# Link test data if available
if [ -d ../../data/rgb_depth_sequences ]; then
    ln -sfn "$(pwd)/../../data/rgb_depth_sequences" ./data/rgb_depth_sequences
fi

echo "[YOLO-Hand-Pose] Running benchmark..."
python benchmark.py \
    --test-dir ./data/rgb_depth_sequences \
    --output "$RESULTS_DIR/yolo"
deactivate

echo "[YOLO-Hand-Pose] Done."
echo ""

# --------------------------------------------------
# Summary
# --------------------------------------------------
echo "=========================================="
echo " Task 1 Complete — $(date)"
echo "=========================================="

echo ""
echo "Results directory: $RESULTS_DIR"
echo ""
echo "--- AWR-Net ---"
if [ -f "$RESULTS_DIR/awrnet/logs/AWR-Net_benchmark.log" ]; then
    cat "$RESULTS_DIR/awrnet/logs/AWR-Net_benchmark.log"
else
    echo "(log not found)"
fi

echo ""
echo "--- YOLO-Hand-Pose ---"
if [ -f "$RESULTS_DIR/yolo/logs/YOLO-Hand-Pose_benchmark.log" ]; then
    cat "$RESULTS_DIR/yolo/logs/YOLO-Hand-Pose_benchmark.log"
else
    echo "(log not found)"
fi

echo ""
echo "Summary written to: $RESULTS_DIR"

# Write combined summary CSV
python3 -c "
import csv, os, glob

all_rows = []
for logfile in glob.glob('$RESULTS_DIR/**/*.log', recursive=True):
    row = {}
    with open(logfile) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                k, v = line.split(':', 1)
                row[k.strip()] = v.strip()
    if row:
        all_rows.append(row)

csv_path = '$RESULTS_DIR/summary.csv'
if all_rows:
    fields = ['model','detect_rate','fps','latency_ms','stability',
              'keypoints_21','keypoints_3d','deployment_time_min','status','failure_reason']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f'Combined CSV: {csv_path}')
else:
    print('No benchmark results found to aggregate.')
"

echo ""
echo "=== task1_runner.sh finished ==="

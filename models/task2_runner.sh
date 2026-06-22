#!/bin/bash
# ==============================================================
# Task 2: GPU Benchmark Runner
# Target: GCP L4 GPU instance (10.128.0.9)
# Models: RTMPose-Hand -> MMPose-Hand (sequential)
# ==============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT="$SCRIPT_DIR/../results"

echo "================================================================="
echo "  Task 2: GPU Benchmark — RTMPose-Hand then MMPose-Hand"
echo "  Instance: 10.128.0.9 (L4, us-central1-a)"
echo "  Timestamp: $TIMESTAMP"
echo "================================================================="
echo ""

# -------------------------------------------------------
# Phase 1: RTMPose-Hand Setup
# -------------------------------------------------------
echo ""
echo "============================================"
echo "  Phase 1: RTMPose-Hand Setup"
echo "============================================"
cd "$SCRIPT_DIR/RTMPose-Hand"
chmod +x setup.sh
./setup.sh
echo "  [OK] RTMPose-Hand setup complete"

# -------------------------------------------------------
# Phase 2: RTMPose-Hand Benchmark
# -------------------------------------------------------
echo ""
echo "============================================"
echo "  Phase 2: RTMPose-Hand Benchmark"
echo "============================================"
source ~/rtmpose_benchmark/venv/bin/activate
python "$SCRIPT_DIR/RTMPose-Hand/benchmark.py" \
    --test-dir "$SCRIPT_DIR/../data/rgb_depth_sequences" \
    --output "$BASE_OUTPUT/RTMPose-Hand"
deactivate
echo "  [OK] RTMPose-Hand benchmark complete"

# -------------------------------------------------------
# Phase 3: MMPose-Hand Setup
# -------------------------------------------------------
echo ""
echo "============================================"
echo "  Phase 3: MMPose-Hand Setup"
echo "============================================"
cd "$SCRIPT_DIR/MMPose-Hand"
chmod +x setup.sh
./setup.sh
echo "  [OK] MMPose-Hand setup complete"

# -------------------------------------------------------
# Phase 4: MMPose-Hand Benchmark
# -------------------------------------------------------
echo ""
echo "============================================"
echo "  Phase 4: MMPose-Hand Benchmark"
echo "============================================"
source ~/mmpose_benchmark/venv/bin/activate
python "$SCRIPT_DIR/MMPose-Hand/benchmark.py" \
    --test-dir "$SCRIPT_DIR/../data/rgb_depth_sequences" \
    --output "$BASE_OUTPUT/MMPose-Hand"
deactivate
echo "  [OK] MMPose-Hand benchmark complete"

# -------------------------------------------------------
# Phase 5: Summary
# -------------------------------------------------------
echo ""
echo "============================================"
echo "  Phase 5: Results Summary"
echo "============================================"
echo ""
echo "--- RTMPose-Hand Results ---"
if [ -f "$BASE_OUTPUT/RTMPose-Hand/logs/RTMPose-Hand_benchmark.log" ]; then
    cat "$BASE_OUTPUT/RTMPose-Hand/logs/RTMPose-Hand_benchmark.log"
else
    echo "  (log not found)"
fi
echo ""
echo "--- MMPose-Hand Results ---"
if [ -f "$BASE_OUTPUT/MMPose-Hand/logs/MMPose-Hand_benchmark.log" ]; then
    cat "$BASE_OUTPUT/MMPose-Hand/logs/MMPose-Hand_benchmark.log"
else
    echo "  (log not found)"
fi

echo ""
echo "================================================================="
echo "  Task 2 Complete"
echo "  Logs and results saved to $BASE_OUTPUT/"
echo "================================================================="
echo ""

# Also log a machine-readable summary
echo "$TIMESTAMP,RTMPose-Hand,\$(grep 'detect_rate:' \"\$BASE_OUTPUT/RTMPose-Hand/logs/RTMPose-Hand_benchmark.log\" 2>/dev/null || echo 'N/A')" > "$BASE_OUTPUT/task2_summary.csv"
echo "$TIMESTAMP,MMPose-Hand,\$(grep 'detect_rate:' \"\$BASE_OUTPUT/MMPose-Hand/logs/MMPose-Hand_benchmark.log\" 2>/dev/null || echo 'N/A')" >> "$BASE_OUTPUT/task2_summary.csv"
echo "  Summary written to $BASE_OUTPUT/task2_summary.csv"
echo ""

#!/bin/bash
# ==============================================================
# Task 3 Runner: IPNet -> HandPointNet (in sequence)
# Target:  SCP this to GCP GPU instance (10.128.0.11) and run
# Usage:   bash task3_runner.sh [--test-dir PATH] [--output-dir PATH]
#
# This script:
#   1. Sets up Python venvs and deps for both models
#   2. Runs IPNet benchmark (depth -> 21 keypoints 3D)
#   3. Runs HandPointNet benchmark (point cloud -> 21 keypoints 3D)
#      with fallback if modern env fails
#   4. Collects results into a summary
# ==============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="${1:-data/rgb_depth_sequences}"
OUTPUT_DIR="${2:-results}"
MODELS_DIR="$PROJECT_DIR/models"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="$OUTPUT_DIR/task3_summary_${TIMESTAMP}.log"

mkdir -p "$OUTPUT_DIR/logs"

echo "============================================" | tee "$SUMMARY_LOG"
echo " Task 3 Runner: IPNet + HandPointNet" | tee -a "$SUMMARY_LOG"
echo " Started: $(date)" | tee -a "$SUMMARY_LOG"
echo " Project: $PROJECT_DIR" | tee -a "$SUMMARY_LOG"
echo " Test:    $TEST_DIR" | tee -a "$SUMMARY_LOG"
echo "============================================" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

# -----------------------------------------------------------
# PHASE 1: IPNet
# -----------------------------------------------------------
echo "" | tee -a "$SUMMARY_LOG"
echo "----- Phase 1: IPNet Setup -----" | tee -a "$SUMMARY_LOG"

cd "$PROJECT_DIR"
if [ -f "$MODELS_DIR/IPNet/setup.sh" ]; then
    bash "$MODELS_DIR/IPNet/setup.sh" 2>&1 | tee -a "$SUMMARY_LOG"
    echo "[OK] IPNet setup completed" | tee -a "$SUMMARY_LOG"
else
    echo "[WARN] IPNet setup.sh not found" | tee -a "$SUMMARY_LOG"
fi

# Link cloned IPNet repo so benchmark.py can find it
if [ -d ~/ipnet_benchmark/IPNet ] && [ ! -d "$MODELS_DIR/IPNet/IPNet" ]; then
    ln -sf ~/ipnet_benchmark/IPNet "$MODELS_DIR/IPNet/IPNet"
    echo "[OK] Symlinked IPNet repo" | tee -a "$SUMMARY_LOG"
fi

echo "" | tee -a "$SUMMARY_LOG"
echo "----- Phase 1: IPNet Benchmark -----" | tee -a "$SUMMARY_LOG"

if [ -f ~/ipnet_benchmark/venv/bin/activate ]; then
    source ~/ipnet_benchmark/venv/bin/activate
    cd "$MODELS_DIR/IPNet"
    python benchmark.py --test-dir "$PROJECT_DIR/$TEST_DIR" --output "$PROJECT_DIR/$OUTPUT_DIR" 2>&1 | tee -a "$SUMMARY_LOG"
    deactivate
    echo "[OK] IPNet benchmark completed" | tee -a "$SUMMARY_LOG"
else
    echo "[WARN] IPNet venv not found, attempting direct run" | tee -a "$SUMMARY_LOG"
    cd "$MODELS_DIR/IPNet"
    python3 benchmark.py --test-dir "$PROJECT_DIR/$TEST_DIR" --output "$PROJECT_DIR/$OUTPUT_DIR" 2>&1 | tee -a "$SUMMARY_LOG"
fi

# -----------------------------------------------------------
# PHASE 2: HandPointNet (with fallback)
# -----------------------------------------------------------
echo "" | tee -a "$SUMMARY_LOG"
echo "----- Phase 2: HandPointNet Setup (Modern Env) -----" | tee -a "$SUMMARY_LOG"

cd "$PROJECT_DIR"
if [ -f "$MODELS_DIR/HandPointNet/setup.sh" ]; then
    bash "$MODELS_DIR/HandPointNet/setup.sh" 2>&1 | tee -a "$SUMMARY_LOG"
    echo "[OK] HandPointNet modern setup completed" | tee -a "$SUMMARY_LOG"
else
    echo "[WARN] HandPointNet setup.sh not found" | tee -a "$SUMMARY_LOG"
fi

# Link cloned HandPointNet repo so benchmark.py can find it
if [ -d ~/handpointnet_benchmark/HandPointNet ] && [ ! -d "$MODELS_DIR/HandPointNet/HandPointNet" ]; then
    ln -sf ~/handpointnet_benchmark/HandPointNet "$MODELS_DIR/HandPointNet/HandPointNet"
    echo "[OK] Symlinked HandPointNet repo" | tee -a "$SUMMARY_LOG"
fi

echo "" | tee -a "$SUMMARY_LOG"
echo "----- Phase 2: HandPointNet Benchmark -----" | tee -a "$SUMMARY_LOG"

HPN_RESULT=""
if [ -f ~/handpointnet_benchmark/venv/bin/activate ]; then
    source ~/handpointnet_benchmark/venv/bin/activate
    cd "$MODELS_DIR/HandPointNet"
    python benchmark.py --test-dir "$PROJECT_DIR/$TEST_DIR" --output "$PROJECT_DIR/$OUTPUT_DIR" 2>&1 | tee -a "$SUMMARY_LOG"
    HPN_EXIT=$?
    deactivate
    if [ $HPN_EXIT -eq 0 ]; then
        echo "[OK] HandPointNet benchmark completed (modern env)" | tee -a "$SUMMARY_LOG"
        HPN_RESULT="modern"
    fi
fi

# FALLBACK: Docker-based HandPointNet with legacy Python 2 / PyTorch 0.3
if [ -z "$HPN_RESULT" ]; then
    echo "" | tee -a "$SUMMARY_LOG"
    echo "----- Phase 2 Fallback: HandPointNet (Legacy Env) -----" | tee -a "$SUMMARY_LOG"

    if command -v docker &>/dev/null && docker info &>/dev/null; then
        echo "[FALLBACK] Docker available. Trying legacy env..." | tee -a "$SUMMARY_LOG"
        if [ -f "$MODELS_DIR/HandPointNet/docker_fallback.sh" ]; then
            bash "$MODELS_DIR/HandPointNet/docker_fallback.sh" 2>&1 | tee -a "$SUMMARY_LOG"
        else
            echo "[FALLBACK] No docker_fallback.sh. Using mock model." | tee -a "$SUMMARY_LOG"
            source ~/handpointnet_benchmark/venv/bin/activate 2>/dev/null || true
            cd "$MODELS_DIR/HandPointNet"
            python3 benchmark.py --test-dir "$PROJECT_DIR/$TEST_DIR" --output "$PROJECT_DIR/$OUTPUT_DIR" 2>&1 | tee -a "$SUMMARY_LOG"
        fi
    else
        echo "[FALLBACK] Docker unavailable. Using mock model fallback." | tee -a "$SUMMARY_LOG"
        source ~/handpointnet_benchmark/venv/bin/activate 2>/dev/null || true
        cd "$MODELS_DIR/HandPointNet"
        python3 benchmark.py --test-dir "$PROJECT_DIR/$TEST_DIR" --output "$PROJECT_DIR/$OUTPUT_DIR" 2>&1 | tee -a "$SUMMARY_LOG"
    fi
fi

# -----------------------------------------------------------
# PHASE 3: Collect Results
# -----------------------------------------------------------
echo "" | tee -a "$SUMMARY_LOG"
echo "============================================" | tee -a "$SUMMARY_LOG"
echo " Task 3 Summary" | tee -a "$SUMMARY_LOG"
echo "============================================" | tee -a "$SUMMARY_LOG"

echo "" | tee -a "$SUMMARY_LOG"
echo "--- IPNet Log ---" | tee -a "$SUMMARY_LOG"
if [ -f "$OUTPUT_DIR/logs/IPNet_benchmark.log" ]; then
    cat "$OUTPUT_DIR/logs/IPNet_benchmark.log" | tee -a "$SUMMARY_LOG"
fi

echo "" | tee -a "$SUMMARY_LOG"
echo "--- HandPointNet Log ---" | tee -a "$SUMMARY_LOG"
if [ -f "$OUTPUT_DIR/logs/HandPointNet_benchmark.log" ]; then
    cat "$OUTPUT_DIR/logs/HandPointNet_benchmark.log" | tee -a "$SUMMARY_LOG"
fi

echo "" | tee -a "$SUMMARY_LOG"
echo "--- CSV Results ---" | tee -a "$SUMMARY_LOG"
if [ -f "$OUTPUT_DIR/benchmark_results.csv" ]; then
    cat "$OUTPUT_DIR/benchmark_results.csv" | tee -a "$SUMMARY_LOG"
fi

echo "" | tee -a "$SUMMARY_LOG"
echo "============================================" | tee -a "$SUMMARY_LOG"
echo " Task 3 Complete: $(date)" | tee -a "$SUMMARY_LOG"
echo " Summary: $SUMMARY_LOG" | tee -a "$SUMMARY_LOG"
echo "============================================" | tee -a "$SUMMARY_LOG"

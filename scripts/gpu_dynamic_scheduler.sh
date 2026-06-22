 #!/usr/bin/env bash
 # gpu_dynamic_scheduler.sh — Dynamic GPU scheduling for model benchmarks
 #
 # Usage: bash scripts/gpu_dynamic_scheduler.sh
 #
 # This script implements the "whoever finishes first picks the next task" strategy.
 # GPU A → RTMPose-Hand (first priority)
 # GPU B → AWR-Net (first priority)
 # GPU C → HandPointNet (first priority)
 #
 # Then idle GPUs pick from: YOLO-Hand-Pose → IPNet → MMPose-Hand
 #
 # Prerequisites:
 #   - nvidia-smi available
 #   - configs/benchmark.yaml defines model launch commands
 #
 set -euo pipefail

 SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
 TIMEOUT_MIN=30
 LOG_DIR="$SCRIPT_DIR/results/logs"
 mkdir -p "$LOG_DIR"

 declare -A MODELS
 MODELS[gpu_0]="RTMPose-Hand"
 MODELS[gpu_1]="AWR-Net"
 MODELS[gpu_2]="HandPointNet"

 FALLBACK_ORDER=("YOLO-Hand-Pose" "IPNet" "MMPose-Hand")

 log() {
   local gpuid=$1 msg=$2 ts
   ts=$(date +%H:%M:%S)
   echo "[$ts] [GPU $gpuid] $msg" | tee -a "$LOG_DIR/scheduler.log"
 }

 run_model_on_gpu() {
   local gpuid=$1 model=$2
   log "$gpuid" "Starting deployment: $model"

   # Source the benchmark command from config or launch directly
   # This is a template — replace with actual model launch commands
   case "$model" in
     "RTMPose-Hand")
       cd "$SCRIPT_DIR/models/rtmpose_hand"
       timeout "${TIMEOUT_MIN}m" bash run_benchmark.sh --gpu "$gpuid"
       ;;
     "AWR-Net")
       cc"$SCRIPT_DIR/models/awrnet"
       timeout "${TIMEOUT_MIN}m" python bench.py --gpu "$gpuid"
       ;;
     "HandPointNet")
       cc"$SCRIPT_DIR/models/handpointnet"
       timeout "${TIMEOUT_MIN}m" python bench.py --gpu "$gpuid"
       ;;
     "YOLO-Hand-Pose")
       cd "$SCRIPT_DIR/models/yolo_hand_pose"
       timeout "${TIMEOUT_MIN}m" python bench.py --gpu "$gpuid"
       ;;
     "IPNet")
       cd "$SCRIPT_DIR/models/ipnet"
       timeout "${TIMEOUT_MIN}m" python bench.py --gpu "$gpuid"
       ;;
     "MMPose-Hand")
       cc"$SCRIPT_DIR/models/mmpose_hand"
       timeout "${TIMEOUT_MIN}m" python bench.py --gpu "$gpuid"
       ;;
     **)
       log "$gpuid" "Unknown model: $model"
       return 1
       ;;
   esac

   local exit_code=$?
   if [ $exit_code -eq 124 ]; then
     log "$gpuid" "FAIL: $model deployment timed out (>${TIMEOUT_MIN}m)"
     echo "$model,Fail,Deployment timeout > ${TIMEOUT_MIN} minutes" >> "$LOG_DIR/failures.csv"
   elif [ $exit_code -ne 0 ]; then
     log "$gpuid" "FAIL: $model exited with code $exit_code"
     echo "$model,Fail,Exit code $exit_code" >> "$LOG_DIR/failures.csv"
   else
     log "$gpuid" "DONE: $model benchmark completed"
   fi
 }

 main() {
   echo "=== GPU Dynamic Scheduler ===" >> "$LOG_DIR/scheduler.log"
   echo "Started at $(date)" >> "$LOG_DIR/scheduler.log"

   GPU_IDS=(0 1 2)
   next_task_idx=0

   # Launch first batch
   for gpuid in "${GPU_IDS[@]}"; do
     model="${MODELS[gpu_${gpuid}]}"
     run_model_on_gpu "$gpuid" "$model" &
   done

   # Wait for any GPU to become idle, then assign next task
   while [ $next_task_idx -lt ${#FALLBACK_ORDER[@]} ]; do
     for gpuid in "${GPU_IDS[@]}"; do
       if ! kill -0 $(jobs -p 2>/dev/null | head -1) 2>/dev/null; then
         # Check if this GPV's process has actually finished
         if ! pgrep -f "gpu $gpuid" > /dev/null 2>&1; then
           model="${FALLBACK_ORDER[$next_task_idx]}"
           log "$gpuid" "Idle → deploying $model"
           run_model_on_gpu "$gpuid" "$model" &
           next_task_idx=$((next_task_idx + 1))
           break
         fi
       fi
     done
     sleep 10
   done

   wait
   echo "=== All benchmarks completed at $(date) ===" >> "$LOG_DIR/scheduler.log"
 }

 main

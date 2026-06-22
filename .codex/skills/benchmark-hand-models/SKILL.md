 # Skill: Benchmark Hand Models

 ## Purpose
 Run a standardized benchmark comparing multiple hand pose estimation models on the same test
 sequence, using dynamic GPU scheduling. Records detect rate, FPS, latency, stability,
 and 21-keypoint output for each model.

 ## Trigger
 - User asks to benchmark models
 - User asks to compare hand tracking models
 - User wants to evaluate a specific model on the test dataset

## Output
 - results/benchmark_results.csv
 - results/model_comparison.md
 - results/logs/{model_name}_*.log

 ## Workflow

 ### 1. Prepare
 - Confirm test video sequence exists in data/rgb_depth_sequences/
 - Read configs/benchmark.yaml for model list and GPU scheduling
 - Check which GPUs are available (nvidia-smi)

 ### 2. Deploy
 - For each model, clone/setup in models/{model_name}/
 - If deployment exceeds timeout (configurable, default 30 min), mark Fail
 - Log deployment steps and errors

### 3. Run
 - Feed same test sequence to each model
 - Record:
   * Detection rate (frames with valid detection / total frames)
   * FPS (total frames / total inference time)
   * Latency (per-frame inference time, averaged)
   * Stability (variance of keypoint positions across consecutive frames)
   * 21 keypoint output (yes/no)
   * 3D keypoint support (yes/no)

 ### 4. Report
 - Aggregate all individual model logs
 - Generate benchmark_results.csv
 - Generate model_comparison.md with a ranked comparison table
 - Highlight top 1-2 candidates

 ### 5. Eliminate
 - If deployment > 30 min → Fail - Environment incompatible
 - If detect rate < 30% → Fail - Low detection
 - If FPS < 10 on high-end GPU ᆒ Fail - Too slow for real-time
 - Record all failure reasons

 ## Integration
 - Used by model-benchmark-agent (subagent)
 - Results feed into experiment-report-agent

## History
 - Created: 2026-06-16
 - Version: 1

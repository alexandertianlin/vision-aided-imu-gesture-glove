# Task Architecture ??2026-06-17

> Follows Codex architecture: 4-layer system, Skill-driven, Config-controlled, Dynamic scheduling, Elimination mechanism, Standardized reporting.
> Based on yesterday's project plan (2026-06-16) Codex architectural patterns.

---

## 1. Architecture Overview

### 4-Layer System Architecture (Codex Pattern)

`
                                   Evaluation Loop
                                  (dynamic GPU scheduling)
                                         |
   Camera Layer       Calibration Layer    Model Layer        Output Layer
+--------------------+------------------+-----------------+------------------+
| Orbbec Astra Plus  | Intrinsic (K)    | AWR-Net (Depth) | 21 Keypoints     |
| RGB + Depth ?2    | Extrinsic (R,T)  | IPNet (Depth)   | 3D Coords        |
| Point Cloud        | Depth???GB      | RTMPose (RGB)   | CSV Report       |
| Timestamp Sync     | ChArUco Board    | YOLO-Hand (RGB) | Model Comparison |
+--------------------+------------------+ MMPose (RGB)    +------------------+
                                          | HandPtNet (PC)  |
                                          +-----------------+
                                                  |
                                          Elimination Gate
                                         (30min timeout per model)
`

### Skill-Driven Execution (Codex Pattern)

Each phase is a Codex Skill with defined input/output:

| Phase | Skill | Input | Output |
|-------|-------|-------|--------|
| 1. Camera Pipeline | camera-verification | Hardware config | RGB+Depth+PC verified |
| 2. Dataset Recording | dataset-recording | Camera streams | data/rgb_depth_sequences/ |
| 3. Model Benchmark | enchmark-hand-models | Test dataset + configs/benchmark.yaml | results/benchmark_results.csv |
| 4. Report | enchmark-summary | Individual logs | model_comparison.md |

---

## 2. Phase 1: Camera Pipeline Verification

### Inputs
- configs/camera.yaml
- configs/calibration.yaml
- Orbbec Astra Plus hardware (RGB ?2 + Depth ?2)

### Workflow

`	ext
Phase 1 Workflow (parallel streams):
?? RGB Stream ??? verify live feed ??? print resolution/FPS
?? Depth Stream ?? verify live feed ??? check reflection artifacts
?? Point Cloud ?? export via Open3D ??? verify alignment
?? Synchronization ?? print timestamps ??? calculate diff (both cameras)
`

### Decision Branch
`
Depth Stable? ?Yes? Priority: AWR-Net > IPNet
      ?
      ?No??? Priority: RTMPose > YOLO-Hand > MMPose
`

### Deliverables
- [ ] camera/camera_test.log (logs each stream status)
- [ ] configs/extrinsic_params.json (R, T if dual camera calibrated)

---

## 3. Phase 2: Unified Test Dataset Recording

### Recording Scene Matrix

| Scene | Bare Hand | Normal Glove | Sensor Glove |
|-------|-----------|--------------|--------------|
| Open palm | ??| ??| ??|
| Fist | ??| ??| ??|
| Rotation | ??| ??| ??|
| Occlusion | ??| ??| ??|
| Near | ??| ??| ??|
| Far | ??| ??| ??|

### Directory Structure (Codex skill-driven pattern)
`
data/rgb_depth_sequences/
 ????? bare_hand/
 ??   ????? scene_{n}.bag / .mp4 (RGB+Depth+timestamps)
 ????? normal_glove/
 ??   ????? scene_{n}.bag / .mp4
 ????? sensor_glove/
      ????? scene_{n}.bag / .mp4
`

### Recording Config (controlled by configs)
- Same camera parameters across all scenes
- Same lighting conditions
- Timestamp sequence for sync verification

---

## 4. Phase 3: 6-Model Parallel Benchmark

### Dynamic GPU Scheduling (Codex dynamic pattern)

`
First Batch (simultaneous launch):
  GPU A ??RTMPose-Hand     (run-through rate: ????????/5)
  GPU B ??AWR-Net          (project need: ????????/5)
  GPU C ??HandPointNet     (env risk: highest, step on landmine first)

Dynamic Fallback (whoever finishes first picks next):
  Free Slot 1 ??YOLO-Hand-Pose  (run-through: ????????/5)
  Free Slot 2 ??IPNet            (project need: ????????/5)
  Free Slot 3 ??MMPose-Hand     (run-through: ????????/5)
`

### Model Deployment Config (from configs/benchmark.yaml)

| Model | Type | Expected Input | Timeout | Risk |
|-------|------|---------------|---------|------|
| AWR-Net | depth | depth_map | 30 min | Medium |
| IPNet | depth | depth_map | 30 min | Medium |
| RTMPose-Hand | rgb | rgb_image | 20 min | Low |
| YOLO-Hand-Pose | rgb | rgb_image | 20 min | Low |
| MMPose-Hand | rgb | rgb_image | 20 min | Low |
| HandPointNet | pointcloud | point_cloud | 30 min | High (Py2/PyTorch0.3/CUDA8) |

### Metrics Recording (Codex standardized reporting)

| Metric | Description | Format |
|--------|-------------|--------|
| Detect Rate | Frames with valid detection / total frames | float 0-1 |
| FPS | Total frames / total inference time | float |
| Latency | Per-frame inference time (ms) | float |
| Stability | Keypoint jitter across frames (1-5) | int |
| 21 Keypoints | Whether outputs standard 21 keypoints | yes/no |
| 3D Support | Whether outputs 3D coordinates | yes/no |
| Deployment Time | Time from start to first successful run | minutes |
| Status | pass / fail | string |

### Elimination Gate (Codex elimination pattern)

`
Each model:
  ?? 30 min deployment timer
  Timeout ??Record "Fail - Environment incompatible" + skip

Pass threshold:
  - Detect Rate >= 30%
  - FPS >= 10 (on high-end GPU)
  - Otherwise ??Fail, record reason
`

---

## 5. Phase 4: Results Aggregation

### Data Flow
`
results/logs/
  ????? AWR-Net_benchmark.log
  ????? RTMPose-Hand_benchmark.log
  ????? ...
  ????? benchmark_summary.py
           ??results/benchmark_results.csv
           ??results/model_comparison.md
`

### Deliverables
- [ ] 
esults/benchmark_results.csv (machine-readable)
- [ ] 
esults/model_comparison.md (human-readable ranked table)
- [ ] Top 1-2 candidate models selected

---

## 6. Today's Execution Schedule

`
09:00 - 10:30  | Phase 1: Camera Pipeline Verification
               |   Verify RGB + Depth + PC + sync
10:30 - 12:00  | Phase 2: Record Unified Test Dataset
               |   3 gloves ? 6 scenes = 18 sequences
13:30 - 16:30  | Phase 3: 6-Model Parallel Benchmark
               |   Dynamic GPU scheduling + elimination
16:30 - 18:00  | Phase 4: Results Aggregation
               |   CSV + MD + top model selection
`

### Success Criteria
- By 11:00: At least 1-2 models running initial results
- If AWR-Net achieves Detect Rate > 90%, FPS > 30: others become reference group
- Final: Find 1-2 models that satisfy: ??? + 21??? + ??(>=30 FPS)

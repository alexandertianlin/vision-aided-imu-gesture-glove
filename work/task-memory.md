# Task Memory ? Main Agent Monitoring

## Role
Main Agent ? Monitoring & Management

## Task Assignments

| Task | Models | GPU Instance | IP | Status |
|------|--------|-------------|-----|--------|
| 1 | AWR-Net ? YOLO-Hand-Pose | instance-123731 | 10.128.0.8 | pending |
| 2 | RTMPose-Hand ? MMPose-Hand | instance-014510 | 10.128.0.9 | pending |
| 3 | IPNet ? HandPointNet | instance-015628 | 10.128.0.11 | pending |
| 4 | Extrinsic Calibration | local | local | pending |

## All 6 Models + 1 Calibration ? Ready
- models/AWR-Net/ ? setup.sh, benchmark.py, config.yaml
- models/YOLO-Hand-Pose/ ? setup.sh, benchmark.py, config.yaml
- models/RTMPose-Hand/ ? setup.sh, benchmark.py, config.yaml
- models/MMPose-Hand/ ? setup.sh, benchmark.py, config.yaml
- models/IPNet/ ? setup.sh, benchmark.py, config.yaml
- models/HandPointNet/ ? setup.sh, benchmark.py, config.yaml
- calibration/ ? calibrate.py, config.yaml

## Monitor
- scripts/task_watch.py ? check status every 60s
- benchmark/utils.py ? shared tools
- scripts/benchmark_summary.py ? final aggregation

## GCP
- Project: future-life-454308-k4
- Zone: us-central1-a
- GPU: L4

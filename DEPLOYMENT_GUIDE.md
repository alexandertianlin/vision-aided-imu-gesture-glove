# Benchmark Deployment Guide ? 3 x GCP L4 GPU

## Architecture Overview

`
                     ???????????????????????????????????????????
                     ?         Orchestrator (This Repo)        ?
                     ?  benchmark/run_all.py                   ?
                     ???????????????????????????????????????????
                           ?          ?          ?
              ??????????????          ?          ??????????????
              ?                       ?                       ?
     ??????????????????   ??????????????????   ??????????????????
     ?  Subagent 1    ?   ?  Subagent 2    ?   ?  Subagent 3    ?
     ?  AWR-Net       ?   ?  RTMPose-Hand  ?   ?  IPNet         ?
     ?  (Depth)       ?   ?  (RGB)         ?   ?  (Depth)       ?
     ?                ?   ?                ?   ?                ?
     ?  10.128.0.8    ?   ?  10.128.0.9    ?   ?  10.128.0.11   ?
     ?  GPU: L4       ?   ?  GPU: L4       ?   ?  GPU: L4       ?
     ??????????????????   ??????????????????   ??????????????????
              ?                   ?                    ?
              ??????????????????????????????????????????
                                  ?
                     ???????????????????????????
                     ?  Results Aggregation    ?
                     ?  scripts/benchmark_summary.py
                     ?  outputs:               ?
                     ?  ? results/logs/*.log   ?
                     ?  ? benchmark_results.csv?
                     ?  ? model_comparison.md  ?
                     ???????????????????????????
`

## Three GCP Instances

| Subagent | Model | Instance Name | Internal IP | Port |
|----------|-------|---------------|-------------|------|
| 1 | AWR-Net | instance-20260612-...-123731 | 10.128.0.8 | SSH |
| 2 | RTMPose-Hand | instance-20260612-...-014510 | 10.128.0.9 | SSH |
| 3 | IPNet | instance-20260612-...-015628 | 10.128.0.11 | SSH |

## Deployment Steps

### Step 1: Upload code to each instance
`ash
# From your local machine, SCP the project to each instance:
# (or use gcloud scp)
gcloud compute scp --recurse ./vision-imu-gesture-glove/ instance-1:~/
gcloud compute scp --recurse ./vision-imu-gesture-glove/ instance-2:~/
gcloud compute scp --recurse ./vision-imu-gesture-glove/ instance-3:~/
`

### Step 2: SSH into each instance and run setup
`ash
# Instance 1 (AWR-Net)
gcloud compute ssh instance-20260612-...-123731 --zone=us-central1-a
cd ~/vision-imu-gesture-glove/models/AWR-Net
chmod +x setup.sh
./setup.sh

# Instance 2 (RTMPose-Hand)
gcloud compute ssh instance-20260612-...-014510 --zone=us-central1-a
cd ~/vision-imu-gesture-glove/models/RTMPose-Hand
chmod +x setup.sh
./setup.sh

# Instance 3 (IPNet)
gcloud compute ssh instance-20260612-...-015628 --zone=us-central1-a
cd ~/vision-imu-gesture-glove/models/IPNet
chmod +x setup.sh
./setup.sh
`

### Step 3: Run benchmarks
`ash
# On each instance, activate venv and run:
cd ~/vision-imu-gesture-glove
source models/AWR-Net/venv/bin/activate   # or rtmpose/ipnet
python benchmark/run_all.py --models awrnet
`

### Step 4: Collect results
`ash
# After all benchmarks complete, copy logs back:
gcloud compute scp instance-1:~/vision-imu-gesture-glove/results/ results/ --recurse
`

## Benchmark Metrics (all models)
| Metric | Description |
|--------|-------------|
| detect_rate | Frames with valid detection / total frames |
| fps | Frames per second (inference only) |
| latency_ms | Per-frame inference latency |
| stability | Keypoint jitter score (1-5, higher=better) |
| keypoints_21 | Whether outputs 21 standard hand keypoints |
| keypoints_3d | Whether outputs 3D coordinates |
| status | pass/fail |
| failure_reason | Reason if failed |

## Success Criteria
- Find 1-2 models satisfying: glove recognition + 21 keypoints + real-time (>=30 FPS)
- AWR-Net target: detect_rate > 90%, FPS > 30

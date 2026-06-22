# Task: Deployment Preparation & Repo Improvement

**Date:** 2026-06-22
**Goal:** Prepare the vision-aided IMU gesture glove repo for deployment on another laptop

## Analysis

### IMU-to-Unity Connection Study (Orbbec Folder)

The Orbbec sensor tools at `sensors/orbbec/` contain:
- `serial_imu_reader.py` -- STM32 protocol parser (0xB5 0xA5 0x55, accel/gyro/euler)
- `orbbec_imu_bridge.py` -- OrbbecSDK ctypes wrapper for camera internal IMU
- `orbbec_viewer_imu.py` -- Multi-modal viewer with video + IMU recording
- `IMUProcessor` class -- Complementary filter for accel/gyro fusion

Key difference: The Orbbec reader uses 115200 baud with raw MPU6050 data and euler
angles computed in firmware. The Vision-Aided repo's Unity SerialReceiver uses
460800 baud with pre-computed quaternions from a different STM32 firmware
(multi-node protocol: palm=0x30, thumb=0x1E, index=0x28, etc.).

### Current Architecture Bottlenecks

1. `VisionFingerCorrectionReceiver.enableVisionCorrection` defaults to **false**
2. No Python IMU monitor tool existed for debugging the serial stream
3. README was incomplete for deployment scenario
4. No `.gitignore` configuration
5. No setup script for Python environment

## Changes Made

| File | Change | Reason |
|------|--------|--------|
| `python/serial_imu_reader.py` | New file | IMU serial monitor adapted from Orbbec patterns |
| `python/run_imu_monitor.bat` | New file | Quick launcher for IMU monitor |
| `python/requirements.txt` | Added pyserial | Dependency for serial communication |
| `setup.bat` | New file | One-click Python venv creation + deps |
| `.gitignore` | New file | Ignore Unity build artifacts, Python caches, local config |
| `README.md` | Rewritten | Complete architecture, layout, quick start, deployment ref |
| `docs/DEPLOYMENT_GUIDE.md` | New file | Step-by-step deployment for another laptop |
| `docs/task-20260622-deployment-prep.md` | New file | This document |

## Deployment Strategy

1. Clone repo on target laptop
2. Run `setup.bat` for Python environment
3. Open `unity/` in Unity 2022.3 LTS
4. Connect sensor glove (COM3, 460800)
5. Run vision pipeline
6. Press Play in Unity
7. Enable vision correction in Inspector

## Next Steps

- [ ] ViTPose/HAMER integration (after GPU arrives)
- [ ] Depth camera wrist position tracking
- [ ] Multi-camera fusion for occlusion handling

---

*Last updated: 2026-06-22*

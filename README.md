# Vision-Aided IMU Gesture Glove

Real-time sensor glove + camera hand tracking fusion system.

The project fuses wearable IMU/tactile sensing with camera-based MediaPipe Hand
recognition to drive a Unity 3D hand. Camera-based visual anchors correct
long-term IMU drift only at reliable gesture states (open palm, fist).

## System Architecture

```
                  +-------------+         +---------------------+
                  | Sensor Glove| 串口   | Unity               |
                  | STM32 IMU   | COM3   | SerialReceiver      |
                  | 6x MPU6050  | 460800 | HandMotionManager   |
                  | + Tactile   +------->+ FingerSolver        +---> 3D Hand
                  |             |        | VisionCorrection    |
                  +-------------+        +---------^-----------+
                                                   |
                  +-------------+         +--------+-----------+
                  | Camera      | UDP     | Python             |
                  | iPhone/Droid| port    | MediaPipe Hand     |
                  | Orbbec Astra| 5055    | Per-finger Classify|
                  | Webcam      +-------->+ Orientation Gate   |
                  |             |         | Calibration DB     |
                  +-------------+         +--------------------+
```

## Highlights

- Real-time Unity 3D hand driven by wearable sensor glove (6 IMU nodes)
- Serial quaternion + force data at 460800 baud
- MediaPipe Hand visual correction over UDP (port 5055)
- Per-finger open/fist calibration database (up to 100 samples/state)
- Palm/back/side orientation gating
- Open-palm refresh module resets IMU drift baseline
- Full diagnostic trace: UDP_RECV -> PARSE_OK -> FILTER -> TAKEOVER -> FINISH
- Python serial IMU monitor tool for offline debugging

## Repository Layout

```text
unity/
  Assets/Scenes/
    SerialReceiver.cs                   -- IMU serial read (COM3, 460800)
    HandMotionManager.cs                -- IMU quaternion -> hand pose
    FingerSolver.cs                     -- Per-finger IK tracking
    VisionFingerCorrectionReceiver.cs   -- UDP vision anchor receiver
    VisionOpenPalmRefreshModule.cs      -- Auto recalibrate on open palm

python/
  mediapipe_udp_sender.py              -- Vision pipeline + UDP sender
  serial_imu_reader.py                 -- IMU serial monitor tool
  run_mediapipe_udp_sender.bat         -- Vision launcher
  run_imu_monitor.bat                  -- IMU monitor launcher
  requirements.txt                     -- Python deps

setup.bat                              -- One-click Python env setup
docs/DEPLOYMENT_GUIDE.md               -- Deployment to another laptop
```

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Sensor Glove | STM32 + 6x MPU6050, USB serial quaternion/force |
| Unity Laptop | Windows, Unity 2022.3 LTS, GPU 4GB+ VRAM |
| Camera | iPhone (DroidCam) / Orbbec Astra Plus / any USB webcam |

## Quick Start

### 1. Python Setup

```powershell
cd python
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

### 2. Run Vision Pipeline

Edit camera URL in `mediapipe_udp_sender.py` (default: iPhone DroidCam),
then run:

```powershell
python mediapipe_udp_sender.py
```

Auto-calibration: 20s per finger, move each finger open/close.
Then enters runtime per-finger open/fist/IDLE classification.

### 3. Unity Scene

1. Open `unity/` in Unity 2022.3 LTS
2. Open `Assets/Scenes/SampleScene.unity`
3. Connect sensor glove via USB, verify COM3 in SerialReceiver Inspector
4. Press Play
5. Enable VisionFingerCorrectionReceiver.enableVisionCorrection = true for fusion

### 4. IMU Monitor

```powershell
python python/serial_imu_reader.py --port COM3 --baud 460800
```

## Diagnostic Log

Unity writes `vision_finger_diagnostic.log` to Application.persistentDataPath.

| Stage | Meaning |
|-------|---------|
| UDP_RECV | Packet received |
| PARSE_OK | JSON parsed |
| REJECT_BY_CONFIDENCE | vis_conf below threshold, check lighting |
| TAKEOVER_START | Visual correction started |
| CANCEL | Correction cancelled by timeout/IDLE |

## Deployment to Another Laptop

See `docs/DEPLOYMENT_GUIDE.md`.

## Roadmap

- [x] Orbbec camera + serial IMU recording
- [x] MediaPipe Hand + UDP sender
- [x] Unity IMU receiver + HandMotionManager IK
- [x] Vision correction with diagnostic tracing
- [x] Serial IMU monitor tool
- [ ] ViTPose/HAMER integration
- [ ] Depth camera wrist tracking

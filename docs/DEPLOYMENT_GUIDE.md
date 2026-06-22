# Deployment Guide -- Vision-Aided IMU Gesture Glove

> How to deploy the full system (sensor glove + camera vision + Unity 3D hand)
> on a second laptop.

## Overview

The system has three processes that run concurrently:

```text
[Laptop 1 -- Dev machine]         [Laptop 2 -- Target machine]

  Unity Editor                        (same repo, same steps)
    + SerialReceiver (COM3)
    + HandMotionManager
    + FingerSolver

  Python vision pipeline
    MediaPipe Hand detection
    UDP -> port 5055 -> Unity
```

Both Python (vision) and Unity (rendering + IMU) run on the same laptop.
The sensor glove connects via USB serial (COM3).
The camera can be an iPhone (DroidCam over WiFi) or USB webcam.

---

## Step 1: Clone the Repository

```powershell
cd C:\Projects
git clone https://github.com/alexandertianlin/vision-imu-gesture-glove.git
cd vision-imu-gesture-glove
```

Or download and extract the ZIP.

---

## Step 2: Python Setup

```powershell
cd python
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

The `requirements.txt` includes:
- opencv-python (camera capture + display)
- mediapipe (hand landmark detection)
- numpy
- pyyaml
- pyserial (IMU monitor tool)

> **Note:** If `python` command opens Microsoft Store, go to
> Settings > App execution aliases and disable the python.exe aliases,
> then install Python from [python.org](https://python.org).

---

## Step 3: Unity Setup

### Prerequisites

- Unity Hub + Unity 2022.3 LTS installed
- Windows 10/11
- At least 4GB free disk space for the Unity project

### Steps

1. Open Unity Hub
2. Click "Add project from disk" and select the `unity/` folder
3. Open the project (it may take a few minutes to import on first load)
4. Go to `Assets/Scenes/SampleScene.unity`
5. In the Hierarchy, find the GameObject with SerialReceiver
6. In the Inspector, verify:
   - Port: COM3
   - Baud: 460800
   - Finger Solver list has all 5 fingers bound

---

## Step 4: Hardware Connection

### Sensor Glove

1. Connect the sensor glove to USB
2. Open Device Manager > Ports (COM & LPT)
3. Find the USB Serial Device (likely COM3)
4. If not COM3, update the port in SerialReceiver.cs Inspector
5. Verify with the IMU monitor:
   ```powershell
   cd python
   python serial_imu_reader.py --port COM3 --baud 460800
   ```
   You should see live quaternion data for all 6 devices

### Camera Setup

**Option A: iPhone DroidCam (recommended for first testing)**

1. Install DroidCam app on iPhone
2. Ensure both laptop and iPhone are on the same WiFi
3. Open DroidCam, note the IP and port (e.g., 192.168.2.139:4747)
4. Edit `python/mediapipe_udp_sender.py`:
   - Change `RTSP_URL` to your DroidCam URL

**Option B: Orbbec Astra Plus**

1. Install Orbbec SDK drivers
2. Connect camera via USB 3.0
3. The camera appears as a standard UVC device (camera_id 0 or 1)
4. Edit `mediapipe_udp_sender.py` to use `cv2.VideoCapture(camera_id)`

**Option C: USB Webcam**

1. Connect webcam
2. In `mediapipe_udp_sender.py`, replace `RTSP_URL` with `camera_id = 0`

---

## Step 5: Run the Full System

### Terminal 1: Start Python Vision Pipeline

```powershell
cd python
.venv\Scripts\Activate
python mediapipe_udp_sender.py
```

Wait for calibration to complete (20s per finger).

### Terminal 2: Start Unity

1. In Unity Editor, press Play
2. Press Space to calibrate the IMU baseline
3. The hand should follow glove movements

### Terminal 3 (optional): IMU Monitor

```powershell
cd python
python serial_imu_reader.py --port COM3 --baud 460800
```

### Enable Vision Correction

In Unity Editor:
1. Select the GameObject with `VisionFingerCorrectionReceiver`
2. In Inspector, check `enableVisionCorrection = true`
3. The diagnostic log `vision_finger_diagnostic.log` will show reception

---

## Step 6: Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| No IMU data in Unity | Wrong COM port | Check Device Manager, update SerialReceiver |
| Hand does not move after calibration | Serial baud mismatch | Verify 460800 in both firmware and Unity |
| No UDP packets received | Firewall blocking port 5055 | Add Windows Firewall rule for port 5055 |
| Vision correction not applied | enableVisionCorrection=false | Check toggle in Inspector |
| `REJECT_BY_CONFIDENCE` in log | Poor lighting or hand angle | Improve lighting, keep palm facing camera |
| Python cannot find camera | Wrong URL or camera ID | Test with VLC or OpenCV first |
| `pyserial` import error | Missing pyserial | `pip install pyserial` |

---

## Step 7: RTX 4060 Deployment Notes (ViTPose / HAMER)

When the GPU arrives, deploy ViTPose or HAMER by:
1. Replacing `mediapipe_udp_sender.py` logic with ViTPose inference
2. Keeping the same UDP packet format (port 5055, same JSON schema)
3. Running on the RTX 4060 laptop (8GB VRAM, FP16 inference)

Recommended model sizes:
- ViTPose-B (86M params): ~2-3GB VRAM, ~20 FPS on RTX 4060
- RTMPose-m (lightweight alternative): ~1GB VRAM, ~60 FPS

> See the main README for architecture details.

---

*Last updated: 2026-06-22*

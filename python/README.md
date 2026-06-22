# Python Vision Pipeline

Two versions available:

## V1: iPhone DroidCam (mediapipe_udp_sender.py)

Original version using iPhone as camera via DroidCam RTSP stream.
Same UDP port 5055, same calibration pipeline.

Edit `RTSP_URL` at the top of the file to match your DroidCam IP.

## V2: Intel D435i (mediapipe_udp_sender_v2.py)

New version using Intel RealSense D435i RGB camera.
Same pipeline, same UDP format, same calibration.

Camera auto-detects ID 0-4, configurable at top of file.

## Usage

```powershell
.venv\Scripts\Activate
python mediapipe_udp_sender_v2.py    # V2: D435i
python mediapipe_udp_sender.py       # V1: iPhone DroidCam
```

## IMU Monitor

```powershell
python serial_imu_reader.py --port COM3 --baud 460800
```

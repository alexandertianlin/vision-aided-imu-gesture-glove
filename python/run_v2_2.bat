@echo off
REM V2.2: D435i + MediaPipe continuous finger angle sender -> Unity (port 5058)
REM Requires MediaPipeAngleReceiver.cs attached to HandMotionManager in Unity.
cd /d "%~dp0"
.venv\Scripts\activate && python mediapipe_udp_sender_v2_2.py

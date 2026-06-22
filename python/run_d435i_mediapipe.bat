@echo off
REM D435i + MediaPipe Hand -> Unity UDP sender (V2)
REM Based on V1 (run_mediapipe_udp_sender.bat) for iPhone DroidCam
cd /d "%~dp0"
.venv\Scripts\activate && python mediapipe_udp_sender_v2.py

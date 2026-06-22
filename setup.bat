@echo off
REM ===================================================
REM Vision-Aided IMU Gesture Glove - Python Setup
REM ===================================================
cd /d "%~dp0python"
echo [SETUP] Creating Python virtual environment...
python -m venv .venv
echo [SETUP] Activating virtual environment...
call .venv\Scripts\activate.bat
echo [SETUP] Installing dependencies...
pip install -r requirements.txt
echo [SETUP] Done!
echo.
echo ================================================
echo Available commands:
echo   python mediapipe_udp_sender.py       -- Vision pipeline (MediaPipe + UDP)
echo   python serial_imu_reader.py --monitor -- IMU serial monitor
echo   call run_imu_monitor.bat             -- Quick IMU monitor
echo ================================================
pause

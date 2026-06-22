@echo off
REM Serial IMU Monitor Launcher
REM Usage: run_imu_monitor.bat [--port COM3] [--baud 460800] [--csv output.csv]
cd /d "%~dp0"
python serial_imu_reader.py %*

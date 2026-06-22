@echo off
REM ===================================================
REM Bug Log Capture Tool
REM Captures Unity Editor.log, Python crash dump, and
REM system info into a timestamped folder.
REM Usage: capture_logs [crash_description]
REM ===================================================
setlocal
set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set LOGDIR=%~dp0..\crash_logs\%TIMESTAMP%
if not "%1"=="" set LOGDIR=%~dp0..\crash_logs\%TIMESTAMP%_%1
mkdir "%LOGDIR%" 2>nul

echo ========================================
echo  Capturing logs to: %LOGDIR%
echo ========================================

REM 1. Capture Unity Editor.log
if exist "%USERPROFILE%\AppData\Local\Unity\Editor\Editor.log" (
    copy "%USERPROFILE%\AppData\Local\Unity\Editor\Editor.log" "%LOGDIR%\Unity_Editor.log" >nul
    echo [OK] Unity Editor.log captured
) else (
    echo [WARN] Unity Editor.log not found
)

REM 2. Capture Unity crash dumps
if exist "%USERPROFILE%\AppData\Local\Temp\Unity\Editor\Crashes" (
    xcopy /E /I "%USERPROFILE%\AppData\Local\Temp\Unity\Editor\Crashes" "%LOGDIR%\Crashes\" >nul 2>nul
    echo [OK] Unity crash dumps captured
)

REM 3. Capture GPU info
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv > "%LOGDIR%\gpu_info.txt" 2>nul

REM 4. Capture process list
tasklist /FI "IMAGENAME eq Unity.exe" /V > "%LOGDIR%\unity_process.txt" 2>nul
tasklist /FI "IMAGENAME eq python.exe" /V > "%LOGDIR%\python_process.txt" 2>nul

echo ========================================
echo  Logs saved to: %LOGDIR%
echo  Share this folder for debugging.
echo ========================================

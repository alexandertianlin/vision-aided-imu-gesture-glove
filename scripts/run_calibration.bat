@echo off
REM Stereo Calibration Pipeline for Orbbec Astra Plus
SETLOCAL ENABLEDELAYEDEXPANSION

SET ROOT_DIR=%~dp0..
cd /d "%ROOT_DIR%"

IF EXIST "python\.venv\Scripts\activate.bat" (
    CALL python\.venv\Scripts\activate.bat
)

echo ============================================
echo  Orbbec Astra Plus - Calibration Pipeline
echo ============================================
echo.

echo [Step 1/4] Testing cameras and capturing depth...
python -m camera.test_depth --n-frames 30 --no-display

echo.
echo [Step 2/4] Running quick stereo calibration test...
python -m camera.calibrate_stereo --quick --output configs\extrinsic_params.json

echo.
echo [Step 3/4] Attempting live stereo calibration...
python -m camera.calibrate_stereo --live --output configs\extrinsic_params.json

echo.
echo [Step 4/4] Verifying calibration output...
IF EXIST "configs\extrinsic_params.json" (
    python -c "import json; d=json.load(open('configs/extrinsic_params.json')); print(f'Baseline: {d.get(\"baseline_mm\",\"?\")} mm'); print(f'Reproj error: {d.get(\"reprojection_error\",\"?\")}'); print(f'Status: {d.get(\"status\",\"?\")}')"
    echo.
    echo [DONE] Calibration complete!
    echo  Results: configs\extrinsic_params.json
) ELSE (
    echo [ERROR] No calibration output found!
    EXIT /B 1
)
echo.
echo Next: python camera\test_depth.py --display

ENDLOCAL

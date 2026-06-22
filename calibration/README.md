# Task 4: Extrinsic Calibration (ChArUco Board)

## Purpose
Compute stereo extrinsic parameters: rotation (R) and translation (T) between left and right cameras.

## Input
- Left/right image pairs of ChArUco board in data/calibration_images/
- Intrinsic parameters from configs/calibration.yaml (or estimated defaults)

## Output
- configs/extrinsic_params.json (R, T, E, F)

## Run
`ash
python calibration/calibrate.py --image-dir data/calibration_images
`

## Notes
- Need at least 3 valid image pairs with ChArUco detection
- ChArUco board: 7x5 squares, 40mm squares, 25mm markers
- Intrinsic params already loaded from calibration.yaml

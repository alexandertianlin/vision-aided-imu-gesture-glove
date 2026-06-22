#!/usr/bin/env python3
"""
Stereo Extrinsic Calibration for Orbbec Astra Plus Dual Cameras.

Computes rotation R and translation T between left and right cameras
using a ChArUco board. Supports both:
  1. Live capture from dual cameras with real-time corner detection
  2. Offline calibration from saved image pairs

Output: configs/extrinsic_params.json (R, T, E, F, reprojection_error)

Usage:
    # Live capture mode
    python camera/calibrate_stereo.py --live

    # Offline from saved images
    python camera/calibrate_stereo.py --image-dir data/calibration_images/stereo

    # Quick single-pair estimate
    python camera/calibrate_stereo.py --quick

Requirements:
    - numpy, opencv-python (with contrib modules for aruco)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stereo Extrinsic Calibration for Orbbec Astra Plus"
    )

    # Input source
    source = parser.add_argument_group("Source")
    source.add_argument("--live", action="store_true",
                        help="Capture live from dual cameras")
    source.add_argument("--image-dir", type=str, default=None,
                        help="Directory with stereo image pairs")
    source.add_argument("--quick", action="store_true",
                        help="Quick single-pair estimate (for testing)")
    source.add_argument("--intrinsic", type=str, default="",
                        help="Path to intrinsic params JSON (optional)")

    # Capture settings
    capture = parser.add_argument_group("Capture")
    capture.add_argument("--left-id", type=int, default=0,
                         help="Left camera device ID")
    capture.add_argument("--right-id", type=int, default=1,
                         help="Right camera device ID")
    capture.add_argument("--n-pairs", type=int, default=15,
                         help="Number of image pairs to capture (live mode)")
    capture.add_argument("--capture-delay", type=float, default=0.5,
                         help="Seconds between captures (live mode)")

    # ChArUco board
    board = parser.add_argument_group("ChArUco Board")
    board.add_argument("--rows", type=int, default=7,
                       help="Number of ChArUco board rows")
    board.add_argument("--cols", type=int, default=5,
                       help="Number of ChArUco board columns")
    board.add_argument("--square-len", type=float, default=0.040,
                       help="Square side length in meters")
    board.add_argument("--marker-len", type=float, default=0.025,
                       help="Marker side length in meters")

    # Output
    output = parser.add_argument_group("Output")
    output.add_argument("--output", type=str,
                        default="configs/extrinsic_params.json",
                        help="Output path for extrinsic params JSON")
    output.add_argument("--save-images", action="store_true", default=True,
                        help="Save captured calibration images")
    output.add_argument("--save-dir", type=str,
                        default="data/calibration_images/stereo",
                        help="Directory to save calibration images")
    output.add_argument("--result-dir", type=str, default="results/")

    return parser.parse_args()


def setup_charuco(rows=7, cols=5, square_length=0.04, marker_length=0.025):
    """Create ChArUco board and detection parameters."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (cols, rows), square_length, marker_length, aruco_dict
    )
    detector_params = cv2.aruco.DetectorParameters()
    return board, aruco_dict, detector_params


def detect_charuco_corners(
    image: np.ndarray, board, aruco_dict, detector_params
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """Detect ChArUco corners in an image.

    Returns (success, charuco_corners, charuco_ids).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
    if ids is None or len(ids) < 4:
        return False, None, None
    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    return ret > 0, charuco_corners, charuco_ids


def match_corners_across_views(
    cc_left, ci_left, cc_right, ci_right, board
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match corresponding ChArUco corners between left and right views.

    Returns (obj_pts, img_pts_left, img_pts_right).
    """
    obj_pts, img_l, img_r = [], [], []
    chessboard_corners = cv2.aruco.CharucoBoard.getChessboardCorners(board)

    for i in range(len(ci_left)):
        for j in range(len(ci_right)):
            if int(ci_left[i]) == int(ci_right[j]):
                idx = int(ci_left[i])
                if idx < len(chessboard_corners):
                    obj_pts.append(chessboard_corners[idx])
                    img_l.append(cc_left[i])
                    img_r.append(cc_right[j])
                break

    if len(obj_pts) < 4:
        return np.array([]), np.array([]), np.array([])

    return (
        np.array(obj_pts, dtype=np.float32).reshape(-1, 1, 3),
        np.array(img_l, dtype=np.float32).reshape(-1, 1, 2),
        np.array(img_r, dtype=np.float32).reshape(-1, 1, 2),
    )


def capture_live_pairs(
    left_id: int, right_id: int,
    n_pairs: int, delay: float,
    save_dir: str, save_images: bool,
    logger: logging.Logger,
    board=None, aruco_dict=None, detector_params=None,
) -> Tuple[List, List, List, Tuple[int, int]]:
    """Capture stereo image pairs live from dual cameras.

    Returns (obj_pts_list, img_pts_left_list, img_pts_right_list, image_size).
    """
    logger.info(f"Opening cameras: Left={left_id}, Right={right_id}")
    cap_left = cv2.VideoCapture(left_id, cv2.CAP_DSHOW)
    cap_right = cv2.VideoCapture(right_id, cv2.CAP_DSHOW)

    for cap in [cap_left, cap_right]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap_left.isOpened() or not cap_right.isOpened():
        logger.error("Failed to open one or both cameras")
        return [], [], [], (0, 0)

    # Warm up
    for _ in range(10):
        cap_left.read()
        cap_right.read()

    image_size = (640, 480)
    obj_pts_list, img_l_list, img_r_list = [], [], []
    captured_count = 0

    if save_images and save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Ready to capture {n_pairs} calibration image pairs.")
    logger.info(f"Hold the ChArUco board in front of both cameras.")
    logger.info(f"Press:")
    logger.info(f"  [SPACE] to capture current pair")
    logger.info(f"  [q] to quit and calibrate with collected pairs")
    logger.info(f"{'='*60}")

    preview_window = "Stereo Calibration - Left (SPACE=capture, q=quit)"

    while captured_count < n_pairs:
        ret_l, left = cap_left.read()
        ret_r, right = cap_right.read()
        if not ret_l or not ret_r:
            continue

        # Detect corners for visual feedback
        if board and aruco_dict and detector_params:
            ok_l, cc_l, ci_l = detect_charuco_corners(left, board, aruco_dict, detector_params)
            ok_r, cc_r, ci_r = detect_charuco_corners(right, board, aruco_dict, detector_params)

            # Draw detected corners
            display = left.copy()
            if ok_l:
                cv2.aruco.drawDetectedCornersCharuco(display, cc_l, ci_l)
            status_text = f"Left: {'OK' if ok_l else 'NO BOARD'} | Right: {'OK' if ok_r else 'NO BOARD'}"
            cv2.putText(display, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if (ok_l and ok_r) else (0, 0, 255), 2)
            cv2.putText(display, f"Captured: {captured_count}/{n_pairs}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            display = left.copy()
            cv2.putText(display, f"Captured: {captured_count}/{n_pairs}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(preview_window, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") or key == ord("c"):
            # Capture this pair
            if board and aruco_dict and detector_params:
                ok_l, cc_l, ci_l = detect_charuco_corners(left, board, aruco_dict, detector_params)
                ok_r, cc_r, ci_r = detect_charuco_corners(right, board, aruco_dict, detector_params)

                if ok_l and ok_r and len(ci_l) >= 4 and len(ci_r) >= 4:
                    obj_pts, img_l, img_r = match_corners_across_views(
                        cc_l, ci_l, cc_r, ci_r, board
                    )
                    if len(obj_pts) >= 4:
                        obj_pts_list.append(obj_pts)
                        img_l_list.append(img_l)
                        img_r_list.append(img_r)
                        captured_count += 1
                        logger.info(f"  [{captured_count}/{n_pairs}] Captured pair with "
                                    f"{len(obj_pts)} matched corners")

                        if save_images and save_dir:
                            ts = time.strftime("%Y%m%d_%H%M%S")
                            cv2.imwrite(f"{save_dir}/left_{ts}_{captured_count:02d}.png", left)
                            cv2.imwrite(f"{save_dir}/right_{ts}_{captured_count:02d}.png", right)
                    else:
                        logger.warning("  Not enough matched corners, try again")
                else:
                    logger.warning("  Board not visible in both cameras, try again")
            else:
                # Save raw image even without detection
                captured_count += 1
                if save_images and save_dir:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"{save_dir}/left_{ts}_{captured_count:02d}.png", left)
                    cv2.imwrite(f"{save_dir}/right_{ts}_{captured_count:02d}.png", right)
                logger.info(f"  [{captured_count}/{n_pairs}] Saved raw pair")

        elif key == ord("q"):
            logger.info("User quit capture session")
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyWindow(preview_window)

    return obj_pts_list, img_l_list, img_r_list, image_size


def load_image_pairs(
    image_dir: str, logger: logging.Logger,
    board=None, aruco_dict=None, detector_params=None,
) -> Tuple[List, List, List, Tuple[int, int]]:
    """Load stereo image pairs from directory and detect ChArUco corners."""
    image_dir = Path(image_dir)

    # Find pairs using pattern matching
    left_images = sorted(image_dir.glob("*left*")) + sorted(image_dir.glob("*L*"))
    right_images = sorted(image_dir.glob("*right*")) + sorted(image_dir.glob("*R*"))

    if not left_images or not right_images:
        logger.error(f"No left/right image pairs found in {image_dir}")
        return [], [], [], (0, 0)

    # Match by index in sorted lists (assumes same ordering)
    pairs = list(zip(left_images, right_images))
    logger.info(f"Found {len(pairs)} image pairs in {image_dir}")

    obj_pts_list, img_l_list, img_r_list = [], [], []
    image_size = (0, 0)

    for i, (left_path, right_path) in enumerate(pairs):
        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        if left is None or right is None:
            logger.warning(f"  [{i+1}] Failed to load pair, skipping")
            continue

        if i == 0:
            image_size = (left.shape[1], left.shape[0])

        if board and aruco_dict and detector_params:
            ok_l, cc_l, ci_l = detect_charuco_corners(left, board, aruco_dict, detector_params)
            ok_r, cc_r, ci_r = detect_charuco_corners(right, board, aruco_dict, detector_params)

            if ok_l and ok_r and len(ci_l) >= 4 and len(ci_r) >= 4:
                obj_pts, img_l, img_r = match_corners_across_views(
                    cc_l, ci_l, cc_r, ci_r, board
                )
                if len(obj_pts) >= 4:
                    obj_pts_list.append(obj_pts)
                    img_l_list.append(img_l)
                    img_r_list.append(img_r)
                    logger.info(f"  [{i+1}/{len(pairs)}] {left_path.name}: {len(obj_pts)} corners")
                else:
                    logger.info(f"  [{i+1}/{len(pairs)}] Insufficient matched corners")
            else:
                logger.info(f"  [{i+1}/{len(pairs)}] Board not detected: left={ok_l}, right={ok_r}")
        else:
            # No detection - just add empty placeholders
            logger.info(f"  [{i+1}/{len(pairs)}] Loaded (no detection)")

    return obj_pts_list, img_l_list, img_r_list, image_size


def load_intrinsics(path: str, logger: logging.Logger):
    """Load intrinsic parameters from JSON or YAML file."""
    if not path or not os.path.exists(path):
        logger.info("No intrinsic file found, using default estimates")
        return (
            np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64),
            np.zeros(5, dtype=np.float64),
            np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64),
            np.zeros(5, dtype=np.float64),
        )

    with open(path, "r") as f:
        data = json.load(f)

    K1 = np.array(data.get("K_left", data.get("K", np.eye(3))), dtype=np.float64)
    K2 = np.array(data.get("K_right", data.get("K", np.eye(3))), dtype=np.float64)
    D1 = np.array(data.get("dist_left", data.get("dist", np.zeros(5))), dtype=np.float64)
    D2 = np.array(data.get("dist_right", data.get("dist", np.zeros(5))), dtype=np.float64)

    # Ensure 1x5 shape
    if len(D1.shape) == 0:
        D1 = np.zeros(5, dtype=np.float64)
    if len(D2.shape) == 0:
        D2 = np.zeros(5, dtype=np.float64)

    logger.info(f"Loaded intrinsics from {path}")
    logger.info(f"  K_left: fx={K1[0,0]:.1f}, fy={K1[1,1]:.1f}, cx={K1[0,2]:.1f}, cy={K1[1,2]:.1f}")
    logger.info(f"  K_right: fx={K2[0,0]:.1f}, fy={K2[1,1]:.1f}, cx={K2[0,2]:.1f}, cy={K2[1,2]:.1f}")
    return K1, D1, K2, D2


def run_stereo_calibration(
    obj_pts_list: List[np.ndarray],
    img_pts_left: List[np.ndarray],
    img_pts_right: List[np.ndarray],
    image_size: Tuple[int, int],
    K1: np.ndarray, D1: np.ndarray,
    K2: np.ndarray, D2: np.ndarray,
    logger: logging.Logger,
) -> dict:
    """Run OpenCV stereo calibration and return results dict."""
    if len(obj_pts_list) < 3:
        logger.error(f"FAIL: Only {len(obj_pts_list)} valid pairs (need >= 3)")
        logger.info("status: fail")
        logger.info("failure_reason: Insufficient calibration pairs")
        return {"status": "fail", "failure_reason": "Insufficient calibration pairs"}

    logger.info(f"\nRunning stereo calibration with {len(obj_pts_list)} pairs...")
    calib_start = time.time()

    # Prepare image size for calibration
    if image_size == (0, 0):
        image_size = (640, 480)

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    try:
        ret, K1_out, D1_out, K2_out, D2_out, R, T, E, F = cv2.stereoCalibrate(
            obj_pts_list, img_pts_left, img_pts_right,
            K1, D1, K2, D2,
            image_size,
            criteria=criteria,
            flags=flags,
        )
    except cv2.error as e:
        logger.error(f"OpenCV stereoCalibrate failed: {e}")
        return {"status": "fail", "failure_reason": f"OpenCV error: {str(e)}"}

    elapsed_min = (time.time() - calib_start) / 60.0
    logger.info(f"\n{'='*50}")
    logger.info(f"Calibration Complete!")
    logger.info(f"{'='*50}")
    logger.info(f"Reprojection error: {ret:.6f}")
    logger.info(f"Translation T (mm): [{T[0,0]*1000:.1f}, {T[1,0]*1000:.1f}, {T[2,0]*1000:.1f}]")
    logger.info(f"Rotation R:\n{R}")

    # Convert rotation matrix to Euler angles (degrees)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        rx = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = 0
    logger.info(f"Euler angles (deg): Rx={rx:.3f}, Ry={ry:.3f}, Rz={rz:.3f}")

    # Baseline distance
    baseline_m = float(np.linalg.norm(T))
    logger.info(f"Baseline distance: {baseline_m*1000:.1f} mm")

    result = {
        "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calibration_method": "ChArUco Stereo",
        "calibration_board": {
            "type": "ChArUco",
            "rows": 7,
            "cols": 5,
            "square_length_mm": 40,
            "marker_length_mm": 25,
        },
        "reprojection_error": float(ret),
        "baseline_mm": float(baseline_m * 1000),
        "R": R.tolist(),
        "T": T.tolist(),
        "euler_angles_deg": {
            "rx": float(rx), "ry": float(ry), "rz": float(rz)
        },
        "E": E.tolist(),
        "F": F.tolist(),
        "K_left": K1_out.tolist() if isinstance(K1_out, np.ndarray) else K1.tolist(),
        "K_right": K2_out.tolist() if isinstance(K2_out, np.ndarray) else K2.tolist(),
        "dist_left": D1_out.tolist() if isinstance(D1_out, np.ndarray) else D1.tolist(),
        "dist_right": D2_out.tolist() if isinstance(D2_out, np.ndarray) else D2.tolist(),
        "image_size": list(image_size),
        "n_pairs_used": len(obj_pts_list),
        "deployment_time_min": f"{elapsed_min:.2f}",
        "status": "pass",
    }

    return result


def save_results(result: dict, output_path: str, logger: logging.Logger):
    """Save calibration results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Extrinsic params saved to {output_path}")


def generate_quick_test_images(save_dir: str, logger: logging.Logger, n_pairs=6):
    """Generate synthetic ChArUco-like test images for verification."""
    logger.info("Generating sample calibration images for testing...")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_pairs):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        for r in range(7):
            for c in range(5):
                x, y = 100 + c * 80, 80 + r * 60
                cv2.rectangle(img, (x - 15, y - 15), (x + 15, y + 15), (0, 0, 0), -1)
                cv2.rectangle(img, (x - 8, y - 8), (x + 8, y + 8), (255, 255, 255), -1)
        # Add slight offset for each pair
        offset = i * 5
        M = np.float32([[1, 0, offset], [0, 1, offset * 0.5]])
        shifted = cv2.warpAffine(img, M, (640, 480))
        cv2.imwrite(str(save_dir / f"left_{i:02d}.png"), shifted)
        cv2.imwrite(str(save_dir / f"right_{i:02d}.png"), shifted)
    logger.info(f"Generated {n_pairs} test pairs in {save_dir}")


def main():
    args = parse_args()

    # Setup logger
    log_dir = Path(args.result_dir) / "logs"
    logger = setup_logger("Stereo-Calibration", str(log_dir))
    logger.info("=" * 60)
    logger.info("Task 4: Stereo Extrinsic Calibration (Orbbec Astra Plus)")
    logger.info("=" * 60)

    # ChArUco board setup
    board, aruco_dict, detector_params = setup_charuco(
        args.rows, args.cols, args.square_len, args.marker_len
    )
    logger.info(f"ChArUco board: {args.cols}x{args.rows}, "
                f"{args.square_len*1000:.0f}mm squares, "
                f"{args.marker_len*1000:.0f}mm markers")

    # Generate quick test images if requested
    if args.quick:
        test_dir = Path(args.save_dir) / "test"
        generate_quick_test_images(str(test_dir), logger)
        args.image_dir = str(test_dir)

    # Collect image pairs
    if args.live:
        logger.info("Mode: Live capture from dual cameras")
        obj_pts, img_l, img_r, img_size = capture_live_pairs(
            args.left_id, args.right_id,
            args.n_pairs, args.capture_delay,
            args.save_dir, args.save_images,
            logger, board, aruco_dict, detector_params,
        )
    elif args.image_dir:
        logger.info(f"Mode: Offline from {args.image_dir}")
        obj_pts, img_l, img_r, img_size = load_image_pairs(
            args.image_dir, logger, board, aruco_dict, detector_params,
        )
    else:
        logger.error("No input source specified. Use --live or --image-dir.")
        return

    if len(obj_pts) < 3:
        if not args.quick and not args.image_dir:
            logger.info("Not enough valid pairs. Generating test images...")
            test_dir = Path(args.save_dir) / "test"
            generate_quick_test_images(str(test_dir), logger)
            args.image_dir = str(test_dir)
            obj_pts, img_l, img_r, img_size = load_image_pairs(
                args.image_dir, logger, board, aruco_dict, detector_params,
            )
        if len(obj_pts) < 3:
            logger.error("FAIL: Insufficient calibration data")
            return

    # Load intrinsics (or use defaults)
    K1, D1, K2, D2 = load_intrinsics(args.intrinsic, logger)

    # Run stereo calibration
    result = run_stereo_calibration(
        obj_pts, img_l, img_r, img_size,
        K1, D1, K2, D2, logger,
    )

    # Save results
    save_results(result, args.output, logger)

    if result.get("status") == "pass":
        baseline = result.get("baseline_mm", 0)
        logger.info(f"\n{'='*50}")
        logger.info(f"  CALIBRATION SUCCESSFUL")
        logger.info(f"  Baseline: {baseline:.1f} mm")
        logger.info(f"  Reprojection error: {result['reprojection_error']:.4f}")
        logger.info(f"{'='*50}")
        # Also print to stdout for easy reading
        print(f"\n  Calibration Complete!")
        print(f"  Baseline distance: {baseline:.1f} mm")
        print(f"  Translation: [{result['T'][0][0]*1000:.1f}, "
              f"{result['T'][1][0]*1000:.1f}, {result['T'][2][0]*1000:.1f}] mm")
        euler = result.get("euler_angles_deg", {})
        print(f"  Euler angles: Rx={euler.get('rx',0):.2f}, "
              f"Ry={euler.get('ry',0):.2f}, Rz={euler.get('rz',0):.2f} deg")
        print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()

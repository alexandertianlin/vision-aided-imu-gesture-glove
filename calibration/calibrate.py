#!/usr/bin/env python3
"""
Task 4: Extrinsic Calibration — ChArUco Board Stereo Calibration
Output: configs/extrinsic_params.json (R, T, E, F)

Requirements:
- Left and right camera images of ChArUco board
- Intrinsic parameters already known (loaded from configs/calibration.yaml)

OpenCV 4.13.0+ uses CharucoDetector/ArucoDetector (new unified API).
"""

import argparse, json, os, sys, time
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.utils import setup_logger

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_intrinsics(path):
    """Load camera intrinsics from YAML or JSON.

    Supports:
      - Single-camera file: {"K": ..., "dist": ...}  (copied to both)
      - Stereo file:        {"K_left": ..., "K_right": ...,
                              "dist_left": ..., "dist_right": ...}
      - YAML with same structure
    Returns (K1, D1, K2, D2) as np.float64 arrays.
    """
    K_default = np.array([[600, 0, 320],
                           [0, 600, 240],
                           [0, 0, 1]], dtype=np.float64)
    D_default = np.zeros(5, dtype=np.float64)

    if not path or not os.path.exists(path):
        print(f"  [WARN] Intrinsic file not found: {path} \u2014 using defaults")
        return K_default, D_default, K_default.copy(), D_default.copy()

    data = None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        if yaml is not None:
            with open(path) as f:
                data = yaml.safe_load(f)

    if data is None:
        print(f"  [WARN] Could not parse {path} \u2014 using defaults")
        return K_default, D_default, K_default.copy(), D_default.copy()

    def _to_K(v):
        return np.array(v, dtype=np.float64).reshape(3, 3)

    def _to_D(v):
        a = np.array(v, dtype=np.float64).ravel()
        if a.size < 5:
            a = np.pad(a, (0, 5 - a.size), "constant")
        return a[:5]

    if "K_left" in data and "K_right" in data:
        return (_to_K(data["K_left"]), _to_D(data.get("dist_left", D_default)),
                _to_K(data["K_right"]), _to_D(data.get("dist_right", D_default)))

    if "K" in data:
        K = _to_K(data["K"])
        D = _to_D(data.get("dist", D_default))
        return K, D, K.copy(), D.copy()

    print(f"  [WARN] No recognisable intrinsics keys in {path} \u2014 using defaults")
    return K_default, D_default, K_default.copy(), D_default.copy()


def _generate_test_images(board, image_dir, count=10, img_size=(640, 480)):
    """Generate synthetic ChArUco board images for pipeline testing.

    Renders ArUco markers at correct board positions using known
    projection geometry so CharucoDetector can actually detect them.
    Returns sorted lists of (left_paths, right_paths).
    """
    import math
    cols, rows = board.getChessboardSize()
    sq_len = board.getSquareLength()

    # Board dimensions in metres
    board_w = cols * sq_len
    board_h = rows * sq_len

    # Scale so board fills ~55 % of image
    scale = min(img_size[0] * 0.55 / board_w, img_size[1] * 0.55 / board_h)

    # Rough intrinsics for rendering
    fx = fy = max(img_size) * 1.2
    cx, cy = img_size[0] / 2.0, img_size[1] / 2.0
    K_test = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]], dtype=np.float64)

    test_dir = image_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    left_paths, right_paths = [], []

    for i in range(count):
        angle = math.radians((i - count / 2) * 6)      # +/-30 deg
        dist  = 0.60 + (i % 3) * 0.10                  # 0.6-0.8 m
        h_off = 0.0 + (i % 5 - 2) * 0.02               # +/-4 cm

        # Board pose in world frame
        c, s = math.cos(angle), math.sin(angle)
        R_world = np.array([[c, 0, s],
                             [0, 1, 0],
                             [-s, 0, c]], dtype=np.float64)
        t_world = np.array([0.0, h_off, dist], dtype=np.float64)

        # Left/right camera poses (8 cm baseline)
        baseline = 0.08
        R_left  = np.eye(3, dtype=np.float64)
        t_left  = np.array([-baseline / 2, 0.0, 0.0], dtype=np.float64)
        R_right = np.eye(3, dtype=np.float64)
        t_right = np.array([baseline / 2, 0.0, 0.0], dtype=np.float64)

        # Board \u2192 camera transform: P_c = R_c^T (R_w P_b + t_w - t_c)
        R1 = R_left.T @ R_world
        t1 = R_left.T @ (t_world - t_left)
        R2 = R_right.T @ R_world
        t2 = R_right.T @ (t_world - t_right)

        # Render left
        left_img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 200
        _render_markers(left_img, board, K_test, R1, t1)
        cv2.imwrite(str(test_dir / f"left_{i:02d}.png"), left_img)

        # Render right
        right_img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 200
        _render_markers(right_img, board, K_test, R2, t2)
        cv2.imwrite(str(test_dir / f"right_{i:02d}.png"), right_img)

        left_paths.append(test_dir / f"left_{i:02d}.png")
        right_paths.append(test_dir / f"right_{i:02d}.png")

    print(f"  Generated {count} synthetic left/right pairs in {test_dir}")
    return sorted(left_paths), sorted(right_paths)


def _render_markers(img, board, K, rvec, tvec, marker_px=80):
    """Render ArUco markers for the CharucoBoard at a given pose.
    Uses generateImageMarker so that CharucoDetector can detect them.
    """
    d = board.getDictionary()
    obj_corners = board.getObjPoints()
    h, w = img.shape[:2]
    for mk_idx, marker_obj in enumerate(obj_corners):
        marker_img = cv2.aruco.generateImageMarker(d, mk_idx, marker_px)
        marker_img = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        pts3d = np.array(marker_obj, dtype=np.float64).reshape(-1, 3)
        pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
        pts2d = pts2d.reshape(4, 2).astype(np.float32)
        src = np.array([[0, 0], [marker_px, 0], [marker_px, marker_px], [0, marker_px]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, pts2d)
        warped = cv2.warpPerspective(marker_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        ones = np.ones_like(marker_img, dtype=np.uint8) * 255
        mask = cv2.warpPerspective(ones, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask_bool = mask.astype(bool)
        img[mask_bool] = warped[mask_bool]


# ---------------------------------------------------------------------------
# Core calibration
# ---------------------------------------------------------------------------

def calibrate_stereo(logger, image_dir, intrinsic_path, output_path):
    """Run stereo calibration using ChArUco board images."""
    logger.info("=== Task 4: Extrinsic Calibration ===")
    calib_start = time.time()

    # ChArUco board parameters
    # IMPORTANT: These measurements are for the CHESSBOARD AREA only
    # (7 rows x 5 cols, 40mm squares).  Do NOT include the outer mounting
    # border/frame when measuring square_length or marker_length.
    CHARUCO_ROWS = 7
    CHARUCO_COLS = 5
    SQUARE_LENGTH = 0.04      # 4 cm
    MARKER_LENGTH = 0.025     # 2.5 cm

    # Load intrinsic parameters
    K1, D1, K2, D2 = _load_intrinsics(intrinsic_path)
    logger.info(f"Loaded intrinsics from {intrinsic_path}")
    logger.info(f"  K1 fx={K1[0,0]:.1f} fy={K1[1,1]:.1f} "
                f"cx={K1[0,2]:.1f} cy={K1[1,2]:.1f}")
    logger.info(f"  K2 fx={K2[0,0]:.1f} fy={K2[1,1]:.1f} "
                f"cx={K2[0,2]:.1f} cy={K2[1,2]:.1f}")

    # Find image pairs
    image_dir = Path(image_dir)
    left_imgs  = sorted(image_dir.glob("*left*")) + sorted(image_dir.glob("*L*"))
    right_imgs = sorted(image_dir.glob("*right*")) + sorted(image_dir.glob("*R*"))

    if len(left_imgs) < 5 or len(right_imgs) < 5:
        logger.info(f"Need at least 5 image pairs. "
                    f"Found: {len(left_imgs)} left, {len(right_imgs)} right")
        logger.info("Generating synthetic ChArUco calibration images for testing...")
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        board_gen = cv2.aruco.CharucoBoard(
            (CHARUCO_COLS, CHARUCO_ROWS),
            SQUARE_LENGTH, MARKER_LENGTH, aruco_dict,
        )
        left_imgs, right_imgs = _generate_test_images(board_gen, image_dir, count=10)

    pairs = list(zip(left_imgs, right_imgs))
    logger.info(f"Found {len(pairs)} image pairs")

    # Build detector (OpenCV 4.13+ new API: CharucoDetector wraps everything)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS),
        SQUARE_LENGTH, MARKER_LENGTH, aruco_dict,
    )
    charuco_detector = cv2.aruco.CharucoDetector(board)

    all_obj_pts, all_img_pts1, all_img_pts2 = [], [], []
    img_size = None

    for left_path, right_path in pairs:
        left_img  = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        if left_img is None or right_img is None:
            logger.warning(f"  Could not read pair: "
                           f"{left_path.name} / {right_path.name}")
            continue

        gray1 = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = gray1.shape[::-1]        # (width, height)

        # detectBoard returns (charucoCorners, charucoIds, markerCorners, markerIds)
        cc1, ci1, _, _ = charuco_detector.detectBoard(gray1)
        cc2, ci2, _, _ = charuco_detector.detectBoard(gray2)

        if ci1 is not None and ci2 is not None and len(ci1) >= 4 and len(ci2) >= 4:
            ci1_flat = ci1.ravel()
            ci2_flat = ci2.ravel()

            # Build ID → corner lookup
            map1 = {int(cid): cc1[i, 0, :] for i, cid in enumerate(ci1_flat)}
            map2 = {int(cid): cc2[i, 0, :] for i, cid in enumerate(ci2_flat)}

            common_ids = sorted(set(map1.keys()) & set(map2.keys()))
            obj_pts  = [board.getChessboardCorners()[cid] for cid in common_ids]
            img_pts1 = [map1[cid] for cid in common_ids]
            img_pts2 = [map2[cid] for cid in common_ids]

            if len(obj_pts) >= 4:
                all_obj_pts.append(
                    np.array(obj_pts, dtype=np.float32).reshape(-1, 1, 3))
                all_img_pts1.append(
                    np.array(img_pts1, dtype=np.float32).reshape(-1, 1, 2))
                all_img_pts2.append(
                    np.array(img_pts2, dtype=np.float32).reshape(-1, 1, 2))
                logger.info(
                    f"  + Pair {left_path.name}/{right_path.name}: "
                    f"{len(obj_pts)} matched corners")
            else:
                logger.info(
                    f"  - Pair {left_path.name}/{right_path.name}: "
                    f"only {len(obj_pts)} common corners (<4)")
        else:
            n1 = len(ci1) if ci1 is not None else 0
            n2 = len(ci2) if ci2 is not None else 0
            logger.info(
                f"  - Pair {left_path.name}/{right_path.name}: "
                f"{n1}/{n2} Charuco corners (<4)")

    if len(all_obj_pts) < 3:
        logger.info(f"FAIL: Only {len(all_obj_pts)} valid pairs (need >=3)")
        logger.info("status: fail")
        logger.info("failure_reason: Insufficient calibration pairs")
        return {"status": "fail", "failure_reason": "Insufficient calibration pairs"}

    if img_size is None:
        logger.info("FAIL: No valid image pairs processed")
        logger.info("status: fail")
        logger.info("failure_reason: No valid image pairs")
        return {"status": "fail", "failure_reason": "No valid image pairs"}

    # Stereo calibration (fix intrinsics, only solve for extrinsics)
    logger.info(f"\nRunning stereo calibration with {len(all_obj_pts)} pairs...")
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, K1_out, D1_out, K2_out, D2_out, R, T, E, F = cv2.stereoCalibrate(
        all_obj_pts, all_img_pts1, all_img_pts2,
        K1, D1, K2, D2, img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=flags,
    )

    logger.info(f"\n=== Calibration Results ===")
    logger.info(f"Reprojection error: {ret:.4f}")
    logger.info(f"R (rotation matrix):\n{R}")
    logger.info(f"T (translation vector):\n{T}")
    logger.info(f"E (essential matrix):\n{E}")
    logger.info(f"F (fundamental matrix):\n{F}")

    # Build result dict
    result = {
        "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reprojection_error": float(ret),
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "K_left":  (K1_out.tolist() if isinstance(K1_out, np.ndarray) else K1.tolist()),
        "K_right": (K2_out.tolist() if isinstance(K2_out, np.ndarray) else K2.tolist()),
        "dist_left":  (D1_out.tolist() if isinstance(D1_out, np.ndarray) else D1.tolist()),
        "dist_right": (D2_out.tolist() if isinstance(D2_out, np.ndarray) else D2.tolist()),
    }

    output_path = Path(output_path)
    os.makedirs(str(output_path.parent), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\nExtrinsic params saved to {output_path}")
    logger.info(f"deployment_time_min: {(time.time() - calib_start) / 60:.2f}")
    logger.info("status: pass")

    print(f"\n  Calibration Complete!")
    print(f"  Translation: T = {T.flatten()}")
    print(f"  Rotation: R (3x3 matrix)")
    print(f"    [{R[0,0]:.4f} {R[0,1]:.4f} {R[0,2]:.4f}]")
    print(f"    [{R[1,0]:.4f} {R[1,1]:.4f} {R[1,2]:.4f}]")
    print(f"    [{R[2,0]:.4f} {R[2,1]:.4f} {R[2,2]:.4f}]")
    return {"status": "pass"}


def main():
    p = argparse.ArgumentParser(description="Task 4: Extrinsic Calibration")
    p.add_argument("--image-dir", default="data/calibration_images",
                   help="Directory with left/right calibration image pairs")
    p.add_argument("--intrinsic", default="configs/calibration.yaml",
                   help="Intrinsic parameters file (JSON or YAML)")
    p.add_argument("--output", default="configs/extrinsic_params.json",
                   help="Output path for extrinsic params")
    p.add_argument("--result-dir", default="results/")
    p.add_argument("--no-test-data", action="store_true",
                   help="Do not auto-generate test images when real ones are missing")
    args = p.parse_args()

    Path(args.image_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger("Extrinsic-Calibration",
                          os.path.join(args.result_dir, "logs"))
    calibrate_stereo(logger, args.image_dir, args.intrinsic, args.output)


if __name__ == "__main__":
    main()

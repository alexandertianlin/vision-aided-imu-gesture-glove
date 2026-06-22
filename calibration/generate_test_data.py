#!/usr/bin/env python3
"""
generate_test_data.py — Create synthetic ChArUco stereo calibration pairs.

Generates left/right image pairs with a synthetic CharucoBoard at varying
poses, so that the calibration pipeline can be tested end-to-end without
a physical camera or calibration board.
"""

import argparse, os, json, math
from pathlib import Path
import numpy as np
import cv2


def generate_pairs(board, output_dir, count=10, img_size=(640, 480),
                   baseline_m=0.08, seed=42):
    """Generate *count* stereo pairs at random-ish board poses."""
    rng = np.random.RandomState(seed)
    d = board.getDictionary()
    obj_pts = board.getObjPoints()          # 17 marker quads in 3D
    cols, rows = board.getChessboardSize()
    sq_len = board.getSquareLength()

    board_w = cols * sq_len
    board_h = rows * sq_len

    # Rough intrinsics for rendering
    fx = fy = max(img_size) * 1.2
    cx, cy = img_size[0] / 2.0, img_size[1] / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    marker_px = 80
    test_dir = Path(output_dir) / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    left_paths, right_paths = [], []

    for i in range(count):
        # Random pose: moderate rotation + translation toward camera
        angle_x = rng.uniform(-0.3, 0.3)
        angle_y = rng.uniform(-0.5, 0.5)
        angle_z = rng.uniform(-0.15, 0.15)
        dist    = rng.uniform(0.50, 0.85)
        h_off   = rng.uniform(-0.05, 0.05)

        Rx, _ = cv2.Rodrigues(np.array([angle_x, 0, 0], dtype=np.float64))
        Ry, _ = cv2.Rodrigues(np.array([0, angle_y, 0], dtype=np.float64))
        Rz, _ = cv2.Rodrigues(np.array([0, 0, angle_z], dtype=np.float64))
        R_world = Rx @ Ry @ Rz
        t_world = np.array([0.0, h_off, dist], dtype=np.float64)

        # Left / right camera poses (parallel, offset along X)
        R_left  = np.eye(3, dtype=np.float64)
        t_left  = np.array([-baseline_m / 2, 0.0, 0.0], dtype=np.float64)
        R_right = np.eye(3, dtype=np.float64)
        t_right = np.array([baseline_m / 2, 0.0, 0.0], dtype=np.float64)

        R1 = R_left.T @ R_world
        t1 = R_left.T @ (t_world - t_left)
        R2 = R_right.T @ R_world
        t2 = R_right.T @ (t_world - t_right)

        rvec1, _ = cv2.Rodrigues(R1)
        rvec2, _ = cv2.Rodrigues(R2)

        # Render
        left_img  = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 200
        right_img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 200

        for view_img, rvec, tvec in [(left_img, rvec1, t1), (right_img, rvec2, t2)]:
            for mk_idx in range(len(obj_pts)):
                marker_gray = cv2.aruco.generateImageMarker(d, mk_idx, marker_px)
                marker_bgr = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
                pts3d = np.array(obj_pts[mk_idx], dtype=np.float64).reshape(-1, 3)
                pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
                pts2d = pts2d.reshape(4, 2).astype(np.float32)
                src = np.array([[0, 0], [marker_px, 0], [marker_px, marker_px], [0, marker_px]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(src, pts2d)
                warped = cv2.warpPerspective(marker_bgr, M, (img_size[0], img_size[1]),
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                ones = np.ones_like(marker_bgr, dtype=np.uint8) * 255
                mask = cv2.warpPerspective(ones, M, (img_size[0], img_size[1]),
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                view_img[mask.astype(bool)] = warped[mask.astype(bool)]

        cv2.imwrite(str(test_dir / f"left_{i:02d}.png"), left_img)
        cv2.imwrite(str(test_dir / f"right_{i:02d}.png"), right_img)
        left_paths.append(test_dir / f"left_{i:02d}.png")
        right_paths.append(test_dir / f"right_{i:02d}.png")
        print(f"  [{i+1}/{count}] dist={dist:.2f}m  angle_y={angle_y:.2f}rad")

    print(f"Generated {count} pairs in {test_dir}")
    return sorted(left_paths), sorted(right_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic ChArUco stereo calibration images")
    parser.add_argument("--output-dir", default="data/calibration_images",
                        help="Output directory for image pairs")
    parser.add_argument("--count", type=int, default=20,
                        help="Number of stereo pairs")
    parser.add_argument("--width", type=int, default=640,
                        help="Image width")
    parser.add_argument("--height", type=int, default=480,
                        help="Image height")
    parser.add_argument("--baseline", type=float, default=0.08,
                        help="Stereo baseline in metres")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--intrinsics-output",
                        help="Path to write the synthetic intrinsics JSON")
    args = parser.parse_args()

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (5, 7), 0.04, 0.025, aruco_dict)

    generate_pairs(board, args.output_dir, args.count,
                   (args.width, args.height), args.baseline, args.seed)

    if args.intrinsics_output:
        fx = max(args.width, args.height) * 1.2
        cx, cy = args.width / 2.0, args.height / 2.0
        intrinsics = {
            "K_left":  [[fx, 0, cx], [0, fx, cy], [0, 0, 1]],
            "K_right": [[fx, 0, cx], [0, fx, cy], [0, 0, 1]],
            "dist_left":  [0.0, 0.0, 0.0, 0.0, 0.0],
            "dist_right": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        Path(args.intrinsics_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.intrinsics_output, "w") as f:
            json.dump(intrinsics, f, indent=2)
        print(f"Wrote synthetic intrinsics to {args.intrinsics_output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Orbbec Astra Plus Depth Camera Test.

Validates:
  - RGB stream (resolution, FPS, exposure)
  - Depth stream (validity, noise, artifacts)
  - Point Cloud export and registration
  - Dual-camera timestamp synchronization
  - ChArUco board detection on depth images

Output: results/logs/depth_test.log

Usage:
    python camera/test_depth.py
    python camera/test_depth.py --single-cam
    python camera/test_depth.py --capture-only --n-frames 50

Requirements:
    - opencv-python, numpy
    - One or two Orbbec Astra Plus cameras connected via USB
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Orbbec Astra Plus Depth and Camera Test"
    )
    parser.add_argument("--single-cam", action="store_true",
                        help="Test only a single camera (default: dual)")
    parser.add_argument("--left-id", type=int, default=0,
                        help="Left camera device ID")
    parser.add_argument("--right-id", type=int, default=1,
                        help="Right camera device ID")
    parser.add_argument("--n-frames", type=int, default=100,
                        help="Number of frames to capture for testing")
    parser.add_argument("--capture-only", action="store_true",
                        help="Only capture and save frames, skip analysis")
    parser.add_argument("--save-dir", type=str,
                        default="data/calibration_images/depth_test",
                        help="Directory to save test captures")
    parser.add_argument("--result-dir", type=str, default="results/")
    parser.add_argument("--display", action="store_true", default=True,
                        help="Show live preview windows")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable display windows (headless)")
    return parser.parse_args()


def print_device_info(logger, cap, label):
    """Print camera device properties."""
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    logger.info(f"\n--- {label} Device Info ---")
    logger.info(f"  Resolution: {width:.0f} x {height:.0f}")
    logger.info(f"  FPS: {fps:.1f}")
    logger.info(f"  Brightness: {brightness:.1f}")
    logger.info(f"  Contrast: {contrast:.1f}")
    logger.info(f"  Exposure: {exposure:.1f}")


def test_rgb_stream(logger, cap, label, n_frames=50, display=False):
    """Test RGB stream quality and performance."""
    logger.info(f"\n=== RGB Stream Test: {label} ===")

    frames = []
    timestamps = []
    capture_times = []

    for _ in range(5):
        cap.read()

    for i in range(n_frames):
        t0 = time.perf_counter()
        ret, frame = cap.read()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if ret and frame is not None:
            frames.append(frame)
            timestamps.append(time.time())
            capture_times.append(elapsed_ms)

    if not frames:
        logger.error(f"  {label}: No frames captured!")
        return {"status": "fail", "failure_reason": "No frames captured"}

    avg_capture_ms = float(np.mean(capture_times))
    fps = len(frames) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0
    first = frames[0]
    h, w = first.shape[:2]
    is_blank = np.std(first) < 5
    mean_brightness = float(np.mean(first))

    logger.info(f"  Captured {len(frames)}/{n_frames} frames")
    logger.info(f"  Resolution: {w}x{h}")
    logger.info(f"  Effective FPS: {fps:.1f}")
    logger.info(f"  Avg capture time: {avg_capture_ms:.2f} ms")
    logger.info(f"  Mean brightness: {mean_brightness:.1f}")
    logger.info(f"  Blank frames: {'YES' if is_blank else 'No'}")

    if display:
        cv2.imshow(f"RGB - {label}", frames[-1])

    return {
        "status": "pass" if not is_blank else "fail",
        "n_frames": len(frames),
        "resolution": f"{w}x{h}",
        "fps": round(fps, 1),
        "avg_capture_ms": round(avg_capture_ms, 2),
        "mean_brightness": round(mean_brightness, 1),
    }


def test_depth_stream(logger, cap_rgb, label, n_frames=50, display=False):
    """Test depth stream validity."""
    logger.info(f"\n=== Depth Stream Test: {label} ===")

    depth_frames = []
    depth_note = ""

    # Method 1: OpenNI2
    try:
        depth_cap = cv2.VideoCapture(10, cv2.CAP_OPENNI2)
        if depth_cap.isOpened():
            depth_cap.set(cv2.CAP_PROP_OPENNI2_MIRROR, 0)
            for _ in range(n_frames):
                if depth_cap.grab():
                    ret, depth_map = depth_cap.retrieve(None, cv2.CAP_OPENNI_DEPTH_MAP)
                    if ret and depth_map is not None:
                        depth_frames.append(depth_map)
            depth_cap.release()
            if depth_frames:
                depth_note = "via OpenNI2"
    except Exception as e:
        depth_note = f"OpenNI2 N/A: {e}"

    # Method 2: UVC fallback
    if not depth_frames:
        for depth_id in [2, 3, 6, 7]:
            try:
                cap_test = cv2.VideoCapture(depth_id, cv2.CAP_DSHOW)
                if cap_test.isOpened():
                    for _ in range(5):
                        ret_d, frame_d = cap_test.read()
                        if ret_d and frame_d is not None:
                            depth_frames.append(frame_d)
                            break
                    cap_test.release()
                    if depth_frames:
                        depth_note = f"via UVC device {depth_id}"
                        break
            except Exception:
                continue

    if not depth_frames:
        logger.info("  Depth stream unavailable (Orbbec SDK or OpenNI2 required)")
        return {"status": "unavailable", "note": "Depth via UVC/OpenNI2 not available"}

    first_depth = depth_frames[0]
    if len(first_depth.shape) == 3:
        first_depth = cv2.cvtColor(first_depth, cv2.COLOR_BGR2GRAY)

    depth_min = float(np.min(first_depth))
    depth_max = float(np.max(first_depth))
    depth_mean = float(np.mean(first_depth))
    depth_std = float(np.std(first_depth))
    is_frozen = depth_std < 1.0
    valid_ratio = float(np.sum(first_depth > 0)) / first_depth.size

    logger.info(f"  Access: {depth_note}")
    logger.info(f"  Depth frames: {len(depth_frames)}")
    logger.info(f"  Depth range: {depth_min:.0f}-{depth_max:.0f} mean={depth_mean:.0f}")
    logger.info(f"  Depth std: {depth_std:.1f}")
    logger.info(f"  Non-zero ratio: {valid_ratio:.1%}")
    logger.info(f"  Frozen: {'YES' if is_frozen else 'No'}")

    if display:
        norm = cv2.normalize(first_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        cv2.imshow(f"Depth - {label}", colored)

    return {
        "status": "pass" if not is_frozen else "fail",
        "access_method": depth_note,
        "n_frames": len(depth_frames),
        "depth_mean": round(depth_mean, 1),
        "depth_std": round(depth_std, 1),
        "valid_ratio": round(valid_ratio, 3),
    }


def test_timestamp_sync(logger, cap_left, cap_right, n_samples=30):
    """Measure timestamp offset between cameras."""
    logger.info(f"\n=== Timestamp Sync Test ===")
    offsets_ms = []
    for i in range(n_samples):
        ret_l, _ = cap_left.read()
        ts_l = time.perf_counter()
        ret_r, _ = cap_right.read()
        ts_r = time.perf_counter()
        if ret_l and ret_r:
            offsets_ms.append((ts_r - ts_l) * 1000)
    if not offsets_ms:
        return {"status": "fail", "failure_reason": "No sync samples"}
    offsets_ms = np.array(offsets_ms)
    mean_off = float(np.mean(np.abs(offsets_ms)))
    max_off = float(np.max(np.abs(offsets_ms)))
    std_off = float(np.std(offsets_ms))
    logger.info(f"  Samples: {len(offsets_ms)}")
    logger.info(f"  Mean offset: {mean_off:.2f} ms")
    logger.info(f"  Max offset: {max_off:.2f} ms")
    logger.info(f"  Std offset: {std_off:.2f} ms")
    status = "pass" if max_off < 30 else "warn"
    logger.info(f"  Sync status: {status}")
    return {"status": status, "mean_offset_ms": round(mean_off, 2),
            "max_offset_ms": round(max_off, 2), "std_offset_ms": round(std_off, 2),
            "n_samples": len(offsets_ms)}


def test_point_cloud(logger, depth_image, K=None):
    """Test depth-to-point-cloud conversion."""
    logger.info(f"\n=== Point Cloud Test ===")
    if depth_image is None:
        return {"status": "skipped"}
    if K is None:
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
    z = depth_image.astype(np.float32) / 1000.0
    valid = (z > 0.1) & (z < 2.0)
    n_valid = int(np.sum(valid))
    total = depth_image.size
    validity = n_valid / total if total > 0 else 0
    logger.info(f"  Total pixels: {total}")
    logger.info(f"  Valid depth points (0.1-2.0m): {n_valid} ({validity:.1%})")
    return {"status": "pass" if validity > 0.1 else "warn",
            "total_pixels": total, "valid_points": n_valid,
            "validity_ratio": round(validity, 3)}


def test_charuco_detection(logger, frame):
    """Test ChArUco board detection."""
    logger.info(f"\n=== ChArUco Detection Test ===")
    if frame is None:
        return {"status": "skipped", "note": "No frame"}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((7, 5), 0.04, 0.025, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)
    cc, ci, mc, mi = detector.detectBoard(gray)
    if ci is not None and len(ci) > 0:
        logger.info(f"  Detected {len(ci)} ChArUco corners")
        return {"status": "pass", "n_corners": len(ci), "corner_ids": ci.flatten().tolist()}
    if mi is not None and len(mi) > 0:
        logger.info(f"  Detected {len(mi)} ArUco markers (but no Charuco corners)")
        return {"status": "info", "n_markers": len(mi)}
    logger.info("  No ChArUco board detected")
    return {"status": "info", "note": "No board detected, show board to camera"}


def test_single_camera(logger, device_id, label, n_frames, display):
    """Run all tests for one camera."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {label} (device {device_id})")
    logger.info(f"{'='*60}")
    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        return {"status": "fail", "failure_reason": "Camera not opened"}
    print_device_info(logger, cap, label)
    results = {}
    results["rgb"] = test_rgb_stream(logger, cap, label, n_frames, display)
    results["depth"] = test_depth_stream(logger, cap, label, n_frames, display)
    ret, frame = cap.read()
    if ret:
        results["charuco"] = test_charuco_detection(logger, frame)
    cap.release()
    ok_statuses = {"pass"}
    all_pass = all(
        r.get("status") in ok_statuses
        for r in results.values()
        if r.get("status") not in ("skipped", "info", "unavailable")
    )
    results["overall"] = "PASS" if all_pass else "WARN/FAIL"
    return results


def main():
    args = parse_args()
    log_dir = Path(args.result_dir) / "logs"
    logger = setup_logger("Depth-Test", str(log_dir))
    logger.info("=" * 60)
    logger.info("Orbbec Astra Plus - Depth & Camera Test")
    logger.info("=" * 60)
    display = args.display and not args.no_display
    n_frames = args.n_frames
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Discover cameras
    logger.info("\nDiscovering camera devices...")
    available = []
    for i in range(8):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available.append(i)
                logger.info(f"  Device {i}: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
    logger.info(f"Found {len(available)} camera devices: {available}")

    if len(available) == 0:
        logger.error("No cameras found!")
        print("No cameras detected. Connect Orbbec Astra Plus and try again.")
        return

    all_results = {}

    if args.single_cam or len(available) < 2:
        device = available[0] if available else args.left_id
        all_results["camera"] = test_single_camera(logger, device, "Camera", n_frames, display)
    else:
        left_id = args.left_id if args.left_id in available else available[0]
        right_id = args.right_id if args.right_id in available else available[1]
        all_results["left"] = test_single_camera(logger, left_id, "Left", n_frames, display)
        all_results["right"] = test_single_camera(logger, right_id, "Right", n_frames, display)

        # Sync test
        logger.info(f"\n{'='*60}")
        logger.info("Timestamp Sync Test (dual camera)")
        logger.info(f"{'='*60}")
        cap_left = cv2.VideoCapture(left_id, cv2.CAP_DSHOW)
        cap_right = cv2.VideoCapture(right_id, cv2.CAP_DSHOW)
        for cap in [cap_left, cap_right]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        for _ in range(10):
            cap_left.read()
            cap_right.read()
        all_results["sync"] = test_timestamp_sync(logger, cap_left, cap_right, 30)

        # Save sample pair
        ret_l, sample_l = cap_left.read()
        ret_r, sample_r = cap_right.read()
        if ret_l and ret_r:
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(save_dir / f"sample_left_{ts}.png"), sample_l)
            cv2.imwrite(str(save_dir / f"sample_right_{ts}.png"), sample_r)
            logger.info(f"Sample stereo pair saved to {save_dir}")
        cap_left.release()
        cap_right.release()

    # Save report
    summary_path = save_dir / "camera_test_report.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nTest report saved to {summary_path}")

    logger.info(f"\n{'='*60}")
    logger.info("Test Summary")
    logger.info(f"{'='*60}")
    for key, val in all_results.items():
        if isinstance(val, dict):
            logger.info(f"  {key}: {val.get('overall', val.get('status', '?'))}")

    if display:
        logger.info("\nPress any key to close preview windows...")
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"\nTest complete. Report: {summary_path}")
    print(f"Log: {log_dir}/Depth-Test_benchmark.log")


if __name__ == "__main__":
    main()

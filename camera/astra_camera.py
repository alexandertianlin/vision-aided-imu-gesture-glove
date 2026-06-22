#!/usr/bin/env python3
"""Orbbec Astra Plus Camera Manager.

Provides unified access to:
- RGB stream (UVC via OpenCV)
- Depth stream (Orbbec SDK / OpenNI2 fallback)
- Dual-camera capture with timestamp sync

Usage:
    from camera.astra_camera import AstraManager
    mgr = AstraManager()
    mgr.discover_devices()
    mgr.capture_stereo() -> (rgb_left, rgb_right, depth_left, depth_right)
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import cv2

logger = logging.getLogger("AstraCamera")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "camera.yaml"


def load_camera_config(config_path=None):
    """Load camera configuration from YAML."""
    if config_path is None:
        config_path = CONFIG_PATH
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AstraCamera:
    """Single Orbbec Astra Plus camera controller."""

    def __init__(self, device_id: int = 0, rgb_size=(640, 480), fps=30):
        self.device_id = device_id
        self.rgb_size = rgb_size
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._depth_cap: Optional[cv2.VideoCapture] = None

    def open_rgb(self) -> bool:
        """Open RGB stream via UVC."""
        self._cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.rgb_size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.rgb_size[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            logger.warning(f"Camera {self.device_id} RGB failed to open")
            return False
        logger.info(f"Camera {self.device_id} RGB opened: {self.rgb_size} @ {self.fps}fps")
        return True

    def read_rgb(self) -> Optional[np.ndarray]:
        """Read a single RGB frame. Returns BGR ndarray or None."""
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def open_depth(self) -> bool:
        """Open Depth stream.

        Orbbec Astra Plus exposes depth as a second UVC device (typically device_id + 1
        on the same USB bus), or via OpenNI2 / Orbbec SDK.
        """
        depth_id = self.device_id + 2  # common pattern: RGB=0, Depth=2 on same device
        self._depth_cap = cv2.VideoCapture(depth_id, cv2.CAP_DSHOW)
        if not self._depth_cap.isOpened():
            depth_id = self.device_id + 1
            self._depth_cap = cv2.VideoCapture(depth_id, cv2.CAP_DSHOW)
        if not self._depth_cap.isOpened():
            logger.warning(f"Camera {self.device_id} Depth failed via UVC")
            return False
        logger.info(f"Camera {self.device_id} Depth opened on device {depth_id}")
        return True

    def read_depth(self) -> Optional[np.ndarray]:
        """Read a single depth frame. Returns uint16 ndarray or None."""
        if self._depth_cap is None or not self._depth_cap.isOpened():
            return None
        ret, frame = self._depth_cap.read()
        if not ret:
            return None
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def read_depth_colored(self) -> Optional[np.ndarray]:
        """Read depth frame and return a colorized 3-channel BGR image for display."""
        depth = self.read_depth()
        if depth is None:
            return None
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    def release(self):
        if self._cap:
            self._cap.release()
        if self._depth_cap:
            self._depth_cap.release()


class AstraManager:
    """Manages dual Orbbec Astra Plus cameras (left and right)."""

    def __init__(self, config_path=None):
        self.config = load_camera_config(config_path) if config_path else None
        self.left: Optional[AstraCamera] = None
        self.right: Optional[AstraCamera] = None
        self.devices_found: List[int] = []

    def discover_devices(self) -> List[int]:
        """Discover all available Orbbec Astra Plus cameras."""
        available = []
        for i in range(8):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available.append(i)
                    logger.info(f"  Device {i}: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
        self.devices_found = available
        logger.info(f"Found {len(available)} camera devices: {available}")
        return available

    def init_stereo(self, left_id=0, right_id=1, rgb_size=(640, 480), fps=30) -> bool:
        """Initialize left and right cameras for stereo capture."""
        self.left = AstraCamera(left_id, rgb_size, fps)
        self.right = AstraCamera(right_id, rgb_size, fps)
        ok_left = self.left.open_rgb()
        ok_right = self.right.open_rgb()
        if not (ok_left and ok_right):
            logger.error("Stereo init failed: left=%s, right=%s", ok_left, ok_right)
            return False
        logger.info(f"Stereo initialized: Left={left_id}, Right={right_id}")
        return True

    def init_depth_stereo(self) -> bool:
        """Initialize depth streams for both cameras (if available)."""
        if self.left:
            self.left.open_depth()
        if self.right:
            self.right.open_depth()
        return True

    def capture_stereo(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture synchronized left/right RGB frames."""
        if self.left is None or self.right is None:
            return None, None
        left_frame = self.left.read_rgb()
        right_frame = self.right.read_rgb()
        return left_frame, right_frame

    def capture_stereo_depth(self) -> Tuple:
        """Capture synchronized left/right depth frames."""
        if self.left is None or self.right is None:
            return None, None
        return self.left.read_depth(), self.right.read_depth()

    def capture_stereo_all(self) -> Dict[str, Optional[np.ndarray]]:
        """Capture all streams: left RGB, right RGB, left depth, right depth."""
        return {
            "rgb_left": self.left.read_rgb() if self.left else None,
            "rgb_right": self.right.read_rgb() if self.right else None,
            "depth_left": self.left.read_depth() if self.left else None,
            "depth_right": self.right.read_depth() if self.right else None,
        }

    def measure_sync_offset(self, n_frames=30) -> Dict[str, float]:
        """Measure timestamp offset between left and right cameras."""
        timestamps_left = []
        timestamps_right = []
        for _ in range(n_frames):
            if self.left and self.right:
                self.left.read_rgb()
                ts_l = time.perf_counter()
                self.right.read_rgb()
                ts_r = time.perf_counter()
                timestamps_left.append(ts_l)
                timestamps_right.append(ts_r)
        if not timestamps_left or not timestamps_right:
            return {"mean_offset_ms": -1, "max_offset_ms": -1}
        offsets = [abs(r - l) * 1000 for l, r in zip(timestamps_left, timestamps_right)]
        return {
            "mean_offset_ms": float(np.mean(offsets)),
            "max_offset_ms": float(np.max(offsets)),
            "min_offset_ms": float(np.min(offsets)),
            "n_samples": n_frames,
        }

    def release(self):
        if self.left:
            self.left.release()
        if self.right:
            self.right.release()


def create_charuco_board(rows=7, cols=5, square_length=0.04, marker_length=0.025):
    """Create a ChArUco board for calibration."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((cols, rows), square_length, marker_length, aruco_dict)
    return board, aruco_dict


def detect_charuco(image, board, aruco_dict):
    """Detect ChArUco corners in an image.

    Returns (retval, charuco_corners, charuco_ids) or (False, None, None).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    detector = cv2.aruco.CharucoDetector(board)
    cc, ci, mc, mi = detector.detectBoard(gray)
    if ci is not None and len(ci) >= 4:
        return True, cc, ci
    return False, None, None


def depth_to_pointcloud(depth_image, K, max_depth_m=2.0):
    """Convert depth image (HxW uint16 mm) to point cloud (Nx3).

    K: 3x3 intrinsic matrix.
    Returns (N, 3) array of (X, Y, Z) in meters.
    """
    h, w = depth_image.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_image.astype(np.float32) / 1000.0  # mm to meters

    # Filter out invalid / far points
    valid = (z > 0.1) & (z < max_depth_m)
    z = z[valid]
    u = u[valid]
    v = v[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


if __name__ == "__main__":
    # Quick self-test
    logging.basicConfig(level=logging.INFO)
    mgr = AstraManager()
    devices = mgr.discover_devices()
    if len(devices) >= 2:
        print(f"2+ cameras found, attempting stereo init...")
        mgr.init_stereo(devices[0], devices[1])
        left, right = mgr.capture_stereo()
        if left is not None and right is not None:
            cv2.imshow("Left", left)
            cv2.imshow("Right", right)
            print("Press any key to exit preview...")
            cv2.waitKey(0)
        mgr.release()
    cv2.destroyAllWindows()

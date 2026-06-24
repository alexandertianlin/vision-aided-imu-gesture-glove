# ViTPose ONNX Real-time Hand Detection — CUDA GPU Accelerated
# Usage: python realtime_vitpose_cuda.py [--model MODEL_PATH] [--cam CAM_ID]

import numpy as np
import cv2
import time
import os
import sys
import argparse
import onnxruntime as ort

# --- Config ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "vitpose")
DEFAULT_ONNX = os.path.join(MODEL_DIR, "vitpose_hand.onnx")
IMG_SIZE = (256, 256)
NUM_KP = 21
UDP_IP = "127.0.0.1"
UDP_PORT = 5055

# Hand skeleton connections (MediaPipe 21-keypoint convention)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
]


def parse_args():
    parser = argparse.ArgumentParser(description="ViTPose ONNX Hand Detection (CUDA)")
    parser.add_argument("--model", default=DEFAULT_ONNX, help="Path to ONNX model")
    parser.add_argument("--cam", type=int, default=0, help="Camera device ID")
    parser.add_argument("--cpu", action="store_true", help="Force CPU provider")
    parser.add_argument("--udp", action="store_true", help="Enable UDP output")
    parser.add_argument("--gpu-mem", type=float, default=4.0, help="GPU memory limit (GB)")
    return parser.parse_args()


def init_session(onnx_path, use_cpu=False, gpu_mem_gb=4.0):
    """Initialize ONNX Runtime session with CUDA provider."""
    if use_cpu:
        providers = ["CPUExecutionProvider"]
        print("Using CPU provider")
    else:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": int(gpu_mem_gb * 1024 * 1024 * 1024),
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                    },
                ),
                "CPUExecutionProvider",
            ]
            print(f"Using CUDA provider (GPU mem limit: {gpu_mem_gb}GB)")
        else:
            print("CUDAExecutionProvider not available, falling back to CPU")
            providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"Input:  {session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
    print(f"Output: {session.get_outputs()[0].name} {session.get_outputs()[0].shape}")
    return session


def preprocess(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    tensor = norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    return tensor


def decode_keypoints(heatmaps, frame_h, frame_w):
    hm = heatmaps[0]
    hm_h, hm_w = hm.shape[1], hm.shape[2]
    keypoints = []
    for j in range(NUM_KP):
        y_idx, x_idx = np.unravel_index(np.argmax(hm[j]), hm[j].shape)
        px = int(x_idx * frame_w / hm_w)
        py = int(y_idx * frame_h / hm_h)
        conf = float(hm[j, y_idx, x_idx])
        keypoints.append((px, py, conf))
    return keypoints


def draw_skeleton(frame, keypoints, connections):
    for j, (px, py, conf) in enumerate(keypoints):
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)
        cv2.circle(frame, (px, py), 5, (0, 180, 0), 1)
    for a, b in connections:
        if a < len(keypoints) and b < len(keypoints):
            x1, y1, _ = keypoints[a]
            x2, y2, _ = keypoints[b]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 100), 1)


def build_udp_message(keypoints, timestamp_ms):
    import json
    msg = {
        "timestampMs": timestamp_ms,
        "type": "VITPOSE_KEYPOINTS",
        "num_keypoints": NUM_KP,
        "keypoints": [
            {"x": kp[0], "y": kp[1], "confidence": kp[2]} for kp in keypoints
        ],
    }
    return json.dumps(msg)


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        sys.exit(1)

    session = init_session(args.model, args.cpu, args.gpu_mem)
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.cam}")
        sys.exit(1)

    udp_sock = None
    if args.udp:
        import socket
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP output: {UDP_IP}:{UDP_PORT}")

    fps_buf = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        tensor = preprocess(frame)
        t0 = time.perf_counter()
        heatmaps = session.run(None, {input_name: tensor})[0]
        lat_ms = (time.perf_counter() - t0) * 1000

        fps_buf.append(lat_ms)
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        avg_fps = 1000.0 / (sum(fps_buf) / len(fps_buf)) if fps_buf else 0

        keypoints = decode_keypoints(heatmaps, h, w)
        draw_skeleton(frame, keypoints, CONNECTIONS)

        provider = "CUDA" if "CUDAExecutionProvider" in session.get_providers() else "CPU"
        cv2.putText(frame, f"FPS: {avg_fps:.0f} | {lat_ms:.0f}ms | {provider}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "ViTPose ONNX + RTX 4060",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        if udp_sock:
            msg = build_udp_message(keypoints, int(time.time() * 1000))
            udp_sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))

        cv2.imshow("ViTPose Hand Detection (CUDA)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()

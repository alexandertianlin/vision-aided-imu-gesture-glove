#!/usr/bin/env python3
"""Astra Pro RGB + Depth live capture via OpenNI2 (primesense)."""
import json, os, sys, time
from pathlib import Path
import cv2, numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["OPENNI2_REDIST64"] = os.path.join(PROJECT, "OpenNI2", "Drivers")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(PROJECT)

from openni import openni2
openni2.initialize([PROJECT])

def load_intrinsics():
    p = Path("configs/astra_pro_intrinsics.json")
    if p.exists():
        d = json.load(open(p))
        K = np.array(d["camera_matrix"], dtype=np.float64)
        D = np.array(d.get("distortion_coefficients", [0]*5), dtype=np.float64)
    else:
        K = np.array([[975.6, 0, 636.7], [0, 973.7, 366.7], [0, 0, 1]], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)
    print(f"Intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
    return K, D

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    K, D = load_intrinsics()
    dev = openni2.Device.open_any()
    print(f"Device: {dev.device_info.name.decode()}\n")

    ds = dev.create_depth_stream()
    cs = dev.create_color_stream()

    dm = cm = None
    for m in ds.get_sensor_info().videoModes:
        if m.resolutionX == 640 and m.resolutionY == 480 and "DEPTH_1_MM" in str(m.pixelFormat):
            dm = m; break
    for m in cs.get_sensor_info().videoModes:
        if m.resolutionX == 1280 and m.resolutionY == 720:
            cm = m; break
    if dm is None: dm = ds.get_sensor_info().videoModes[4]
    if cm is None: cm = cs.get_sensor_info().videoModes[0]

    print(f"Depth: {dm.resolutionX}x{dm.resolutionY}@{dm.fps}fps")
    print(f"Color: {cm.resolutionX}x{cm.resolutionY}@{cm.fps}fps\n")
    print("[s/S/SPACE] save  [q/Q] quit\n")

    ds.set_video_mode(dm); cs.set_video_mode(cm)
    ds.start(); cs.start()

    save_dir = Path("data/calibration_images/depth_capture")
    save_dir.mkdir(parents=True, exist_ok=True)
    show = not args.no_show
    saved = fc = 0
    t0 = time.time()

    while True:
        df = ds.read_frame(); cf = cs.read_frame()
        if df is None or cf is None: time.sleep(0.01); continue

        depth = np.frombuffer(df.get_buffer_as_uint16(), dtype=np.uint16).reshape(df.height, df.width)
        color = np.frombuffer(cf.get_buffer_as_uint8(), dtype=np.uint8).reshape(cf.height, cf.width, 3)
        bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        fc += 1

        dvis = cv2.normalize(np.clip(depth.astype(np.float32), 0, 2000), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dvis = cv2.applyColorMap(dvis, cv2.COLORMAP_JET)
        vn = int((depth > 0).sum())
        cv2.putText(bgr, f"Valid: {vn} ({vn/depth.size*100:.0f}%)", (10, bgr.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        if show and fc == 1:
            cv2.namedWindow("RGB (1280x720)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("RGB (1280x720)", 1280, 720)
            cv2.namedWindow("Depth (640x480)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Depth (640x480)", 640, 480)
        if show:
            cv2.imshow("RGB (1280x720)", bgr)
            cv2.imshow("Depth (640x480)", dvis)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = -1

        if key in [ord("q"), ord("Q")]: break
        if key in [ord("s"), ord("S"), ord(" ")]:
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(save_dir / f"cap_{ts}_rgb.png"), bgr)
            cv2.imwrite(str(save_dir / f"cap_{ts}_depth.png"), depth)
            valid = (depth.astype(np.float32)/1000 > 0.1) & (depth.astype(np.float32)/1000 < 3.0)
            n = int(valid.sum())
            print(f"Saved cap_{ts} | Depth:{depth.min()}-{depth.max()}mm | Points:{n}({n/valid.size*100:.1f}%)")
            saved += 1

    ds.stop(); cs.stop()
    openni2.unload(); cv2.destroyAllWindows()
    if fc: print(f"\nDone. Frames:{fc} fps:{fc/(time.time()-t0):.1f} Saved:{saved}")

if __name__ == "__main__":
    main()

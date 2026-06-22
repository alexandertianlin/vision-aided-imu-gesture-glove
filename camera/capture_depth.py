"""Astra Pro Depth + RGB Capture - FINAL WORKING VERSION.
Depth via OpenNI2, RGB via DSHOW. Sequential reads, short paths.
"""
import os, time, cv2, numpy as np

# Setup
os.chdir(r"C:\Users\tianl\Documents\Codex\2026-06-16\alexandertianlin-vision-imu-gesture-glove-4ab6097")
os.environ["OPENNI2_REDIST64"] = os.path.join(os.getcwd(), "OpenNI2", "Drivers")
if hasattr(os, "add_dll_directory"): os.add_dll_directory(os.getcwd())

from openni import openni2
openni2.initialize([os.getcwd()])
dev = openni2.Device.open_any()

# Setup depth stream
ds = dev.create_depth_stream()
for m in ds.get_sensor_info().videoModes:
    if m.resolutionX == 640 and m.resolutionY == 480 and "DEPTH_1_MM" in str(m.pixelFormat):
        ds.set_video_mode(m); break
ds.start()

# Setup RGB camera
rgb_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(10): rgb_cap.read()

save_dir = "work/depth_capture"
os.makedirs(save_dir, exist_ok=True)

print("Astra Pro Depth + RGB Capture")
print("Depth: OpenNI2 640x480 | RGB: DSHOW 640x480\n")
print("[ENTER] capture  [q] quit\n")

saved = 0
while True:
    cmd = input().strip()
    if cmd.lower() in ("q", "quit", "exit"): break
    if cmd == "":
        # Capture depth
        df = ds.read_frame()
        depth = np.frombuffer(df.get_buffer_as_uint16(),
                              dtype=np.uint16).reshape(df.height, df.width).copy()
        # Capture RGB
        _, rgb = rgb_cap.read()

        ts = time.strftime("%Y%m%d_%H%M%S")
        dp_raw = os.path.join(save_dir, f"cap_{ts}_depth.raw")
        rp = os.path.join(save_dir, f"cap_{ts}_rgb.png")
        dp_vis = os.path.join(save_dir, f"cap_{ts}_depth_vis.png")

        depth.tofile(dp_raw)
        cv2.imwrite(rp, rgb)
        dvis = cv2.normalize(np.clip(depth.astype(np.float32), 0, 2000),
                             None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(dp_vis, cv2.applyColorMap(dvis, cv2.COLORMAP_JET))

        vn = int((depth > 0).sum())
        saved += 1
        print(f"  #{saved} cap_{ts} | Depth:{depth.min()}-{depth.max()}mm Valid:{vn}({vn/depth.size*100:.0f}%)")
        os.startfile(rp)

ds.stop(); rgb_cap.release(); openni2.unload()
print(f"\nDone. {saved} pairs saved to {save_dir}")

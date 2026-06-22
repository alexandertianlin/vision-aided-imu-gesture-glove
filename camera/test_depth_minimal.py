"""Astra Pro depth test - minimal version, save frames to disk."""
import os, sys, time
import numpy as np

PROJECT = r"C:\Users\tianl\Documents\Codex\2026-06-16\alexandertianlin-vision-imu-gesture-glove-4ab6097"
os.environ["OPENNI2_REDIST64"] = os.path.join(PROJECT, "OpenNI2", "Drivers")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(PROJECT)

from openni import openni2
openni2.initialize([PROJECT])

dev = openni2.Device.open_any()
print(f"Device: {dev.device_info.name.decode()}")

ds = dev.create_depth_stream()
cs = dev.create_color_stream()

for m in ds.get_sensor_info().videoModes:
    if m.resolutionX == 640 and m.resolutionY == 480 and "DEPTH_1_MM" in str(m.pixelFormat):
        ds.set_video_mode(m); break

for m in cs.get_sensor_info().videoModes:
    if m.resolutionX == 1280 and m.resolutionY == 720:
        cs.set_video_mode(m); break

ds.start(); cs.start()
print("Streams started. Capturing 3 frames...")

save_dir = os.path.join(PROJECT, "data", "calibration_images", "depth_test")
os.makedirs(save_dir, exist_ok=True)

for i in range(30):
    df = ds.read_frame()
    cf = cs.read_frame()
    if df is None or cf is None:
        time.sleep(0.01)
        continue

    depth = np.frombuffer(df.get_buffer_as_uint16(), dtype=np.uint16).reshape(df.height, df.width)
    color = np.frombuffer(cf.get_buffer_as_uint8(), dtype=np.uint8).reshape(cf.height, cf.width, 3)

    if i < 3:
        # Save as raw numpy for analysis
        np.save(os.path.join(save_dir, f"depth_{i}.npy"), depth)
        # Save color as PPM (no OpenCV dependency at all)
        ppm_path = os.path.join(save_dir, f"color_{i}.ppm")
        with open(ppm_path, "wb") as f:
            f.write(f"P6\n{cf.width} {cf.height}\n255\n".encode())
            f.write(color.tobytes())

    v = (depth > 0).sum()
    print(f"  Frame {i}: depth={depth.shape} [{depth.min()}-{depth.max()}mm] valid={v} ({v/depth.size*100:.1f}%)  color={color.shape}")

    if i >= 2:
        break

ds.stop(); cs.stop()
openni2.unload()

print(f"\nDone! Saved to {save_dir}")
print(f"  depth_0.npy, depth_1.npy, depth_2.npy")
print(f"  color_0.ppm, color_1.ppm, color_2.ppm")

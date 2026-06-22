"""Serial IMU Reader - Multi-protocol parser for sensor glove hardware

Protocol: 0xB5 0xA5 0x55 header, 35-byte packets
  - Byte 6: device_id (0x30=Palm, 0x1E=Thumb, 0x28=Index, 0x32=Middle, 0x3C=Ring, 0x46=Little)
  - Bytes 8-15: int16 quaternion (w,x,y,z) scaled by 1/10000
  - Bytes 22-33: float32 force vector (for fingers, not palm)
  - Baud: 460800

Adapted from:
  C:\\Users\\tianl\\Documents/Codex/sensors/orbbec/work/serial_imu_reader.py
  C:\\Users\\tianl\\Documents/Codex/sensors/orbbec/work/orbbec_viewer_imu.py
"""

import serial
import struct
import time
import math
import argparse

FRAME_HEADER = b"\xb5\xa5\x55"
QUAT_SCALE = 10000.0

FINGER_NAMES = {
    0x30: "Palm", 0x1E: "Thumb", 0x28: "Index",
    0x32: "Middle", 0x3C: "Ring", 0x46: "Little",
}


class SerialIMUReader:
    """Read and parse sensor glove IMU packets over USB serial."""

    def __init__(self, port="COM3", baud=460800):
        self.port = port
        self.baud = baud
        self._ser = None
        self._buf = b""
        self._packet_count = 0
        self._error = ""

    def start(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self._buf = b""
            self._packet_count = 0
            time.sleep(0.2)
            self._ser.reset_input_buffer()
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def read_packet(self):
        if not self._ser or not self._ser.is_open:
            return None
        try:
            chunk = self._ser.read(self._ser.in_waiting or 1)
            if chunk:
                self._buf += chunk
            while len(self._buf) >= 35:
                idx = self._buf.find(FRAME_HEADER)
                if idx < 0:
                    self._buf = self._buf[-3:]
                    return None
                if idx > 0:
                    self._buf = self._buf[idx:]
                if len(self._buf) < 35:
                    return None
                pkt = self._buf[:35]
                self._buf = self._buf[35:]
                result = self._parse(pkt, time.time())
                if result:
                    self._packet_count += 1
                    result["frame"] = self._packet_count
                    return result
            return None
        except Exception as e:
            self._error = str(e)
            return None

    def _parse(self, data, tstamp):
        did = data[6]
        try:
            w, x, y, z = struct.unpack("<hhhh", data[8:16])
        except:
            return None
        qw, qx, qy, qz = w / QUAT_SCALE, x / QUAT_SCALE, y / QUAT_SCALE, z / QUAT_SCALE
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm > 0.001:
            qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        else:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        force = None
        if did != 0x30:
            try:
                fx, fy, fz = struct.unpack("<fff", data[22:34])
                force = (fx, fy, fz)
            except:
                pass
        return {
            "type": "quaternion", "tstamp": tstamp,
            "device_id": did, "finger": FINGER_NAMES.get(did, f"0x{did:02X}"),
            "quaternion": (qw, qx, qy, qz), "force": force,
        }

    def close(self):
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except:
                pass

    @property
    def error_message(self):
        return self._error


def monitor_main():
    parser = argparse.ArgumentParser(description="Monitor sensor glove IMU stream")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--csv", help="Save to CSV file")
    args = parser.parse_args()

    reader = SerialIMUReader(args.port, args.baud)
    if not reader.start():
        print(f"[FAIL] {args.port} @ {args.baud}: {reader.error_message}")
        return 1

    print(f"[OK] Serial IMU monitor: {args.port} @ {args.baud}")
    print("  Palm=0x30 Thumb=0x1E Index=0x28 Middle=0x32 Ring=0x3C Little=0x46")
    print("  Ctrl+C to stop\n")

    csv_f = None
    csv_w = None
    if args.csv:
        import csv
        csv_f = open(args.csv, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["time", "did", "finger", "qw", "qx", "qy", "qz", "fx", "fy", "fz"])

    stats = {}
    t0 = time.time()
    try:
        while True:
            pkt = reader.read_packet()
            if pkt and pkt["type"] == "quaternion":
                now = time.time()
                did = pkt["device_id"]
                name = pkt["finger"]
                if did not in stats:
                    stats[did] = 0
                stats[did] += 1
                q = pkt["quaternion"]
                f = pkt["force"]
                f_str = f"F({f[0]:.2f},{f[1]:.2f},{f[2]:.2f})" if f else "F(n/a)"
                print(f"T+{now-t0:6.1f}s [{name:>12}] Q({q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f}) {f_str}")
                if csv_w:
                    csv_w.writerow([f"{now:.6f}", f"0x{did:02X}", name, f"{q[0]:.6f}", f"{q[1]:.6f}", f"{q[2]:.6f}", f"{q[3]:.6f}", *(f or (0,0,0))])
    except KeyboardInterrupt:
        dur = time.time() - t0
        print(f"\n=== {reader.packet_count} packets in {dur:.1f}s ===")
        for did, cnt in sorted(stats.items()):
            name = FINGER_NAMES.get(did, f"0x{did:02X}")
            print(f"  {name:>12} (0x{did:02X}): {cnt} @ {cnt/dur:.1f} Hz")
    finally:
        reader.close()
        if csv_f:
            csv_f.close()
    return 0


if __name__ == "__main__":
    exit(monitor_main())

#!/usr/bin/env python3
"""
Main Agent Monitor ? Task Watch
Checks all 4 task statuses every N seconds.
Usage: python scripts/task_watch.py [--interval 60]
"""

import argparse
import time
import os
import json
from pathlib import Path
from datetime import datetime
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.utils import write_benchmark_csv

TASKS = [
    {
        "id": 1,
        "name": "AWR-Net + YOLO-Hand-Pose",
        "instance": "instance-...-123731",
        "ip": "10.128.0.8",
        "models": ["AWR-Net", "YOLO-Hand-Pose"],
        "gpu": "L4",
        "status": "pending",
        "current_model": None,
        "log_files": [],
    },
    {
        "id": 2,
        "name": "RTMPose-Hand + MMPose-Hand",
        "instance": "instance-...-014510",
        "ip": "10.128.0.9",
        "models": ["RTMPose-Hand", "MMPose-Hand"],
        "gpu": "L4",
        "status": "pending",
        "current_model": None,
        "log_files": [],
    },
    {
        "id": 3,
        "name": "IPNet + HandPointNet",
        "instance": "instance-...-015628",
        "ip": "10.128.0.11",
        "models": ["IPNet", "HandPointNet"],
        "gpu": "L4",
        "status": "pending",
        "current_model": None,
        "log_files": [],
    },
    {
        "id": 4,
        "name": "Extrinsic Calibration",
        "instance": "local",
        "ip": "local",
        "models": ["Calibration"],
        "gpu": "CPU",
        "status": "pending",
        "current_model": None,
        "log_files": [],
    },
]


def check_local_logs(log_dir):
    """Check local log directory for completed benchmark logs."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []

    logs = []
    for f in sorted(log_dir.glob("*.log")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        status = "unknown"
        if "status: pass" in text.lower():
            status = "pass"
        elif "status: fail" in text.lower():
            status = "fail"

        # Extract key metrics
        metrics = {}
        for line in text.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if k in ["detect_rate","fps","latency_ms","stability",
                         "keypoints_21","keypoints_3d","status","failure_reason"]:
                    metrics[k] = v

        logs.append({
            "file": f.name,
            "status": status,
            "metrics": metrics,
        })
    return logs


def check_task_status(log_dir):
    """Check status of all tasks."""
    logs = check_local_logs(log_dir)
    for task in TASKS:
        task_logs = [l for l in logs if any(m in l["file"] for m in task["models"])]
        task["log_files"] = task_logs

        if any(l["status"] == "fail" for l in task_logs):
            task["status"] = "failed"
        elif all(l["status"] == "pass" for m in task["models"]
                 for l in task_logs if m in l["file"]):
            task["status"] = "completed"
        elif any(l["status"] == "pass" for l in task_logs):
            task["status"] = "partial"
            # Find which model is done
            for l in task_logs:
                if l["status"] == "pass":
                    for m in task["models"]:
                        if m in l["file"]:
                            task["current_model"] = m
        else:
            task["status"] = "pending"


def print_status():
    """Print current status dashboard."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"  Main Agent Monitor ? {now}")
    print(f"{'='*70}")
    print(f"{'Task':30s} {'Instance':25s} {'Status':12s} {'Current':15s}")
    print(f"{'-'*30} {'-'*25} {'-'*12} {'-'*15}")

    for task in TASKS:
        status_icon = {"pending": "?", "partial": "??", "completed": "?", "failed": "?"}
        icon = status_icon.get(task["status"], "?")
        print(f"{icon} {task['name']:30s} {task['ip']:20s} {task['status']:12s} "
              f"{task['current_model'] or '-':15s}")

    print(f"\n  Log files found: {sum(len(t['log_files']) for t in TASKS)}")
    for task in TASKS:
        if task["log_files"]:
            for lf in task["log_files"]:
                print(f"    {lf['file']}: {lf['status']}")


def run_all_tasks():
    """If all tasks are local, run them in sequence."""
    for task in TASKS:
        if task["ip"] != "local":
            print(f"\n  Task {task['id']} runs on remote GPU ({task['ip']}) ? skipping")
            continue

        # Task 4 (Calibration) runs locally
        print(f"\n  Starting Task {task['id']}: {task['name']}")
        result = subprocess.run(
            [sys.executable, "calibration/calibrate.py"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        print(f"  Task {task['id']} output: {result.stdout[-200:]}")

    # Aggregate results
    print("\n  Aggregating all results...")
    subprocess.run([sys.executable, "scripts/benchmark_summary.py",
                    "--log-dir", "results/logs", "--output", "results"])


def main():
    parser = argparse.ArgumentParser(description="Main Agent Monitor")
    parser.add_argument("--interval", type=int, default=60,
                        help="Check interval in seconds (default: 60)")
    parser.add_argument("--watch", action="store_true",
                        help="Watch mode: continuously check every N seconds")
    parser.add_argument("--all-local", action="store_true",
                        help="Run all local tasks (not remote GPU)")
    args = parser.parse_args()

    if args.all_local:
        run_all_tasks()
        return

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "logs")

    if args.watch:
        print(f"  Watch mode: checking every {args.interval}s. Press Ctrl+C to stop.")
        try:
            while True:
                check_task_status(log_dir)
                print_status()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n  Monitoring stopped.")
    else:
        check_task_status(log_dir)
        print_status()


if __name__ == "__main__":
    main()

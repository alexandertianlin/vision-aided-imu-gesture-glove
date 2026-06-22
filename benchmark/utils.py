# benchmark/utils.py
"""Shared benchmark utilities for hand pose model evaluation."""

import os
import time
import json
import logging
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Log format constants
BENCHMARK_LOG_FORMAT = {
    "model": "",
    "date": "",
    "detect_rate": 0.0,
    "fps": 0.0,
    "latency_ms": 0.0,
    "stability": 0,
    "keypoints_21": "no",
    "keypoints_3d": "no",
    "deployment_time_min": 0.0,
    "status": "",
    "failure_reason": "",
}

def setup_logger(model_name: str, log_dir: str = "results/logs") -> logging.Logger:
    """Create a logger that writes to both console and a model-specific log file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{model_name}_benchmark.log"
    
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"model: {model_name}")
    logger.info(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    return logger


def log_metrics(logger: logging.Logger, metrics: Dict) -> None:
    """Write all benchmark metrics to the log in standard format."""
    for key, val in metrics.items():
        logger.info(f"{key}: {val}")


def write_benchmark_csv(results: List[Dict], output_dir: str = "results/") -> str:
    """Write benchmark results to CSV. Returns the file path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / "benchmark_results.csv"
    
    fields = [
        "model", "detect_rate", "fps", "latency_ms",
        "stability", "keypoints_21", "keypoints_3d",
        "deployment_time_min", "status", "failure_reason"
    ]
    
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    
    return str(path)


def measure_latency(fn, *args, warmup: int = 10, trials: int = 100, **kwargs) -> Tuple[float, float]:
    """Measure average latency and FPS of a function.
    Returns (avg_latency_ms, fps)."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    
    start = time.perf_counter()
    for _ in range(trials):
        fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    
    avg_latency_ms = (elapsed / trials) * 1000
    fps = trials / elapsed
    return avg_latency_ms, fps


def compute_detect_rate(detections: List[bool]) -> float:
    """Compute detection rate from a list of detection booleans."""
    if not detections:
        return 0.0
    return sum(detections) / len(detections)


def compute_stability(keypoints_sequence: List) -> float:
    """Compute stability score (1-5) based on keypoint jitter across frames.\n    Higher = more stable / less jitter."""
    if len(keypoints_sequence) < 3:
        return 5.0
    
    import numpy as np
    diffs = []
    for i in range(1, len(keypoints_sequence)):
        if keypoints_sequence[i] is not None and keypoints_sequence[i-1] is not None:
            diff = np.mean(np.linalg.norm(
                np.array(keypoints_sequence[i]) - np.array(keypoints_sequence[i-1]), axis=-1
            ))
            diffs.append(diff)
    
    if not diffs:
        return 5.0
    
    avg_jitter = float(np.mean(diffs))
    # Map jitter to 1-5 scale (lower jitter = higher score)
    if avg_jitter < 0.5:
        return 5.0
    elif avg_jitter < 1.0:
        return 4.0
    elif avg_jitter < 2.0:
        return 3.0
    elif avg_jitter < 5.0:
        return 2.0
    else:
        return 1.0

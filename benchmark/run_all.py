#!/usr/bin/env python3
"""
Benchmark Orchestrator ? runs all 3 model benchmarks and aggregates results.
Designed to be run on each GCP instance independently, or locally for testing.

Usage:
    python benchmark/run_all.py --models all --test-dir data/rgb_depth_sequences

For individual models:
    python benchmark/run_all.py --models awrnet --test-dir data/rgb_depth_sequences
    python benchmark/run_all.py --models rtmpose --test-dir data/rgb_depth_sequences
    python benchmark/run_all.py --models ipnet --test-dir data/rgb_depth_sequences
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_single_model(model_name: str, script_path: str, test_dir: str, output_dir: str) -> dict:
    """Run a single model's benchmark script and return its output."""
    print(f"\n{'='*60}")
    print(f"  Running {model_name} benchmark...")
    print(f"{'='*60}")

    cmd = [
        sys.executable, script_path,
        "--test-dir", test_dir,
        "--output", output_dir,
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)
    return {
        "model": model_name,
        "returncode": result.returncode,
    }


def aggregate_results(output_dir: str):
    """Run benchmark_summary.py to aggregate all logs."""
    print(f"\n{'='*60}")
    print(f"  Aggregating results...")
    print(f"{'='*60}")

    summary_script = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'benchmark_summary.py')
    if os.path.exists(summary_script):
        cmd = [sys.executable, summary_script, "--log-dir", f"{output_dir}/logs", "--output", output_dir]
        subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Orchestrator")
    parser.add_argument("--models", nargs="+",
                        choices=["all", "awrnet", "rtmpose", "ipnet"],
                        default=["all"],
                        help="Models to benchmark")
    parser.add_argument("--test-dir", default="data/rgb_depth_sequences",
                        help="Directory with test sequences")
    parser.add_argument("--output", default="results/",
                        help="Output directory")
    args = parser.parse_args()

    models_to_run = args.models
    if "all" in models_to_run:
        models_to_run = ["awrnet", "rtmpose", "ipnet"]

    model_scripts = {
        "awrnet": os.path.join(os.path.dirname(__file__), '..', 'models', 'AWR-Net', 'benchmark.py'),
        "rtmpose": os.path.join(os.path.dirname(__file__), '..', 'models', 'RTMPose-Hand', 'benchmark.py'),
        "ipnet": os.path.join(os.path.dirname(__file__), '..', 'models', 'IPNet', 'benchmark.py'),
    }

    results = []
    for model_key in models_to_run:
        script_path = model_scripts.get(model_key)
        if script_path and os.path.exists(script_path):
            result = run_single_model(model_key.upper(), script_path, args.test_dir, args.output)
            results.append(result)
        else:
            print(f"  WARNING: Script not found for {model_key}: {script_path}")

    # Aggregate
    aggregate_results(args.output)

    print(f"\n{'='*60}")
    print(f"  Benchmark Complete!")
    print(f"  Results: {args.output}/benchmark_results.csv")
    print(f"  Summary: {args.output}/model_comparison.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

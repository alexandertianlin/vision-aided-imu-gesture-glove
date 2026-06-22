 #!/usr/bin/env python3
 """
 benchmark_summary.py — Aggregate individual model logs into a single CSV/Markdown report.

 Usage:
     python scripts/benchmark_summary.py --log-dir results/logs/ --output results/

 Reads per-model log files, extracts metrics, and generates:
   - results/benchmark_results.csv
   - results/model_comparison.md
 """

 import argparse
 import csv
 import json
 import os
 import re
 import sys
 from pathlib import Path
 from typing import Dict, List, Optional


 METRICS_FIELDS = [
     "model",
     "detect_rate",
     "fps",
     "latency_ms",
     "stability",
     "keypoints_21",
     "keypoints_3d",
     "deployment_time_min",
     "status",
     "failure_reason",
 ]

 
 def parse_log_file(log_path: Path) -> Dict[str, str]:
     """Parse a single model's log file for benchmark metrics."""
     result = {field: "" for field in METRICS_FIELDS}
     result["model"] = log_path.stem.replace("_benchmark", "").replace("_log", "")
     result["status"] = "pass"

     text = log_path.read_text()

     # Detect failures
     if "FAIL" in text or "fail" in text.lower():
         result["status"] = "fail"
         # Try to extract failure reason
         fail_match = re.search(r"FAIL:\s*(.+)", text)
         if fail_match:
             result["failure_reason"] = fail_match.group(1).strip()
         else:
             result["failure_reason"] = "Unknown failure"
         return result

     # Parse metrics
     patterns = {
         "detect_rate": r"(?:detect_rate|detection rate|detect rate)[:\s]*([0-9.]+)",
         "fps": r"(?:fps|frame rate)[:\s]*([0-9.]+)",
         "latency_ms": r"(?:latency|latency_ms)[:\s]*([0-9.]+)",
         "stability": r"(?:stability)[:\s]*([0-9.]+)",
         "keypoints_21": r"(?:keypoints_21|21.keypoints)[:\s]*(yes|no|true|false)",
         "keypoints_3d": r"(?:keypoints_3d|3d.keypoints)[:\s]*(yes|no|true|false)",
         "deployment_time_min": r"(?:deployment_time|deploy.time)[:\s]*([0-9.]+)",
     }

     for field, pattern in patterns.items():
         match = re.search(pattern, text, re.IGNORECASE)
         if match:
             result[field] = match.group(1).lower().replace("true", "yes").replace("false", "no")

     return result

 
 def generate_csv(results: List[Dict[str, str]], output_path: Path) -> None:
     """Write benchmark results to CSV."""
     output_path.parent.mkdir(parents=True, exist_ok=True)
     with open(output_path, "w", newline="") as f:
         writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
         writer.writeheader()
         writer.writerows(results)
     print(f"CSV written to {output_path}")

 
 def generate_markdown(results: List[Dict[str, str]], output_path: Path) -> None:
     """Write benchmark comparison table to Markdown."""
     # Sort: passes first, then by detect_rate descending
     passed = [r for r in results if r["status"] == "pass"]
     failed = [r for r in results if r["status"] == "fail"]
     passed.sort(key=lambda r: float(r.get("detect_rate") or 0), reverse=True)
     sorted_results = passed + failed

     lines = [
         "# Model Comparison — Glove Hand Tracking\n",
         "## Summary\n",
         f"- **Tested:** {len(results)} models",
         f"- **Passed:** {len(passed)}",
         f"- **Failed:** {len(failed)}",
         f"- **Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
         "## Results Table\n",
     ]

     # Build markdown table
     header = "| Model | Detect Rate | FPS | Latency (ms) | Stability | 21 KP | 3D KP | Status |"
     sep = "|------|--------------|-----|---------------|-----------|-------|-------|--------|"
     lines.append(header)
     lines.append(sep)

     for r in sorted_results:
         row = (
             f"| {r.get('model', '?')} "
             f"| {r.get('detect_rate', '-')} "
             f"| {r.get('fps', '-')} "
             f"| {r.get('latency_ms', '-')} "
             f"| {r.get('stability', '-')} "
             f"| {r.get('keypoints_21', '-')} "
             f"| {r.get('keypoints_3d', '-')} "
             f"| {r.get('status', '?')} |"
         )
         lines.append(row)

     # Add failure reasons
     if failed:
         lines.extend([
             "\n## Failed Models\n",
             "| Model | Failure Reason |",
             "|-------|---------------|",
         ])
         for r in failed:
             lines.append(f"| {r['model']} | {r.get('failure_reason', 'Unknown')} |")

     # Add recommendation
     if passed:
         best = passed[0]
         lines.extend([
             "\n## Recommendation\n",
             f"**Top Candidate:** {best['model']}",
             f"- Detect Rate: {best.get('detect_rate', 'N/A')}",
             f"- FPS: {best.get('fps', 'N/A')}",
             f"- Latency: {best.get('latency_ms', 'N/A')} ms",
             "",
         ])
         if len(passed) > 1:
             lines.append(f"**Runner-up:** {passed[1]['model']}")
     else:
         lines.append("\n## Recommendation\nNo model passed all benchmarks.\n")

     output_path.parent.mkdir(parents=True, exist_ok=True)
     output_path.write_text("\n".join(lines) + "\n")
     print(f"Markdown written to {output_path}")

 
 def main():
     parser = argparse.ArgumentParser(
         description="Aggregate model benchmark logs into summary report"
     )
     parser.add_argument(
         "--log-dir",
         default="results/logs",
         help="Directory containing per-model log files",
     )
     parser.add_argument(
         "--output",
         default="results/",
         help="Output directory for CSV and Markdown reports",
     )
     args = parser.parse_args()

     log_dir = Path(args.log_dir)
     output_dir = Path(args.output)

     if not log_dir.exists():
         print(f"ERROR: Log directory not found: {log_dir}")
         sys.exit(1)

     # Collect all log files
     log_files = sorted(log_dir.glob("*.log"))
     if not log_files:
         print(f"WARNING: No log files found in {log_dir}")
         # Still generate empty report
         log_files = []

     results = []
     for log_path in log_files:
         result = parse_log_file(log_path)
         results.append(result)
         print(f"  Parsed: {log_path.name} → {result['model']}: {result['status']}")

     generate_csv(results, output_dir / "benchmark_results.csv")
     generate_markdown(results, output_dir / "model_comparison.md")

     print(f"\nDone. {len(results)} models processed.")

 if __name__ == "__main__":
     main()

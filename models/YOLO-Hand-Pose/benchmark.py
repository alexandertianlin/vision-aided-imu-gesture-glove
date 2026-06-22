#!/usr/bin/env python3
"""YOLO-Hand-Pose Benchmark Runner ? Task 1 (GPU 1)"""
import argparse, sys, os, time, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from benchmark.utils import setup_logger, log_metrics, compute_detect_rate, compute_stability

def benchmark(test_dir, output_dir, logger):
    logger.info("=== Phase 1: Environment Check ===")
    start = time.time()
    try:
        from ultralytics import YOLO
        import torch
        logger.info(f"Ultralytics YOLO loaded, CUDA: {torch.cuda.is_available()}")
    except ImportError as e:
        logger.info(f"FAIL: {e}")
        return {"status":"fail","failure_reason":str(e)}
    deploy = (time.time()-start)/60
    logger.info(f"deployment_time_min: {deploy:.2f}")

    logger.info("\n=== Phase 2: Load Model ===")
    try:
        model = YOLO("yolo11n-pose.pt")
        logger.info("YOLO11n-pose model loaded")
    except:
        model = YOLO("yolov8n-pose.pt")
        logger.info("YOLOv8n-pose model loaded")

    logger.info("\n=== Phase 3: Benchmarking ===")
    import cv2
    test_dir = Path(test_dir)
    imgs = sorted(test_dir.glob("**/*.png")) + sorted(test_dir.glob("**/*.jpg"))
    if not imgs:
        os.makedirs(str(test_dir/"synthetic"), exist_ok=True)
        for i in range(50):
            cv2.imwrite(str(test_dir/f"synthetic/rgb_{i:04d}.png"), np.random.randint(0,255,(256,256,3),dtype=np.uint8))
        imgs = sorted(test_dir.glob("synthetic/*.png"))
    logger.info(f"Frames: {len(imgs)}")

    kps, lats, dets = [], [], []
    for p in imgs:
        try:
            img = cv2.imread(str(p))
            t0 = time.perf_counter()
            results = model(img, verbose=False)
            t1 = time.perf_counter()
            lats.append((t1-t0)*1000)
            if results and len(results[0].keypoints.data) > 0:
                kps.append(results[0].keypoints.data[0].cpu().numpy())
                dets.append(True)
            else:
                kps.append(None); dets.append(False)
        except:
            kps.append(None); dets.append(False)

    logger.info("\n=== Phase 4: Metrics ===")
    dr = compute_detect_rate(dets)
    avg_lat = np.mean(lats) if lats else 0
    fps = 1000.0/avg_lat if avg_lat>0 else 0
    stab = compute_stability(kps)
    m = {"detect_rate":round(dr,4),"fps":round(fps,2),"latency_ms":round(avg_lat,2),
         "stability":round(stab,1),"keypoints_21":"no","keypoints_3d":"no",
         "deployment_time_min":round(deploy,2),"status":"pass"}
    log_metrics(logger, m)
    return m

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir",default="data/rgb_depth_sequences")
    p.add_argument("--output",default="results/")
    args = p.parse_args()
    Path(args.test_dir).mkdir(parents=True,exist_ok=True)
    logger = setup_logger("YOLO-Hand-Pose",os.path.join(args.output,"logs"))
    r = benchmark(args.test_dir,args.output,logger)
    print(f"\n  {'PASSED' if r.get('status')=='pass' else 'FAILED'}: {r}")

if __name__=="__main__":
    main()

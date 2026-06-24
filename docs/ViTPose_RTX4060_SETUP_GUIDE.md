# ViTPose GPU Setup Guide — RTX 4060 (CUDA 12.x)

> Applicable to the new laptop with RTX 4060 for ViTPose environment setup
> 2026-06-24

---

## Hardware Specs

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 4060 (Laptop, ~6GB VRAM) |
| CUDA Architecture | Ada Lovelace (SM 8.9) |
| CUDA Version | 12.4 (recommended) / 12.1 (compatible) |
| VRAM | 6-8GB GDDR6 |

RTX 4060 supports CUDA 12.x. Install PyTorch 2.1+ with CUDA 12.1.

---

## Strategy: Two Parallel Approaches

Since ViTPose official InterHand2.6M (hand 21-keypoint) pretrained weights
are marked "Coming Soon" in the GitHub repo, we need two parallel approaches:

### Approach A: ONNX Runtime GPU (Quick Start — use this week)

Use the existing vitpose_hand.onnx + vitpose_hand.onnx.data, but switch to
GPU inference instead of CPU.

**Current issue**: realtime_vitpose.py uses
`providers=['CPUExecutionProvider']` (CPU only). We just need to enable
CUDAExecutionProvider to leverage RTX 4060.

```
# Install onnxruntime-gpu (CUDA 12.x version)
pip install onnxruntime-gpu==1.17.1
# or latest:
python -m pip install onnxruntime-gpu ^
  --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/^
  _packaging/onnxruntime-cuda-12/pypi/simple/
```

**Modify realtime_vitpose.py** — change the session creation line:
```python
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
    }),
    'CPUExecutionProvider',
]
session = ort.InferenceSession(ONNX_PATH, providers=providers)
```

**Verify**:
```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should include CUDAExecutionProvider
```

**Model location** (copy to task directory):
```
C:\Users\tianl\Documents\Codex\2026-06-16\github-...\outputs\
  vitpose_hand.onnx      (541KB)
  vitpose_hand.onnx.data (359MB)
```

---

### Approach B: PyTorch ViTPose Full Environment (Long-term)

Clone and configure the complete ViTPose environment for native CUDA inference.

**B1. Environment Setup**

```
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose

pip install torch==2.1.0 torchvision==0.16.0 ^
  --index-url https://download.pytorch.org/whl/cu121

pip install mmcv==2.1.0 -f ^
  https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

pip install -v -e .
pip install timm==0.4.9 einops opencv-python
```

Note: RTX 4060 is Ada Lovelace (SM 8.9). If mmcv prebuilt wheel fails,
build from source:
```
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv && git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
```

**B2. Download Pretrained Weights**

Method 1: ViTPose++ WholeBody (Recommended — includes hand keypoints)

ViTPose++ multi-task model includes 133 keypoints total (body 17 + face 68
+ left_hand 21 + right_hand 21 + foot 6). Hand keypoints extractable.

| Model | OneDrive Link |
|-------|--------------|
| ViTPose++-S (384 dim) | https://1drv.ms/u/s!AimBgYV7JjTlgccrwORr61gT9E4n8g?e=kz9sz5 |
| ViTPose++-B (768 dim) | https://1drv.ms/u/s!AimBgYV7JjTlgcckRZk1bIAuRa_E1w?e=ylDB2G |
| ViTPose++-L (1024 dim) | https://1drv.ms/u/s!AimBgYV7JjTlgccs1SNFUGSTsmRJ8w?e=a9zKwZ |

RTX 4060 recommendation: ViTPose++-S (~300MB) or ViTPose++-B (~700MB)

Method 2: COCO body keypoint model (fallback)

| Model | Resolution | AP | Download |
|-------|-----------|-----|---------|
| ViTPose-S | 256x192 | 73.8 | https://1drv.ms/u/s!AimBgYV7JjTlgccifT1XlGRatxg3vw |
| ViTPose-B | 256x192 | 75.8 | https://1drv.ms/u/s!AimBgYV7JjTlgSMjp1_NrV3VRSmK |

**B3. Model Split and Inference**

ViTPose++ weights are multi-task. Split first:
```
python tools/model_split.py ^
  --source ./checkpoints/vitpose++_base_wholebody.pth
```

Then run inference with hand config:
```
bash tools/dist_test.sh ^
  configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/^
  ViTPose_base_interhand2d_all_256x192.py ^
  ./checkpoints/vitpose++_hand.pth 1
```

**Config key parameters** (from GitHub ViTPose_base_interhand2d_all_256x192.py):
- Input size: 256x192 (HxW)
- Heatmap size: 48x64
- Output: 21 hand keypoints
- Backbone: ViT-Base (embed_dim=768, depth=12, num_heads=12)
- Normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]

---

## Approach Comparison

| Dimension | A: ONNX GPU | B: PyTorch WholeBody |
|-----------|------------|---------------------|
| Deploy time | 10 minutes | 1-2 hours |
| GPU type | onnxruntime-gpu CUDA | Native PyTorch CUDA |
| Hand kpts | 21 (dedicated) | 21 (from WholeBody) |
| Model size | 359MB | S:~300MB / B:~700MB |
| RTX 4060 | Out of box | May need compile |
| Fine-tunable | No | Yes |
| Use | Quick start this week | Long-term solution |

Suggestion: Start with Approach A to verify ViTPose pipeline on RTX 4060,
while setting up Approach B in parallel.

---

## Model Compression (Phase 5)

### ONNX FP16 Quantization
```
pip install onnx-simplifier
python -m onnxsim vitpose_hand.onnx vitpose_hand_sim.onnx

python -c "import onnx; from onnxconverter_common import float16; \
m=onnx.load('vitpose_hand.onnx'); \
onnx.save(float16.convert_float_to_float16(m), 'vitpose_hand_fp16.onnx')"
```

### PyTorch Compression
```
model.half()
torch.jit.script(model).save('vitpose_hand_ts.pt')
```

---

## Reference Links

- ViTPose GitHub: https://github.com/ViTAE-Transformer/ViTPose
- mmpose docs: https://mmpose.readthedocs.io/
- ONNX Runtime GPU: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- InterHand2.6M: https://mks0601.github.io/InterHand2.6M/

---

> Last updated: 2026-06-24

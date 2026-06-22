# Vision-Aided IMU Gesture Glove - AGENTS.md

> 视觉算法 + IMU 传感器融合，驱动 Unity 3D 手部展示

---

## 项目概述

融合摄像头手部识别（MediaPipe / HAMER）与可穿戴 IMU 传感器手套，
通过 UDP 协议将两者信号同时发送到 Unity，实现 3D 手部实时驱动。

核心设计原则：
- 视觉信号存在时，作为 IMU 漂移的校正锚点
- 视觉信号消失时，IMU 信号独立接管
- 每指独立校准数据库（open/fist），掌朝向门控（PALM/BACK/SIDE）
- Unity 端全链路诊断日志（UDP_RECV到TAKEOVER_FINISH）

---

## 系统架构

```text
                   +-------------------+        +----------------------+
                   | 传感器手套        | 串口   | Unity                |
                   | STM32 + 6xIMU     | COM3   | SerialReceiver       |
                   | 触觉传感器        | 460800 | HandMotionManager    |
                   | 四元数 + 力数据   +------->+ FingerSolver IK     +---> 3D手模型
                   |                   |        | VisionCorrection     |
                   +-------------------+        +---------^------------+
                                                            |
                   +-------------------+        +----------+-----------+
                   | 摄像头            | UDP    | Python               |
                   | D435i RGB         | port   | MediaPipe Hand       |
                   | iPhone DroidCam   | 5055   | 每指 open/fist 分类  |
                   | 任意 USB 摄像头   +------->+ 掌朝向门控           |
                   |                   |        | 校准数据库           |
                   +-------------------+        +----------------------+
```

**融合逻辑：**
1. IMU 手套通过串口持续发送 6 节点四元数 + 力数据到 Unity
2. 摄像头通过 UDP 发送每指状态（open/fist/IDLE）到 Unity
3. Unity 的 HandMotionManager 将 IMU 四元数映射为手部姿态
4. VisionFingerCorrectionReceiver 根据视觉信号校正 IMU 累积漂移
5. 视觉信号超时 → IMU 独立接管（无视觉依赖）
6. 掌心朝前且五指全开 → VisionOpenPalmRefreshModule 触发 IMU 基线重置

---

## 版本历史

| 版本 | 特征 | 日期 | 状态 |
|------|------|------|------|
| V1 | Orbbec Astra Plus + MediaPipe + 手指标定 | 2026-06-15 | completed |
| V1.1 | ChArUco 标定 + RGB-D 对齐 | 2026-06-05 | completed |
| V2.0 | D435i + MediaPipe Hand 基础检测 | 2026-06-21 | completed |
| V2.1 | D435i + MediaPipe + UDP + IMU (当前版本) | 2026-06-22 | completed |
| V2.2 | MediaPipe 直接驱骨（端口 5056） | 2026-06-21 | in_progress |
| V2.3 | FingerSolver IK 管线集成（端口 5057） | 2026-06-21 | in_progress |
| V3 | HAMER + IMU 深度融合 | TBD | pending |

---

## V1 实现（奥比中光文件夹）

V1 完整实现在：
```
C:\Users\tianl\Documents\Codex\sensors\orbbec\
```

关键文件：
- `work/serial_imu_reader.py` — STM32 串口 IMU 协议解析（b5 a5 55, 115200 baud）
- `work/orbbec_imu_bridge.py` — OrbbecSDK ctypes 包装（摄像头内置 IMU）
- `work/orbbec_viewer_imu.py` — 多模态录制 + IMU 同步显示（6 轮迭代）
- `work/orbbec_viewer_imu.py:IMUProcessor` — 互补滤波器（accel/gyro → euler）

V1 中的手指标定逻辑（奥比中光版本）：
- 基于 MediaPipe Hand 检测 21 关键点
- 计算每指 distance score（tip-to-mcp / bone_chain_length）
- 通过 open-close 自校准建立每指的最小/最大范围
- 识别到手指完全打开时 → UDP 发送 FINGER_OPEN → Unity 端强制手指复位
- 识别到握拳时 → UDP 发送 FINGER_FIST

---

## 当前状态（V2）

### 已完成
- [x] D435i RGB 流 + MediaPipe HandLandmarker 检测
- [x] 21 关键点 + 每指 curl/spread + 手腕朝向四元数
- [x] UDP 发送到 Unity（端口 5055）
- [x] Unity SerialReceiver 串口接收（COM3, 460800）
- [x] HandMotionManager + FingerSolver IK（Swing-Twist 分解）
- [x] VisionFingerCorrectionReceiver 诊断日志
- [x] VisionOpenPalmRefreshModule 基线重置
- [x] Python IMU 监控工具（serial_imu_reader.py）
- [x] 部署文档（DEPLOYMENT_GUIDE.md）

### 进行中
- [ ] MediaPipe 直接驱骨（v2.2, 端口 5056）
- [ ] FingerSolver IK 管线集成（v2.3, 端口 5057）
- [ ] 端到端验证：视觉信号消失 → IMU 接管

### 待办
- [ ] HAMER 模型部署 + Unity 集成
- [ ] 4060 GPU 上 ViTPose 推理测试
- [ ] 深度摄像头手腕平移追踪
- [ ] 多摄像头融合（遮挡处理）

---

## 诊断流程（信号流向排查）

```text
摄像头 → Python MediaPipe → UDP → Unity VisionFingerCorrectionReceiver
                                     ↓
 检查点 1: vision_finger_diagnostic.log 中是否有 UDP_RECV?
  → 有: 网络通，检查 PARSE_OK / FILTER_REJECT
  → 无: 防火墙拦截？端口 5055 是否开放？

传感器手套 → 串口 → Unity SerialReceiver → HandMotionManager
                                     ↓
 检查点 2: Unity Console 中是否有 ID:{hex} 四元数日志？
  → 有: 串口通
  → 无: 检查 COM3 端口、460800 波特率

HandMotionManager → FingerSolver → 3D 手
                ↓
 检查点 3: enableVisionCorrection 是否为 true？
  → true: 视觉校正启用
  → false: 仅 IMU 驱动，无水合
```

---

## 硬件配置

| 组件 | 规格 |
|------|------|
| 传感器手套 | STM32 + 6x MPU6050, USB 串口四元数/力数据 |
| 深度摄像头 | Intel RealSense D435i (RGB + 深度) |
| 传统摄像头 | Orbbec Astra Plus / iPhone DroidCam / USB 摄像头 |
| GPU（当前） | RTX 4060 Laptop (8GB VRAM) |
| GPU（规划） | 3 独立 GPU（GPU A/B/C 并行） |
| Unity 版本 | 2022.3 LTS |
| 串口 | COM3 @ 460800 baud |

---

## 相关路径

| 资源 | 路径 |
|------|------|
| V1 Orbbec 完整实现 | `C:\Users\tianl\Documents\Codex\sensors\orbbec\` |
| V2 仓库根目录 | `C:\Users\tianl\Documents\Codex\repos\alexandertianlin-vision-imu-gesture-glove\` |
| Python 视觉管线 | `python/mediapipe_udp_sender.py` |
| Unity IMU 接收 | `unity/Assets/Scenes/SerialReceiver.cs` |
| Unity 手部驱动 | `unity/Assets/Scenes/HandMotionManager.cs` |
| Unity 手指 IK | `unity/Assets/Scenes/FingerSolver.cs` |
| Unity 视觉校正 | `unity/Assets/Scenes/VisionFingerCorrectionReceiver.cs` |
| Unity 基线重置 | `unity/Assets/Scenes/VisionOpenPalmRefreshModule.cs` |
| Python IMU 监控 | `python/serial_imu_reader.py` |
| 部署指南 | `docs/DEPLOYMENT_GUIDE.md` |

---

*最后更新: 2026-06-22*

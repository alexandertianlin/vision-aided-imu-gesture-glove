# Task Specification: Vision-Aided IMU Gesture Glove

> 视觉算法 + IMU 传感器融合，驱动 Unity 3D 手部展示

---

## 1. 项目目标

融合摄像头手部识别算法与可穿戴 IMU 传感器手套的数据，通过串口 + UDP
双通道将信号同时发送到 Unity 上位机，实现 3D 手部模型的实时、鲁棒驱动。

### 核心原则

- **视觉优先**：视觉信号存在时作为 IMU 漂移的校正锚点
- **IMU 接管**：视觉信号丢失时（遮挡、角度差、光照不足），IMU 独立驱动
- **状态门控**：仅在可靠手势状态（掌心朝前 + 五指全开）时重置 IMU 基线
- **每指独立**：每指 open/fist 状态独立校准、独立校正

---

## 2. V1 实现（奥比中光版本）

### 2.1 位置

```
C:\Users\tianl\Documents\Codex\sensors\orbbec\
```

### 2.2 实现内容

| 模块 | 文件 | 说明 |
|------|------|------|
| 串口 IMU 解析 | `work/serial_imu_reader.py` | 0xB5 0xA5 0x55 协议, COM3 @ 115200 baud |
| OrbbecSDK IMU | `work/orbbec_imu_bridge.py` | ctypes 包装 OrbbecSDK.dll 读取摄像头内置 IMU |
| 多模态 Viewer | `work/orbbec_viewer_imu.py` | 视频 + IMU 同步采集/录制/分析 |
| 互补滤波器 | `IMUProcessor` class | accel/gyro → euler 角度融合 |
| 同步分析 | `analyze_sync_offsets()` | 视频帧-IMU 时间戳二分匹配 |

### 2.3 手指标定逻辑（V1 核心）

```
摄像头 → MediaPipe Hand → 21 关键点
                          ↓
              每指 tip-to-mcp distance
              / bone_chain_length 归一化
                          ↓
              打开-关闭自校准 → 每指 min/max 范围
                          ↓
              实时分类: OPEN / FIST / IDLE
                          ↓
              UDP → Unity → 手指复位/握拳
```

- 手指完全打开 → 发送 `FINGER_OPEN` → Unity 强制手指复位到 `openPitch`
- 手指握拳 → 发送 `FINGER_FIST` → Unity 强制手指弯曲到 `fistPitch`
- 使用 `HAND_OPEN` / `HAND_FIST` 做手势级命令
- 掌心朝向门控：只有 PALM 朝向时才允许校正

---

## 3. V2 实现（当前版本）

### 3.1 架构

```text
[传感器手套]                              [摄像头]
    STM32 + 6x MPU6050                    D435i / iPhone / Webcam
    USB 串口 (COM3)                       RTSP / UVC
         |                                     |
         v                                     v
  [Unity: SerialReceiver]              [Python: mediapipe_udp_sender.py]
    460800 baud                          MediaPipe HandLandmarker
    6 节点四元数 + 力数据                   每指 distance score
         |                                    每指 open/fist 分类
         v                                    校准数据库
  [HandMotionManager]                    手朝向分类 (PALM/BACK/SIDE)
    Swing-Twist 分解                           |
    IMU 校准基线                                v
         |                              UDP (端口 5055)
         v                                     |
  [FingerSolver IK]                      [VisionFingerCorrectionReceiver]
    每指增量追踪                              visual confidence 门控
    握拳边界姿态                              每指 visual hold watchdog
         |                                    open-palm refresh
         v                                     |
  [Unity 3D 手模型] ← 融合校正 ←---------------+
```

### 3.2 关键差异：V1 vs V2

| 维度 | V1（奥比中光） | V2（当前） |
|------|---------------|-----------|
| 摄像头 | Orbbec Astra Plus | D435i / iPhone DroidCam |
| 串口波特率 | 115200 | 460800 |
| 数据类型 | raw accel/gyro + euler | pre-computed quaternion + force |
| 节点数 | 1 | 6（palm + 5 指） |
| Unity IK | 简单角度复位 | FingerSolver 增量追踪 + 握拳边界 |
| 视觉校正 | 强制复位 | 渐进 blend + 每指 hold 保护 |
| 校准 | 实时范围学习中 | 持久化校准数据库 (JSON) |
| 诊断 | 无 | 完整 vision_finger_diagnostic.log |

### 3.3 协议规格

**串口协议（0xB5 0xA5 0x55）：**

| 偏移 | 长度 | 字段 |
|------|------|------|
| 0-2 | 3 | 帧头 0xB5 0xA5 0x55 |
| 6 | 1 | device_id: 0x30=Palm, 0x1E=Thumb, 0x28=Index, 0x32=Middle, 0x3C=Ring, 0x46=Little |
| 8-15 | 8 | int16[4] 四元数 (w,x,y,z) / 10000 |
| 22-33 | 12 | float32[3] 力向量（仅手指节点）|

**UDP JSON 协议（端口 5055）：**

```json
{
  "timestampMs": 12345,
  "sequenceId": 1,
  "command": "FINGER_OPEN",
  "fingerName": "Index",
  "fingerIndex": 1,
  "confidence": 0.85,
  "vis_conf": 0.90,
  "score": 0.78,
  "isPalmFacing": true
}
```

commands: `FINGER_OPEN`, `FINGER_FIST`, `TRIGGER_OPEN`, `TRIGGER_FIST`, `IDLE`

---

## 4. 信号流向排查指南

### 4.1 排查场景 A：Unity 3D 手不动

```
[串口] → SerialReceiver → HandMotionManager → FingerSolver
   ①                   ②                    ③
```

**① 串口是否收到数据？**
- 在 Unity Console 中过滤 `ID:` 日志
- 或在 Python 终端运行：
  ```cmd
  cd python
  python serial_imu_reader.py --port COM3 --baud 460800
  ```
- 预期输出 6 个节点的四元数（Palm/Thumb/Index/Middle/Ring/Little）

**② HandMotionManager 是否校准？**
- Console 输出 `状态: 已校准跟踪中` 或 `状态: 未校准`
- 按空格键校准
- 检查 `isCalibrated` 是否为 true

**③ FingerSolver 是否有骨骼绑定？**
- Inspector 中检查 `FingerSolver.rootBone / midBone / tipBone`
- [Auto Assign Bones] 上下文菜单自动分配

### 4.2 排查场景 B：视觉校正不生效

```
[Python] → UDP (5055) → [VisionFingerCorrectionReceiver]
    ①                    ②                 ③
```

**① Python 端是否在发送？**
- 终端输出 `PY_SEND seq=... finger=Index state=OPEN`
- 确认摄像头连接正常、手部可见

**② Unity 是否收到 UDP？**
- 检查 `vision_finger_diagnostic.log` 中 `UDP_RECV` 条目
- 如果无 `UDP_RECV`：防火墙阻止了 5055 端口
- 如果有 `UDP_RECV` + `PARSE_OK`：网络通
- 如果有 `REJECT_BY_CONFIDENCE`：置信度低于 `visualConfidenceThreshold(0.75)`

**③ enableVisionCorrection 是否为 true？**
- 默认 **false**！必须在 Inspector 中勾选
- 勾选后应有 `UDP_START` 日志

### 4.3 排查场景 C：open-palm refresh 不触发

- 需要 `enableOpenPalmRefresh = true`
- 需要 `enableVisionCorrection = true`
- 需要 `isPalmFacing = true`（掌心朝向摄像头）
- 有 3 秒冷却时间（`refreshCooldownSeconds`）
- 检查 Console 中 `Vision open-palm refresh` 日志

---

## 5. 双笔记本部署策略

### 5.1 硬件清单（目标笔记本）

| 项目 | 备注 |
|------|------|
| Windows 笔记本 | Unity 2022.3 LTS, Python 3.10+ |
| 传感器手套 | USB 串口, 6x IMU |
| D435i / 摄像头 | USB 3.0 |
| RTX 4060（可选） | 仅用于 ViTPose/HAMER |

### 5.2 部署步骤

1. 克隆仓库：
   ```cmd
   git clone https://github.com/alexandertianlin/vision-aided-imu-gesture-glove.git
   ```
2. Python 环境：
   ```cmd
   cd python
   python -m venv .venv
   pip install -r requirements.txt
   ```
3. Unity 项目：Hub → Add → 选择 `unity/` 文件夹
4. 连接传感器手套，确认 COM3 端口
5. 运行视觉管线：`python mediapipe_udp_sender.py`
6. Unity 中 Play → 按空格校准 → 手跟随

### 5.3 视觉配置（摄像头）

| 摄像头类型 | 配置文件位置 | 修改方式 |
|-----------|-------------|---------|
| iPhone DroidCam | `mediapipe_udp_sender.py:RTSP_URL` | 修改 IP 和端口 |
| D435i | `mediapipe_udp_sender.py` | 替换为 `cv2.VideoCapture(0)` |
| Orbbec Astra Plus | `mediapipe_udp_sender.py` | 使用 UVC 驱动 |
| 普通 USB 摄像头 | `mediapipe_udp_sender.py` | 替换为 `cv2.VideoCapture(camera_id)` |

---

## 6. 未来规划（V3：HAMER + IMU 深度融合）

### 6.1 架构

```text
[摄像头] → [Python: HAMER / ViTPose]
              ↓ 每指 21 关键点 + 3D 旋转
              ↓ UDP 端口 5055 (兼容现有接口)
[传感器手套] → [Unity: SerialReceiver + HandMotionManager]
              ↓ IMU 四元数
[融合层: VisionFingerCorrectionReceiver]
    - HAMER 提供高精度 3D 手部姿态
    - IMU 提供高频低延迟追踪
    - 融合策略：视觉锚点 + IMU 插值
```

### 6.2 前置条件

- [ ] RTX 4060 或更好 GPU 到位
- [ ] ViTPose 模型下载 + 环境配置
- [ ] HAMER 模型依赖（MANO, PyTorch3D）
- [ ] 与现有 UDP 接口兼容性验证
- [ ] 性能基准测试（FPS, 延迟, 显存）

---

## 7. 文件索引

| 文件 | 用途 |
|------|------|
| `AGENTS.md` | Agent 操作指南 |
| `README.md` | 项目概览 + 快速开始 |
| `_TASK_SPEC.md` | 本文件：详细规格说明 |
| `docs/DEPLOYMENT_GUIDE.md` | 部署指南 |
| `python/mediapipe_udp_sender.py` | MediaPipe 视觉管线 |
| `python/serial_imu_reader.py` | IMU 串口监控 |
| `unity/Assets/Scenes/SerialReceiver.cs` | 串口接收 |
| `unity/Assets/Scenes/HandMotionManager.cs` | IMU → 手部驱动 |
| `unity/Assets/Scenes/FingerSolver.cs` | 手指 IK |
| `unity/Assets/Scenes/VisionFingerCorrectionReceiver.cs` | 视觉校正 |
| `unity/Assets/Scenes/VisionOpenPalmRefreshModule.cs` | 基线重置 |

---

## 8. 已知问题

1. **enableVisionCorrection 默认关闭** — 需要在 Unity Inspector 中手动勾选
2. **视觉置信度阈值 0.75** — 光照较差时需要降低
3. **串口波特率必须匹配 460800** — 与固件不一致则无数据
4. **防火墙可能阻止 UDP 5055** — 需添加入站规则
5. **D435i hardware_reset 导致 USB 永久中断** — 已从代码中移除

---

*最后更新: 2026-06-22*
*版本: v2.2*

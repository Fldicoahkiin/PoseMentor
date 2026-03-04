# 训练框架与流程

## 1. 目标与范围

本项目训练链路的目标是：将单摄像头 2D 关键点序列映射为 3D 关节轨迹，并把输出接入动作评分与纠错模块。

当前可用训练主链路基于 AIST++，同时保留自采单视角和自采四机位扩展位。

## 2. 全流程框架

```mermaid
flowchart LR
    A["数据注册（configs/datasets.yaml）"] --> B["数据准备"]
    B --> C["2D关键点构建"]
    C --> D["2D/3D样本配对"]
    D --> E["标准化统计（mean/std）"]
    E --> F["3D Lift 训练"]
    F --> G["验证与导出"]
    G --> H["在线推理与评分"]

    B1["AIST++ 注释与视频"] --> B
    B2["自采单视角视频"] --> B
    B3["自采四机位视频"] --> B
    B3 --> M["多机位对齐/格式化"]
    M --> C
```

## 3. 训练输入与输出约定

### 3.1 2D 输入文件（`*.npz`）

路径：`data/processed/<dataset>/yolo2d/*.npz`

字段约定：

- `keypoints2d`：`[T, 17, 3]`，最后一维是 `x, y, conf`
- `style`：动作类别或舞种标识（可选）
- 其他字段允许保留，但训练脚本至少需要 `keypoints2d`

### 3.2 3D 标签文件（`*.npz`）

路径：`data/processed/<dataset>/gt3d/*.npz`

字段约定：

- `joints3d`：`[T, 17, 3]`
- 推荐使用与 2D 同名文件，便于自动配对

### 3.3 样本配对规则

实现位置：`src/posementor/data/aist_dataset.py`

- 通过文件名 stem 做 2D/3D 配对
- 两端帧长不一致时按最短长度截断
- 训练/验证按序列名 hash 做稳定划分（默认 `val_ratio=0.1`）
- 序列长度不足最小窗口时自动跳过

## 4. AIST++ 当前训练主链路

### 4.1 数据准备

- 脚本：`download_and_prepare_aist.py`
- 输入：`data/raw/aistpp/annotations`
- 输出：`data/processed/aistpp/gt3d`

### 4.2 2D 构建

两种方式可互换：

1. 使用官方 2D 标注快速构建训练输入  
   脚本：`extract_pose_aist2d.py`
   - AIST++ `keypoints2d` 是 9 机位聚合格式（`cAll`）
   - 训练链路使用单视角输入，默认每条序列选取一个机位
   - 可用 `--camera-index 1` 固定机位，保证全量数据视角一致

2. 从视频跑 YOLO11-Pose 提取 2D  
   脚本：`extract_pose_yolo11.py`

两种方式输出目录一致：`data/processed/aistpp/yolo2d`

### 4.3 模型与损失

实现位置：`train_3d_lift_demo.py`

- 模型：`PoseLiftTransformer`（时序 Transformer）
- 输入：`kp2d [B, T, 17, 2]` + `conf [B, T, 17, 1]`
- 输出：`pred3d [B, T, 17, 3]`
- 损失构成：
  - 位置损失：关键点置信度加权 L1
  - 速度损失：相邻帧速度差约束
- 训练框架：PyTorch 2.x + Lightning

### 4.4 评估与导出

- 验证指标：`val_loss`、`val_mpjpe`
- 产物目录：`artifacts/`
- 默认产物：
  - `lift_demo.ckpt`
  - `lift_demo_norm.npz`
  - `visualizations/training_curves.html`
  - `visualizations/training_history.csv`
  - `visualizations/samples/sample_2d_latest.png`
  - `visualizations/samples/sample_3d_latest.html`
- 可选导出：`lift_demo.onnx`

## 5. 训练可视化链路

实现位置：`src/posementor/utils/training_viz.py`

每个 epoch 完成后会更新：

- 曲线页面：loss、MPJPE、学习率
- 样例页面：同一片段的 2D 输入和 3D 预测骨架
- 历史 CSV：便于后续做实验对比、报表和回归检查

## 6. 四机位扩展流程

```mermaid
flowchart LR
    A["四机位原始视频"] --> B["时序对齐"]
    B --> C["统一帧率/分辨率导出"]
    C --> D["YOLO2D提取"]
    D --> E["相机标定参数"]
    E --> F["三角化3D真值"]
    F --> G["质检与平滑"]
    G --> H["训练集合并"]
```

### 6.1 已实现

- 时序对齐：`prepare_multiview_dataset.py`
- 统一格式导出：`src/posementor/multiview/formatter.py`
- 四机位 2D 递归提取：`extract_pose_yolo11.py --recursive`
- 报告可视化：`visualize_multiview_report.py`

### 6.2 规划中

- 标定参数求解与版本管理
- 三角化 3D 真值生成
- 三角化质量自动评分与异常剔除

## 7. 运行清单

训练主链路最短命令：

```bash
uv run python download_and_prepare_aist.py --config configs/data.yaml --download --extract
uv run python extract_pose_aist2d.py --config configs/data.yaml
uv run python train_3d_lift_demo.py --config configs/train.yaml --export-onnx
```

## 8. 当前状态总览

| 模块 | 状态 |
|---|---|
| AIST++ 单视角训练主链路 | 已实现 |
| 训练曲线与样例可视化 | 已实现 |
| 自采四机位对齐与格式化 | 已实现 |
| 自采四机位三角化 3D 真值 | 规划中 |
| 多数据源统一注册与选择 | 已实现（dataset registry） |

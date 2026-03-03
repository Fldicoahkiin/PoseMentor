# 模型产物说明

训练 `train_3d_lift_demo.py` 后，会在 `artifact_dir`（默认 `artifacts/`）生成：

## 1) `lift_demo.ckpt`

- 形式：Lightning/PyTorch 权重文件
- 用途：主推理模型（2D -> 3D）

## 2) `lift_demo_norm.npz`

- 形式：NumPy 压缩文件
- 内容：`mean_2d`、`std_2d`
- 用途：推理阶段做同分布归一化，必须与训练一致

## 3) `lift_demo.onnx`（可选）

- 形式：ONNX 模型文件
- 用途：跨平台部署（ONNX Runtime / TensorRT / OpenVINO）

## 4) `visualizations/training_curves.html`

- 形式：Plotly HTML
- 用途：查看训练 loss、val loss、MPJPE 和学习率趋势

## 5) `visualizations/training_history.csv`

- 形式：CSV
- 用途：用于报表、对比实验、追踪收敛

---

## 最小使用方式

```bash
uv run python inference_pipeline_demo.py --source webcam --show --style gBR
```

如果要在其他数据集上训练，请使用：

```bash
uv run python train_3d_lift_demo.py \
  --config configs/train.yaml \
  --yolo2d-dir /path/to/2d \
  --gt3d-dir /path/to/3d \
  --artifact-dir /path/to/output
```


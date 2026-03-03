# Troubleshooting

## 1. `FileNotFoundError: 未找到 AIST++ 3D 注释文件`

现象：
- 运行 `download_and_prepare_aist.py --download --extract` 后仍报找不到 `data/raw/aistpp/annotations`

排查：

```bash
ls -la data/raw/aistpp
ls -la data/raw/aistpp/annotations
```

修复：
- 重新执行：

```bash
uv run python download_and_prepare_aist.py --config configs/data.yaml --download --extract
```

- 验证标注数量（fullset 应为 1408）：

```bash
uv run python -c "from pathlib import Path; from posementor.data.aist_loader import find_gt3d_files; print(len(find_gt3d_files(Path('data/raw/aistpp/annotations'))))"
```

## 2. 浏览器显示 `detail: "Not Found"`

现象：
- 打开后端地址时只看到 `Not Found`

确认接口：
- `http://127.0.0.1:8787/`
- `http://127.0.0.1:8787/health`
- `http://127.0.0.1:8787/api`
- `http://127.0.0.1:8787/api/health`

如果这些地址仍返回 404，请先确认后端进程是否在运行。

## 3. 无法连接 `127.0.0.1:7860`

现象：
- 浏览器提示连接失败

排查：
- 前端是否已启动：

```bash
cd frontend
pnpm dev --host 127.0.0.1 --port 7860
```

- 端口是否被占用：

```bash
lsof -i :7860
```

## 4. `找不到 3D lift 权重: artifacts/lift_demo.ckpt`

现象：
- 推理时提示未找到模型权重

修复：

```bash
uv run python train_3d_lift_demo.py --config configs/train.yaml --export-onnx
```

确保以下文件存在：
- `artifacts/lift_demo.ckpt`
- `artifacts/lift_demo_norm.npz`

## 5. `yolo11m-pose.pt` 不存在

现象：
- 关键点提取或推理启动失败

修复：
- 将 `yolo11m-pose.pt` 放到项目根目录，或在命令里传入 `--weights /absolute/path/to/yolo11m-pose.pt`
- 如果当前只需要先跑通训练，不依赖 YOLO，可改用：

```bash
uv run python extract_pose_aist2d.py --config configs/data.yaml
```

## 6. 训练慢或显存不足

建议：
- 在 `configs/train.yaml` 调小 `train.batch_size`
- 保持 `seq_len=81`，先不要增大
- 先跑子集验证流程，再跑全量

## 7. 摄像头无法打开

常见原因：
- 系统未授权终端/浏览器访问摄像头
- 设备被其他应用占用

修复：
- 关闭占用摄像头的应用
- 在系统隐私设置里给当前应用授权

## 8. 任务卡住或无输出

排查日志：
- `outputs/job_center/jobs.json`
- `outputs/job_center/logs/*.log`

后端日志建议直接看实时输出，便于定位子进程命令参数。

## 9. 训练完成但没看到曲线文件

检查目录：

```bash
ls -la artifacts/visualizations
```

正常应包含：
- `training_curves.html`
- `training_history.csv`

## 10. 后端返回 `未知 dataset_id`

现象：
- 调用任务接口时返回 `未知 dataset_id`

修复：
- 先检查 `configs/datasets.yaml` 是否存在该 `id`
- 确认请求里 `dataset_id` 拼写一致
- 可先调用：

```bash
curl http://127.0.0.1:8787/datasets
```

## 11. CLI 提示本地配置不存在

现象：
- 运行 `uv run posementor up` 或 `uv run posementor quickstart` 前未初始化配置

修复：

```bash
uv run posementor config
```

或：

```bash
uv run python scripts/config_setup.py
```

## 12. `status` 一直显示 stopped

排查顺序：

```bash
uv run posementor doctor
uv run posementor up
uv run posementor status
uv run posementor logs --service all --lines 120
```

如果 `doctor` 显示端口占用，请先释放对应端口或调整 `configs/local.yaml` 端口配置。

# Quickstart

## 0. 快速模式（推荐）

### macOS

```bash
cd /Users/mac/WorkSpace/Python_Project/posementor
uv run python scripts/config_setup.py
uv run posementor init
uv run posementor quickstart --epochs 1 --export-onnx --up
```

`init` 后可直接：

```bash
./posementor config
./posementor doctor
./posementor quickstart --epochs 1 --up
```

### Windows

```powershell
cd C:\path\to\posementor
uv run python scripts/config_setup.py
uv run posementor init
uv run posementor quickstart --epochs 1 --export-onnx --up
```

`init` 后可直接：

```powershell
posementor.exe config
posementor.exe doctor
posementor.exe quickstart --epochs 1 --up
```

常用管理命令：

```bash
uv run posementor doctor
uv run posementor status
uv run posementor logs --service all --lines 120
uv run posementor down
```

## 1. 前置依赖

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Node.js 20+
- [pnpm](https://pnpm.io/)

## 2. 安装依赖

### macOS

```bash
cd /Users/mac/WorkSpace/Python_Project/posementor
uv sync --group dev --group mac
```

### Windows

```powershell
cd C:\path\to\posementor
uv sync --group dev --group windows
```

### 前端依赖（两端通用）

```bash
cd frontend
pnpm install
```

## 3. 准备 AIST++ 数据

AIST++ 官方文件：
- [fullset.zip](https://storage.googleapis.com/aist_plusplus_public/20210308/fullset.zip)
- [video_list.txt](https://storage.googleapis.com/aist_plusplus_public/20121228/video_list.txt)

注释下载 + 解压 + 预处理：

```bash
uv run python download_and_prepare_aist.py --config configs/data.yaml --download --extract
```

可选：下载部分视频（需显式同意协议）

```bash
uv run python download_and_prepare_aist.py --config configs/data.yaml --download-videos --video-limit 120 --agree-aist-license --skip-preprocess
```

可选：验证 3D 标注文件数量（fullset 应为 1408）

```bash
uv run python -c "from pathlib import Path; from posementor.data.aist_loader import find_gt3d_files; print(len(find_gt3d_files(Path('data/raw/aistpp/annotations'))))"
```

## 4. 提取 2D 关键点

```bash
uv run python extract_pose_yolo11.py --weights yolo11m-pose.pt --config configs/data.yaml
```

若当前阶段优先验证训练链路，可直接使用 AIST++ 官方 2D 注释：

```bash
uv run python extract_pose_aist2d.py --config configs/data.yaml
```

## 5. 训练 3D Lift

```bash
uv run python train_3d_lift_demo.py --config configs/train.yaml --export-onnx
```

训练输出：
- `artifacts/lift_demo.ckpt`
- `artifacts/lift_demo_norm.npz`
- `artifacts/lift_demo.onnx`
- `artifacts/visualizations/training_curves.html`
- `artifacts/visualizations/training_history.csv`

## 6. 启动服务

前后端分离启动：
- 后端负责数据处理、训练、推理调度
- 前端负责交互界面与任务展示

### 终端 A：启动后端

```bash
uv run python backend_api.py
```

后端地址：`http://127.0.0.1:8787`

### 终端 B：启动前端

```bash
cd frontend
pnpm dev --host 127.0.0.1 --port 7860
```

前端地址：`http://127.0.0.1:7860`

## 7. 快速验证

健康检查：

```bash
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/api/health
curl http://127.0.0.1:8787/datasets
```

命令行推理：

```bash
uv run python inference_pipeline_demo.py --source webcam --show --style gBR
```

## 8. 扩展命令

离线评测：

```bash
uv run python evaluate_model_suite.py --input-dir data/raw/aistpp/videos --style gBR --max-videos 10 --output-csv outputs/eval/summary.csv
```

四机位预处理：

```bash
uv run python prepare_multiview_dataset.py --config configs/multiview.yaml --limit-sessions 20
```

四机位 YOLO2D 提取（和 AIST++ 同一处理方式）：

```bash
uv run python extract_pose_yolo11.py --config configs/data.yaml --video-root data/processed/multiview --out-dir data/processed/multiview_pose2d --recursive --weights yolo11m-pose.pt
```

四机位可视化报告：

```bash
uv run python visualize_multiview_report.py --manifest data/processed/multiview/multiview_manifest.csv
```

## 9. CLI 一体化命令

```bash
uv run posementor config --force
uv run posementor init
uv run posementor quickstart --epochs 1 --export-onnx --up
uv run posementor status
uv run posementor logs --service backend_api --lines 120
uv run posementor down

./posementor status
./posementor logs --service all --lines 120

uv run python posementor_cli.py prepare-aist --config configs/data.yaml --download --extract
uv run python posementor_cli.py extract-aist2d --config configs/data.yaml
uv run python posementor_cli.py train-lift --config configs/train.yaml --epochs 2
```

## 10. 自定义数据集扩展位（预留）

当前后端通过 `configs/datasets.yaml` 管理数据集注册信息。  
如果后续加入自采集数据，推荐先走这条方式：

1. 在 `configs/datasets.yaml` 新增一个 `dataset_id`
2. 提取 2D 到自定义目录（可复用 YOLO 脚本）  
3. 准备对应 3D 标签目录（单视角或四机位三角化结果）
4. 训练时用目录覆盖参数：

```bash
uv run python train_3d_lift_demo.py \
  --config configs/train.yaml \
  --yolo2d-dir /path/to/custom_2d \
  --gt3d-dir /path/to/custom_3d \
  --artifact-dir /path/to/custom_artifacts
```

## 11. Docker 启动（可选）

```bash
cd docker
docker compose up --build
```

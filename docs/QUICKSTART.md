# Quickstart

## 0. 前置依赖

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Node.js 20+
- [pnpm](https://pnpm.io/)

## 1. 安装依赖

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

## 2. 准备 AIST++ 数据

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

## 3. 提取 2D 关键点

```bash
uv run python extract_pose_yolo11.py --weights yolo11m-pose.pt --config configs/data.yaml
```

## 4. 训练 3D Lift

```bash
uv run python train_3d_lift_demo.py --config configs/train.yaml --export-onnx
```

训练输出：
- `artifacts/lift_demo.ckpt`
- `artifacts/lift_demo_norm.npz`
- `artifacts/lift_demo.onnx`

## 5. 启动服务

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

## 6. 快速验证

健康检查：

```bash
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/api/health
```

命令行推理：

```bash
uv run python inference_pipeline_demo.py --source webcam --show --style gBR
```

## 7. 扩展命令

离线评测：

```bash
uv run python evaluate_model_suite.py --input-dir data/raw/aistpp/videos --style gBR --max-videos 10 --output-csv outputs/eval/summary.csv
```

四机位预处理：

```bash
uv run python prepare_multiview_dataset.py --config configs/multiview.yaml --limit-sessions 20
```

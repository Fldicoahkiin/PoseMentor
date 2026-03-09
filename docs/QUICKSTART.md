# Quickstart

## 0. 快速模式（推荐）

统一脚本入口（macOS / Windows / Linux 都一致）：

```bash
cd <project_root>
./pm config
./pm init
./pm quickstart --epochs 1 --export-onnx --up
```

Windows PowerShell：

```powershell
cd <project_root>
.\pm.ps1 config
.\pm.ps1 init
.\pm.ps1 quickstart --epochs 1 --export-onnx --up
```

常用管理命令：

```bash
./pm doctor
./pm status
./pm logs --service all --lines 120
./pm down
```

`config` 会进入交互式 UI，可直接选择多机位下载 Profile：
- `mv3_quick`：c01,c02,c03 × 40 组
- `mv5_standard`：c01~c05 × 80 组
- `mv9_core`：c01~c09 × 120 组
- `mv9_full`：c01~c09 全量

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

脚本会自动识别 `annotations` 下的实际 3D 标注目录（含嵌套目录场景），无需手动搬运文件。

可选：下载部分视频（需显式同意协议）

```bash
uv run python download_and_prepare_aist.py --config configs/data.yaml --download-videos --group-limit 40 --camera-ids c01,c02,c03 --min-cameras-per-group 3 --agree-aist-license --skip-preprocess
```

全机位（c01~c09）全量下载：

```bash
uv run python download_and_prepare_aist.py --config configs/data.yaml --download-videos --group-limit 0 --camera-ids c01,c02,c03,c04,c05,c06,c07,c08,c09 --min-cameras-per-group 9 --agree-aist-license --skip-preprocess
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
uv run python extract_pose_aist2d.py --config configs/data.yaml --camera-index 1
```

说明：
- AIST++ 官方 2D 注释是 9 机位（`c01`~`c09`）打包在同一个 `cAll` 文件。
- 当前训练链路按“单视角输入”使用，因此每个动作样本会固定选取一个机位，输出一个 2D 序列。
- `--camera-index 1` 表示固定使用 `c01`；不传时默认按检测稳定性自动选机位。

## 数据集本地管理

- 前端入口：`http://127.0.0.1:7860/` → `训练数据集` 下拉
- 注册文件：`configs/datasets.yaml`
- 当前仅保留两类数据源：
  - AIST++：`data/raw/aistpp/videos`
  - 四机位：`data/raw/multiview`
- 新增四机位素材时，按 `configs/multiview.yaml` 目录结构放置。
- 后端接口：`POST /datasets/upsert`

## 5. 训练 3D Lift

```bash
uv run python train_3d_lift_demo.py --config configs/train.yaml --export-onnx
```

前端可视化路径：
- `http://127.0.0.1:7860/` 点击 `开始训练`
- 页面会自动轮询任务进度并联动同步片段播放（素材/2D/3D）

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

如果之前跑过旧版前端（如历史 Gradio 服务）导致端口混乱，先执行：

```bash
uv run posementor cleanup
```

训练工作台的四联同步播放依赖以下训练产物（训练一轮后自动生成）：

- `artifacts/visualizations/samples/sample_video_latest.mp4`
- `artifacts/visualizations/samples/sample_2d_latest.mp4`
- `artifacts/visualizations/samples/sample_3d_latest.mp4`

常用服务命令：

```bash
./pm start
./pm status
./pm stop
./pm restart
```

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
uv run python extract_pose_yolo11.py --config configs/data.yaml --video-root data/processed/multiview --out-dir data/processed/multiview/yolo2d --recursive --weights yolo11m-pose.pt
```

四机位三角化 3D 真值：

```bash
uv run python triangulate_multiview_dataset.py --config configs/multiview.yaml --calibration configs/calibration/fourview_template.yaml --limit-sessions 20
```

四机位可视化报告：

```bash
uv run python visualize_multiview_report.py --manifest data/processed/multiview/multiview_manifest.csv
```

## 9. CLI 一体化命令

```bash
./pm config
./pm config --plain --force --download-now
./pm init
./pm quickstart --download-videos --video-profile mv5_standard --epochs 1 --export-onnx --up
./pm status
./pm logs --service backend_api --lines 120
./pm down

uv run posementor config
uv run posementor quickstart --download-videos --video-profile mv9_core --epochs 1
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

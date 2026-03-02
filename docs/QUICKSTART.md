# 快速执行清单

## 1) 一键启动全栈服务

### macOS

```bash
cd /Users/mac/WorkSpace/Python_Project/posementor
./scripts/launch_macos.sh all
```

### Windows

```powershell
cd C:\path\to\posementor
powershell -ExecutionPolicy Bypass -File .\scripts\launch_windows.ps1 -Action all
```

入口地址：
- 在线系统：`http://127.0.0.1:7860`
- 管理后台：`http://127.0.0.1:7861`
- Backend API：`http://127.0.0.1:8787`

## 2) 一键数据准备

```bash
uv run python download_and_prepare_aist.py --config configs/data.yaml --download --extract
```

```bash
uv run python download_and_prepare_aist.py \
  --config configs/data.yaml \
  --download-videos \
  --video-limit 80 \
  --agree-aist-license \
  --skip-preprocess
```

```bash
uv run python extract_pose_yolo11.py --weights yolo11m-pose.pt --config configs/data.yaml
```

## 3) 训练 + 导出 + 测试

```bash
uv run python train_3d_lift_demo.py --config configs/train.yaml --export-onnx
```

```bash
uv run python evaluate_model_suite.py --input-dir data/raw/aistpp/videos --style gBR --max-videos 10 --output-csv outputs/eval/summary.csv
```

## 4) 四机位对齐与格式化（扩展）

```bash
uv run python prepare_multiview_dataset.py --config configs/multiview.yaml --limit-sessions 20
```

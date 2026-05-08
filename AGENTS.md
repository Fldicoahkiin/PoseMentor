# AGENTS.md — PoseMentor Development Guide

## Project Overview

PoseMentor 是基于 AIST++ 数据集的 AI 单摄像头舞蹈/体育动作实时评估与纠正系统。

**ML Pipeline:** YOLO11-Pose (2D) -> Kalman Smoothing -> PoseLiftTransformer (3D) -> DTW Alignment -> MPJPE/Angle Scoring -> TTS Feedback

## Architecture

| Layer | Tech | Entry Point |
|-------|------|-------------|
| Backend API | FastAPI 0.116+ | `backend_api.py` |
| Frontend | React 19 + Vite 8 + TypeScript | `frontend/src/main.tsx` |
| ML Package | PyTorch + Lightning | `src/posementor/` |
| CLI | Typer | `posementor_cli.py` |
| Infra | JobStore (JSON) + JobRunner (thread pool) | `src/posementor/infra/` |

## Directory Structure

```
backend_api.py              # FastAPI 后端，路由通过 APIRouter 挂载到 / 和 /api
posementor_cli.py           # CLI 入口
src/posementor/
  data/                     # AIST++ 数据加载与对齐
  models/lift_net.py        # PoseLiftTransformer
  pipeline/                 # 推理管线、预览渲染
  multiview/                # 多机位对齐、标定、三角测量
  infra/                    # JobStore + JobRunner
  utils/                    # I/O、关节定义、3D 数学、Kalman、评分、可视化、TTS
frontend/src/
  pages/DemoPage.tsx        # 主工作台（训练数据浏览 + 多视角同步播放）
  components/               # Pose2DViewport, Pose3DViewport, AlignmentInfoPanel
  hooks/                    # useSyncPlayback, useTrainingFollow, usePosePreview, useDatasetSelection, useSourceGroups
  lib/videoUtils.ts         # 公共辅助函数（seekVideo, pickMedian, formatBytes 等）
  lib/api.ts                # 后端 API 客户端和类型定义
configs/                    # YAML 配置（data, train, infer, multiview, datasets, standards）
tests/                      # pytest 测试
```

## Development Commands

```bash
# 全栈（推荐）
./pm up                                    # 启动前后端服务
./pm down                                  # 停止前后端服务
./pm restart                               # 重启
./pm status                                # 查看服务状态
./pm logs                                  # 查看服务日志

# CLI 其他命令
./pm config                                # 生成或更新本地配置
./pm init                                  # 安装依赖并初始化
./pm doctor                                # 检查运行环境和依赖
./pm cleanup                               # 清理僵尸进程和 PID

# 单独启动
uv run python backend_api.py              # 仅启动后端 API (0.0.0.0:8787)
cd frontend && pnpm dev                    # 仅启动前端开发服务器 (localhost:7860)

# 质量检查
uv run pytest tests/                       # 运行测试
uv run ruff check src/ backend_api.py      # Lint
uv run mypy src/                           # 类型检查
cd frontend && pnpm lint                   # 前端 ESLint
cd frontend && pnpm build                  # 前端生产构建
```

## Code Conventions

- **Python:** ruff (line-length=100, target py311), mypy strict, Python 3.11-3.12
- **TypeScript:** ESLint + strict mode, React 19, Vite 8
- **UI 文本:** 全中文 (zh-CN)
- **配置文件:** YAML under `configs/`
- **图标:** lucide-react, 不使用 emoji
- **测试:** pytest, 文件名 `tests/test_*.py`

## Key Patterns

### Job Lifecycle
`JobStore` 持久化到 `outputs/job_center/jobs.json`。`JobRunner` 用 `subprocess.Popen` 在后台线程执行。
- Status: `queued` -> `running` -> `succeeded` | `failed`
- 超时: 默认 7200s (`POSEMENTOR_JOB_TIMEOUT` env)
- 前端通过 `fetchJobProgress` 轮询日志解析进度

### Route Structure
所有业务路由定义在 `APIRouter` 上，挂载到 `/` 和 `/api` 两个前缀:
```python
app.include_router(router)
app.include_router(router, prefix="/api")
```

### Dataset Registry
`configs/datasets.yaml` + `configs/standards.yaml` 作为数据集和标准动作的注册表。

### CORS
通过 `POSEMENTOR_CORS_ORIGINS` 环境变量配置允许的 origins，默认: `http://localhost:7860,http://localhost:5173,http://127.0.0.1:7860,http://127.0.0.1:5173`

### Input Validation
所有 Pydantic Request Model 的路径字段通过 `_validate_safe_path` 校验：不以 `-` 开头、不含 `..`、仅允许安全字符。

### Frontend State Management
DemoPage 主工作台状态已全部拆分为独立 hooks（2099 -> 1321 行，-37%）:
- `useSyncPlayback` — 多视频同步播放、timer、seek
- `useTrainingFollow` — 训练进度轮询、完成检测
- `usePosePreview` — 骨架预览获取/缓存
- `useDatasetSelection` — dataset/standard/group 自动选择
- `useSourceGroups` — 源视频分组逻辑

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSEMENTOR_JOB_WORKERS` | `1` | 后台任务并发数 |
| `POSEMENTOR_JOB_TIMEOUT` | `7200` | 单个作业超时秒数 |
| `POSEMENTOR_CORS_ORIGINS` | `localhost:7860,5173` | CORS 允许的 origins |
| `POSEMENTOR_PREVIEW_YOLO_WEIGHTS` | `yolo11m-pose.pt` | 预览用 YOLO 权重 |
| `POSEMENTOR_PREVIEW_YOLO_CONF` | `0.35` | 预览用 YOLO 置信度阈值 |
| `VITE_BACKEND_URL` | `http://127.0.0.1:8787` | 前端连接后端的 URL |

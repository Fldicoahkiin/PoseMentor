# 常见问题与优化策略

## 精度与稳定性

1. 先固定舞种模板（如 `gBR`），降低 DTW 错配概率。  
2. 保持训练与推理归一化参数一致（`lift_demo_norm.npz`）。  
3. 在对比前做髋中心对齐，减少平移误差干扰。  
4. 用 Kalman 平滑 2D 关键点，抑制抖动。  
5. 训练中启用速度损失，降低时序闪烁。  
6. 评分阈值与语音阈值分开调参，避免过度提示。

## 实时性能

1. 输入分辨率先用 `640x480` 或 `960x540`。  
2. YOLO 权重优先 `yolo11m-pose.pt`。  
3. 若 GPU 紧张，降低 `seq_len` 和输入分辨率。  
4. 优先使用本地 SSD，减少视频 IO 抖动。

## 数据与多机位

1. 若对齐误差大，检查 `configs/multiview.yaml` 的 `motion_ratio`。  
2. 四机位 session 必须包含 `front/left/right/back` 四个文件。  
3. 若视频起始静止时间长，提高 `scan_frames`。  
4. 对齐后可先人工抽查 `session_meta.json`。

## 后台任务

1. 任务状态文件：`outputs/job_center/jobs.json`。  
2. 任务日志目录：`outputs/job_center/logs/`。  
3. 管理后台连接失败先检查 `http://127.0.0.1:8787/health`。  
4. 若任务卡住，先看日志最后输出再决定是否重启任务。

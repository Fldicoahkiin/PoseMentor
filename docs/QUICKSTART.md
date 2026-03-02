# 2天快速出Demo执行清单

## Day 1（数据+模型）

1. `uv sync` 安装环境。  
2. 一键下载并解压注释：`uv run python download_and_prepare_aist.py --config configs/data.yaml --download --extract --skip-preprocess`。  
3. 一键下载少量视频（先 30~80 段）：`uv run python download_and_prepare_aist.py --config configs/data.yaml --download-videos --video-limit 80 --agree-aist-license --skip-preprocess`。  
4. 运行 `download_and_prepare_aist.py` 产出 `gt3d`。  
5. 运行 `extract_pose_yolo11.py` 产出 `yolo2d`。  
6. 跑 `train_3d_lift_demo.py`（先 10~15 epoch 看收敛）。

## Day 2（在线演示）

1. 启动 `app_demo.py`，先跑视频上传路径验证结果。  
2. 再切换摄像头流式路径，检查实时 FPS。  
3. 录制 1~2 个关键动作片段，生成输出视频。  
4. 调整评分阈值和语音文案，确保结果可解释。

## 最小演示配置建议

- 分辨率：`640x480` 或 `960x540`
- 模型：`yolo11m-pose.pt`
- 序列长度：`81`
- 推理阈值：`det_conf=0.35`
- 错误关节阈值：`60mm`

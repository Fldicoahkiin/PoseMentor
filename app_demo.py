#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from posementor.pipeline.realtime_coach import CoachConfig, RealtimeDanceCoach
from posementor.utils.io import ensure_dir

CUSTOM_CSS = """
:root {
  --bg-0: #f7f1e7;
  --bg-1: #efe3d1;
  --card: #fffaf1;
  --line: #e7d8bf;
  --text: #2e2a24;
  --muted: #6b6358;
  --accent: #c96e16;
  --accent-2: #8f3f15;
}

.gradio-container {
  background: radial-gradient(circle at top right, #fff8ed 0%, var(--bg-0) 45%, var(--bg-1) 100%);
  color: var(--text);
  font-family: "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
}

#app-hero {
  background: linear-gradient(120deg, #fff2df 0%, #ffe8cc 54%, #fbdabc 100%);
  border: 1px solid #eecba2;
  border-radius: 16px;
  padding: 16px 20px;
  box-shadow: 0 10px 28px rgba(132, 84, 24, 0.12);
  margin-bottom: 10px;
}

#app-hero h1 {
  margin: 0;
  font-size: 24px;
  color: #201a14;
}

#app-hero p {
  margin: 6px 0 0;
  color: #4e463d;
}

button.primary,
button.lg.primary {
  background: linear-gradient(110deg, var(--accent) 0%, var(--accent-2) 100%) !important;
  border: none !important;
  color: #fff !important;
}

.block {
  border-radius: 12px !important;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PoseMentor AIST++ 快速 Demo")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--yolo-weights", type=str, default="yolo11m-pose.pt")
    parser.add_argument("--lift-ckpt", type=str, default="artifacts/lift_demo.ckpt")
    parser.add_argument("--norm", type=str, default="artifacts/lift_demo_norm.npz")
    parser.add_argument("--template-dir", type=str, default="data/processed/aistpp/gt3d")
    parser.add_argument("--tts-engine", type=str, default="edge_tts", choices=["edge_tts", "gtts"])
    return parser.parse_args()


def build_app(coach: RealtimeDanceCoach | None, init_error: str | None) -> gr.Blocks:
    styles = coach.available_styles() if coach is not None else []
    styles = styles or ["gBR"]

    def reset_stream_state() -> tuple[dict, str, None]:
        if coach is not None:
            coach.reset()
        return {"frames": 0}, "已重置缓存，请重新开始动作。", None

    def process_webcam_frame(
        frame_rgb: np.ndarray,
        style: str,
        state: dict,
    ) -> tuple[np.ndarray, dict, str, str | None, object, dict]:
        if coach is None:
            empty = np.zeros((480, 640, 3), dtype=np.uint8)
            return (
                empty,
                {},
                f"模型未就绪：{init_error or '请先完成训练'}",
                None,
                None,
                state or {"frames": 0},
            )

        if frame_rgb is None:
            empty = np.zeros((480, 640, 3), dtype=np.uint8)
            return empty, {}, "请打开摄像头", None, None, state

        result = coach.process_frame(frame_rgb=frame_rgb, style=style)
        state = state or {"frames": 0}
        state["frames"] = int(state.get("frames", 0)) + 1

        score_panel = {
            "score": round(float(result["score"]), 2),
            "mpjpe_mm": round(float(result["mpjpe_mm"]), 2),
            "angle_error_deg": round(float(result["angle_error_deg"]), 2),
            "bad_joints": result["bad_joints"],
            "frames": state["frames"],
        }

        advice = str(result["advice"])
        audio = result["audio"]
        return result["annotated"], score_panel, advice, audio, result["plot"], state

    def process_video_file(video_path: str, style: str, progress: gr.Progress = gr.Progress()) -> tuple[str, str, object, str | None]:
        if coach is None:
            raise gr.Error(f"模型未就绪：{init_error or '请先完成训练'}")
        if not video_path:
            raise gr.Error("请先上传或录制一段视频")

        coach.reset()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise gr.Error(f"无法打开视频: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        out_dir = ensure_dir(Path("outputs") / "gradio")
        out_video = out_dir / f"demo_{int(time.time())}.mp4"
        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        scores: list[float] = []
        mpjpe_vals: list[float] = []
        angle_vals: list[float] = []
        last_plot = None
        last_audio = None
        last_advice = ""

        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = coach.process_frame(frame_rgb=frame_rgb, style=style)
            vis_bgr = cv2.cvtColor(result["annotated"], cv2.COLOR_RGB2BGR)
            writer.write(vis_bgr)

            scores.append(float(result["score"]))
            mpjpe_vals.append(float(result["mpjpe_mm"]))
            angle_vals.append(float(result["angle_error_deg"]))
            last_plot = result["plot"]
            if result["audio"]:
                last_audio = result["audio"]
            last_advice = str(result["advice"])

            idx += 1
            if total > 0 and idx % 5 == 0:
                progress((idx / total), desc=f"处理中 {idx}/{total}")

        cap.release()
        writer.release()

        if not scores:
            raise gr.Error("视频未读取到有效帧")

        summary_md = (
            f"### Demo结果\n"
            f"- 平均得分: **{np.mean(scores):.1f}**\n"
            f"- 平均 MPJPE: **{np.mean(mpjpe_vals):.1f} mm**\n"
            f"- 平均角度误差: **{np.mean(angle_vals):.1f}°**\n"
            f"- 最后建议: **{last_advice}**"
        )

        return str(out_video), summary_md, last_plot, last_audio

    with gr.Blocks(title="PoseMentor Demo") as app:
        gr.HTML(
            """
<div id="app-hero">
  <h1>PoseMentor 单摄像头动作教学系统</h1>
  <p>YOLO11-Pose -> 3D Lift -> DTW 对齐 -> 实时打分 / 关节高亮 / 语音纠错</p>
</div>
"""
        )
        if init_error:
            gr.Markdown(f"⚠️ 模型初始化失败：`{init_error}`")

        with gr.Tab("实时摄像头（实时输入）"):
            with gr.Row():
                style_dropdown = gr.Dropdown(choices=styles, value=styles[0], label="舞种模板")
                reset_btn = gr.Button("重置缓存")

            with gr.Row():
                webcam_input = gr.Image(
                    sources=["webcam"],
                    type="numpy",
                    streaming=True,
                    label="摄像头输入",
                )
                annotated_output = gr.Image(type="numpy", label="实时骨架与纠错高亮")

            score_json = gr.JSON(label="实时评分面板")
            advice_text = gr.Textbox(label="纠错建议", lines=2)
            audio_output = gr.Audio(label="语音提示", autoplay=True)
            skeleton_plot = gr.Plot(label="3D骨骼对比")
            stream_state = gr.State({"frames": 0})

            webcam_input.stream(
                fn=process_webcam_frame,
                inputs=[webcam_input, style_dropdown, stream_state],
                outputs=[
                    annotated_output,
                    score_json,
                    advice_text,
                    audio_output,
                    skeleton_plot,
                    stream_state,
                ],
                stream_every=0.04,
            )

            reset_btn.click(
                fn=reset_stream_state,
                inputs=None,
                outputs=[stream_state, advice_text, audio_output],
            )

        with gr.Tab("视频上传/录制 Demo"):
            video_input = gr.Video(
                sources=["upload", "webcam"],
                label="输入视频",
            )
            style_for_video = gr.Dropdown(choices=styles, value=styles[0], label="舞种模板")
            run_btn = gr.Button("开始分析", variant="primary")

            video_output = gr.Video(label="输出视频（红绿高亮+打分）")
            summary_output = gr.Markdown()
            plot_output = gr.Plot(label="最后一帧 3D 骨骼对比")
            audio_for_video = gr.Audio(label="最后一次语音建议")

            run_btn.click(
                fn=process_video_file,
                inputs=[video_input, style_for_video],
                outputs=[video_output, summary_output, plot_output, audio_for_video],
            )

    return app


def main() -> None:
    args = parse_args()

    coach: RealtimeDanceCoach | None = None
    init_error: str | None = None
    try:
        coach = RealtimeDanceCoach(
            CoachConfig(
                yolo_weights=args.yolo_weights,
                lift_checkpoint=args.lift_ckpt,
                norm_file=args.norm,
                template_dir=args.template_dir,
                tts_engine=args.tts_engine,
            )
        )
    except Exception as exc:  # noqa: BLE001
        init_error = str(exc)

    app = build_app(coach, init_error)

    try:
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            theme=gr.themes.Base(),
            css=CUSTOM_CSS,
        )
    except OSError as exc:
        if "Cannot find empty port" not in str(exc):
            raise
        print(f"[WARN] 端口 {args.port} 不可用，自动切换到随机可用端口")
        app.launch(
            server_name=args.host,
            server_port=None,
            share=args.share,
            theme=gr.themes.Base(),
            css=CUSTOM_CSS,
        )


if __name__ == "__main__":
    main()

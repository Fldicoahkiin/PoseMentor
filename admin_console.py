#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

import gradio as gr
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PoseMentor 后台管理前端")
    parser.add_argument("--api", type=str, default="http://127.0.0.1:8787")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _post_json(api_base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}{path}"
    response = requests.post(url, json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def _get_json(api_base: str, path: str) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}{path}"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json()


def build_app(default_api: str) -> gr.Blocks:
    with gr.Blocks(title="PoseMentor 管理后台", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
# PoseMentor 管理后台
统一管理数据下载、关键点提取、训练、四机位处理、模型测试与任务日志。
"""
        )

        api_base = gr.Textbox(label="Backend API 地址", value=default_api)

        with gr.Row():
            health_btn = gr.Button("检查后端状态")
            health_text = gr.Textbox(label="状态")

        def check_health(api: str) -> str:
            try:
                data = _get_json(api, "/health")
                return json.dumps(data, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001
                return f"后端不可用: {exc}"

        health_btn.click(check_health, inputs=[api_base], outputs=[health_text])

        with gr.Tab("数据准备"):
            data_cfg = gr.Textbox(label="data.yaml 路径", value="configs/data.yaml")
            with gr.Row():
                download_ann = gr.Checkbox(label="下载注释", value=True)
                extract_ann = gr.Checkbox(label="解压注释", value=True)
                download_videos = gr.Checkbox(label="下载视频", value=False)
                agree_license = gr.Checkbox(label="已同意 AIST++ 许可", value=False)

            with gr.Row():
                video_limit = gr.Number(label="视频下载上限", value=120, precision=0)
                preprocess_limit = gr.Number(label="预处理序列上限", value=0, precision=0)

            run_data_btn = gr.Button("启动数据任务")
            data_job_id = gr.Textbox(label="任务 ID")

            def start_data_task(
                api: str,
                cfg_path: str,
                dl_ann: bool,
                ex_ann: bool,
                dl_videos: bool,
                agree: bool,
                v_limit: float,
                p_limit: float,
            ) -> str:
                payload = {
                    "config": cfg_path,
                    "download_annotations": dl_ann,
                    "extract_annotations": ex_ann,
                    "download_videos": dl_videos,
                    "video_limit": int(v_limit),
                    "agree_license": agree,
                    "preprocess_limit": int(p_limit),
                }
                data = _post_json(api, "/jobs/data/prepare", payload)
                return str(data["job_id"])

            run_data_btn.click(
                start_data_task,
                inputs=[
                    api_base,
                    data_cfg,
                    download_ann,
                    extract_ann,
                    download_videos,
                    agree_license,
                    video_limit,
                    preprocess_limit,
                ],
                outputs=[data_job_id],
            )

        with gr.Tab("关键点提取"):
            ex_cfg = gr.Textbox(label="data.yaml 路径", value="configs/data.yaml")
            ex_weights = gr.Textbox(label="YOLO 权重", value="yolo11m-pose.pt")
            ex_conf = gr.Slider(label="检测阈值", minimum=0.05, maximum=0.8, value=0.35, step=0.01)
            ex_max = gr.Number(label="视频数量上限", value=0, precision=0)
            run_extract_btn = gr.Button("启动关键点提取")
            extract_job_id = gr.Textbox(label="任务 ID")

            def start_extract_task(
                api: str,
                cfg_path: str,
                weights: str,
                conf: float,
                max_videos: float,
            ) -> str:
                payload = {
                    "config": cfg_path,
                    "weights": weights,
                    "conf": conf,
                    "max_videos": int(max_videos),
                }
                data = _post_json(api, "/jobs/pose/extract", payload)
                return str(data["job_id"])

            run_extract_btn.click(
                start_extract_task,
                inputs=[api_base, ex_cfg, ex_weights, ex_conf, ex_max],
                outputs=[extract_job_id],
            )

        with gr.Tab("训练管理"):
            train_cfg = gr.Textbox(label="train.yaml 路径", value="configs/train.yaml")
            export_onnx = gr.Checkbox(label="导出 ONNX", value=True)
            run_train_btn = gr.Button("启动训练")
            train_job_id = gr.Textbox(label="任务 ID")

            def start_train_task(api: str, cfg_path: str, need_onnx: bool) -> str:
                payload = {"config": cfg_path, "export_onnx": need_onnx}
                data = _post_json(api, "/jobs/train", payload)
                return str(data["job_id"])

            run_train_btn.click(
                start_train_task,
                inputs=[api_base, train_cfg, export_onnx],
                outputs=[train_job_id],
            )

        with gr.Tab("四机位处理"):
            mv_cfg = gr.Textbox(label="multiview.yaml 路径", value="configs/multiview.yaml")
            mv_limit = gr.Number(label="处理 session 上限", value=0, precision=0)
            run_mv_btn = gr.Button("启动四机位对齐与格式化")
            mv_job_id = gr.Textbox(label="任务 ID")

            def start_multiview_task(api: str, cfg_path: str, limit_sessions: float) -> str:
                payload = {"config": cfg_path, "limit_sessions": int(limit_sessions)}
                data = _post_json(api, "/jobs/multiview/prepare", payload)
                return str(data["job_id"])

            run_mv_btn.click(
                start_multiview_task,
                inputs=[api_base, mv_cfg, mv_limit],
                outputs=[mv_job_id],
            )

        with gr.Tab("模型测试"):
            eval_input = gr.Textbox(label="输入视频目录", value="data/raw/aistpp/videos")
            eval_style = gr.Textbox(label="舞种模板", value="gBR")
            eval_max = gr.Number(label="测试视频上限", value=10, precision=0)
            eval_csv = gr.Textbox(label="输出报告 CSV", value="outputs/eval/summary.csv")
            run_eval_btn = gr.Button("启动模型测试")
            eval_job_id = gr.Textbox(label="任务 ID")

            def start_eval_task(
                api: str,
                input_dir: str,
                style: str,
                max_videos: float,
                output_csv: str,
            ) -> str:
                payload = {
                    "input_dir": input_dir,
                    "style": style,
                    "max_videos": int(max_videos),
                    "output_csv": output_csv,
                }
                data = _post_json(api, "/jobs/evaluate", payload)
                return str(data["job_id"])

            run_eval_btn.click(
                start_eval_task,
                inputs=[api_base, eval_input, eval_style, eval_max, eval_csv],
                outputs=[eval_job_id],
            )

        with gr.Tab("任务中心"):
            refresh_btn = gr.Button("刷新任务列表")
            jobs_table = gr.Dataframe(
                headers=["job_id", "name", "status", "created_at", "started_at", "finished_at"],
                datatype=["str", "str", "str", "number", "number", "number"],
                label="任务列表",
            )

            log_job_id = gr.Textbox(label="查看日志的任务 ID")
            read_log_btn = gr.Button("读取日志")
            log_text = gr.Textbox(label="任务日志", lines=22)

            def refresh_jobs(api: str) -> list[list[Any]]:
                jobs = _get_json(api, "/jobs")["jobs"]
                rows = []
                for item in jobs:
                    rows.append(
                        [
                            item["job_id"],
                            item["name"],
                            item["status"],
                            item["created_at"],
                            item["started_at"],
                            item["finished_at"],
                        ]
                    )
                return rows

            def read_log(api: str, job_id: str) -> str:
                if not job_id:
                    return "请输入任务 ID"
                try:
                    return _get_json(api, f"/jobs/{job_id}/log")["log"]
                except Exception as exc:  # noqa: BLE001
                    return f"读取日志失败: {exc}"

            refresh_btn.click(refresh_jobs, inputs=[api_base], outputs=[jobs_table])
            read_log_btn.click(read_log, inputs=[api_base, log_job_id], outputs=[log_text])

    return app


def main() -> None:
    args = parse_args()
    app = build_app(default_api=args.api)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any

import gradio as gr
import plotly.graph_objects as go
import requests

CUSTOM_CSS = """
:root {
  --bg-0: #f6f2ea;
  --bg-1: #efe7da;
  --card: #fffdf8;
  --line: #e4d7c5;
  --text: #2d2a26;
  --muted: #6f665a;
  --accent: #d97706;
  --accent-2: #9a3412;
  --ok: #2f855a;
  --warn: #b45309;
  --err: #c2410c;
}

.gradio-container {
  background: radial-gradient(circle at top right, #fff8ee 0%, var(--bg-0) 42%, var(--bg-1) 100%);
  color: var(--text);
  font-family: "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
}

#hero-panel {
  background: linear-gradient(120deg, #fff6e7 0%, #ffeeda 48%, #fde5cf 100%);
  border: 1px solid #efcfa9;
  border-radius: 16px;
  padding: 18px 22px;
  box-shadow: 0 12px 36px rgba(153, 95, 26, 0.12);
}

#hero-panel h1 {
  margin: 0;
  font-size: 26px;
  letter-spacing: 0.4px;
  color: #1f1a14;
}

#hero-panel p {
  margin: 8px 0 0;
  color: #4f463f;
  font-size: 14px;
}

.dashboard-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 8px 28px rgba(76, 58, 33, 0.08);
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(120px, 1fr));
  gap: 12px;
}

.kpi-item {
  background: #fff9ef;
  border: 1px solid #edd7bd;
  border-radius: 12px;
  padding: 10px 12px;
}

.kpi-label {
  color: var(--muted);
  font-size: 12px;
}

.kpi-value {
  color: #2c241d;
  font-weight: 700;
  font-size: 22px;
  margin-top: 4px;
}

.section-title {
  font-size: 16px;
  font-weight: 700;
  color: #2e251b;
  margin-bottom: 10px;
}

button.primary,
button.lg.primary {
  background: linear-gradient(120deg, var(--accent) 0%, var(--accent-2) 100%) !important;
  border: none !important;
  color: #fff !important;
}

button.secondary {
  border-color: #d6b992 !important;
  color: #5b4424 !important;
}

textarea, input {
  border-radius: 10px !important;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PoseMentor 后台管理前端")
    parser.add_argument("--api", type=str, default="http://127.0.0.1:8787")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _candidate_urls(api_base: str, path: str) -> list[str]:
    base = api_base.strip().rstrip("/")
    suffix = "/" + path.strip().lstrip("/")

    candidates = [f"{base}{suffix}"]
    if base.endswith("/api"):
        base_without_api = base[: -len("/api")]
        candidates.append(f"{base_without_api}{suffix}")
    elif not suffix.startswith("/api/") and suffix != "/api":
        candidates.append(f"{base}/api{suffix}")

    seen: set[str] = set()
    dedup: list[str] = []
    for url in candidates:
        if url not in seen:
            seen.add(url)
            dedup.append(url)
    return dedup


def _post_json(api_base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for url in _candidate_urls(api_base, path):
        try:
            response = requests.post(url, json=payload, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    assert last_error is not None
    raise last_error


def _get_json(api_base: str, path: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for url in _candidate_urls(api_base, path):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    assert last_error is not None
    raise last_error


def _format_time(ts: float | None) -> str:
    if ts is None or ts <= 0:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S")


def _summarize_jobs(jobs: list[dict[str, Any]]) -> dict[str, int | float]:
    total = len(jobs)
    running = sum(1 for j in jobs if j["status"] == "running")
    queued = sum(1 for j in jobs if j["status"] == "queued")
    success = sum(1 for j in jobs if j["status"] == "success")
    failed = sum(1 for j in jobs if j["status"] == "failed")
    success_rate = (success / max(1, success + failed)) * 100.0

    return {
        "total": total,
        "running": running,
        "queued": queued,
        "success": success,
        "failed": failed,
        "success_rate": round(success_rate, 1),
    }


def _build_overview_html(stats: dict[str, int | float]) -> str:
    return f"""
<div class="dashboard-card">
  <div class="section-title">任务总览</div>
  <div class="kpi-grid">
    <div class="kpi-item"><div class="kpi-label">总任务</div><div class="kpi-value">{stats['total']}</div></div>
    <div class="kpi-item"><div class="kpi-label">运行中</div><div class="kpi-value">{stats['running']}</div></div>
    <div class="kpi-item"><div class="kpi-label">排队中</div><div class="kpi-value">{stats['queued']}</div></div>
    <div class="kpi-item"><div class="kpi-label">成功</div><div class="kpi-value">{stats['success']}</div></div>
    <div class="kpi-item"><div class="kpi-label">失败</div><div class="kpi-value">{stats['failed']}</div></div>
  </div>
  <div style="margin-top: 10px; color: #6f665a; font-size: 13px;">成功率（success / (success + failed)）：{stats['success_rate']}%</div>
</div>
"""


def _build_status_figure(stats: dict[str, int | float]) -> go.Figure:
    labels = ["queued", "running", "success", "failed"]
    values = [int(stats["queued"]), int(stats["running"]), int(stats["success"]), int(stats["failed"])]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.56,
                marker={"colors": ["#f4a261", "#2a9d8f", "#43aa8b", "#e76f51"]},
                textinfo="label+value",
            )
        ]
    )
    fig.update_layout(
        title="任务状态分布",
        margin={"l": 10, "r": 10, "t": 48, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_timeline_figure(jobs: list[dict[str, Any]]) -> go.Figure:
    if not jobs:
        fig = go.Figure()
        fig.add_annotation(text="暂无任务记录", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="任务时长分布")
        return fig

    ordered = list(reversed(jobs[:16]))
    y_labels: list[str] = []
    durations: list[float] = []
    colors: list[str] = []

    color_map = {
        "queued": "#f4a261",
        "running": "#2a9d8f",
        "success": "#43aa8b",
        "failed": "#e76f51",
    }

    for item in ordered:
        created = float(item.get("created_at") or 0.0)
        started = float(item.get("started_at") or created)
        finished = item.get("finished_at")
        right = float(finished) if finished else max(started, created)
        duration = max(0.0, right - started)

        y_labels.append(f"{item['name']}#{item['job_id'][:6]}")
        durations.append(duration)
        colors.append(color_map.get(item["status"], "#9c6644"))

    fig = go.Figure(
        data=[
            go.Bar(
                x=durations,
                y=y_labels,
                orientation="h",
                marker={"color": colors},
                hovertemplate="%{y}<br>持续: %{x:.2f}s<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="最近任务时长（秒）",
        margin={"l": 10, "r": 10, "t": 48, "b": 10},
        yaxis={"automargin": True},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_table_rows(jobs: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in jobs:
        rows.append(
            [
                item["job_id"],
                item["name"],
                item["status"],
                _format_time(item.get("created_at")),
                _format_time(item.get("started_at")),
                _format_time(item.get("finished_at")),
            ]
        )
    return rows


def _job_option_list(jobs: list[dict[str, Any]]) -> list[str]:
    return [item["job_id"] for item in jobs]


def build_app(default_api: str) -> gr.Blocks:
    with gr.Blocks(title="PoseMentor 管理后台", theme=gr.themes.Base(), css=CUSTOM_CSS) as app:
        gr.HTML(
            """
<div id="hero-panel">
  <h1>PoseMentor 管理后台</h1>
  <p>统一管理数据准备、关键点提取、训练、多机位处理、模型测试与任务日志。</p>
</div>
"""
        )

        with gr.Row():
            api_base = gr.Textbox(label="Backend API 地址", value=default_api, scale=5)
            health_btn = gr.Button("连接检测", variant="secondary", scale=1)
            health_text = gr.Textbox(label="后端状态", scale=2)

        def check_health(api: str) -> str:
            try:
                data = _get_json(api, "/health")
                return f"连接正常: {data}"
            except Exception as exc:  # noqa: BLE001
                return f"连接失败: {exc}"

        health_btn.click(check_health, inputs=[api_base], outputs=[health_text])

        with gr.Tab("总览看板"):
            with gr.Row():
                refresh_btn = gr.Button("刷新看板", variant="primary")
                auto_refresh_tip = gr.Markdown("自动刷新：每 8 秒更新一次")

            overview_html = gr.HTML()
            with gr.Row():
                status_plot = gr.Plot(label="状态分布", scale=1)
                timeline_plot = gr.Plot(label="任务时长", scale=2)

            jobs_table = gr.Dataframe(
                headers=["job_id", "name", "status", "created_at", "started_at", "finished_at"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                label="任务列表",
            )

            with gr.Row():
                job_selector = gr.Dropdown(label="选择任务查看详情", choices=[], scale=4)
                detail_btn = gr.Button("查看详情", variant="secondary", scale=1)

            job_detail = gr.JSON(label="任务详情")
            log_text = gr.Textbox(label="任务日志", lines=20)

            def refresh_dashboard(api: str) -> tuple[str, Any, Any, list[list[str]], dict[str, Any]]:
                try:
                    jobs = _get_json(api, "/jobs").get("jobs", [])
                except Exception as exc:  # noqa: BLE001
                    empty_stats = {
                        "total": 0,
                        "running": 0,
                        "queued": 0,
                        "success": 0,
                        "failed": 0,
                        "success_rate": 0.0,
                    }
                    return (
                        _build_overview_html(empty_stats),
                        _build_status_figure(empty_stats),
                        _build_timeline_figure([]),
                        [],
                        gr.update(choices=[], value=None),
                    )

                stats = _summarize_jobs(jobs)
                return (
                    _build_overview_html(stats),
                    _build_status_figure(stats),
                    _build_timeline_figure(jobs),
                    _build_table_rows(jobs),
                    gr.update(choices=_job_option_list(jobs), value=_job_option_list(jobs)[0] if jobs else None),
                )

            def view_job_detail(api: str, job_id: str) -> tuple[dict[str, Any], str]:
                if not job_id:
                    return {"message": "请选择任务"}, ""
                try:
                    detail = _get_json(api, f"/jobs/{job_id}")
                    logs = _get_json(api, f"/jobs/{job_id}/log").get("log", "")
                    return detail, logs
                except Exception as exc:  # noqa: BLE001
                    return {"error": str(exc)}, ""

            refresh_btn.click(
                refresh_dashboard,
                inputs=[api_base],
                outputs=[overview_html, status_plot, timeline_plot, jobs_table, job_selector],
            )
            detail_btn.click(view_job_detail, inputs=[api_base, job_selector], outputs=[job_detail, log_text])

            dashboard_timer = gr.Timer(8.0)
            dashboard_timer.tick(
                refresh_dashboard,
                inputs=[api_base],
                outputs=[overview_html, status_plot, timeline_plot, jobs_table, job_selector],
            )

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

            run_data_btn = gr.Button("启动数据任务", variant="primary")
            data_result = gr.Textbox(label="启动结果")

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
                return f"任务已创建: {data['job_id']}"

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
                outputs=[data_result],
            )

        with gr.Tab("关键点提取"):
            ex_cfg = gr.Textbox(label="data.yaml 路径", value="configs/data.yaml")
            ex_weights = gr.Textbox(label="YOLO 权重", value="yolo11m-pose.pt")
            ex_conf = gr.Slider(label="检测阈值", minimum=0.05, maximum=0.8, value=0.35, step=0.01)
            ex_max = gr.Number(label="视频数量上限", value=0, precision=0)
            run_extract_btn = gr.Button("启动关键点提取", variant="primary")
            extract_result = gr.Textbox(label="启动结果")

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
                return f"任务已创建: {data['job_id']}"

            run_extract_btn.click(
                start_extract_task,
                inputs=[api_base, ex_cfg, ex_weights, ex_conf, ex_max],
                outputs=[extract_result],
            )

        with gr.Tab("训练管理"):
            train_cfg = gr.Textbox(label="train.yaml 路径", value="configs/train.yaml")
            export_onnx = gr.Checkbox(label="导出 ONNX", value=True)
            run_train_btn = gr.Button("启动训练", variant="primary")
            train_result = gr.Textbox(label="启动结果")

            def start_train_task(api: str, cfg_path: str, need_onnx: bool) -> str:
                payload = {"config": cfg_path, "export_onnx": need_onnx}
                data = _post_json(api, "/jobs/train", payload)
                return f"任务已创建: {data['job_id']}"

            run_train_btn.click(
                start_train_task,
                inputs=[api_base, train_cfg, export_onnx],
                outputs=[train_result],
            )

        with gr.Tab("四机位处理"):
            mv_cfg = gr.Textbox(label="multiview.yaml 路径", value="configs/multiview.yaml")
            mv_limit = gr.Number(label="处理 session 上限", value=0, precision=0)
            run_mv_btn = gr.Button("启动四机位对齐与格式化", variant="primary")
            mv_result = gr.Textbox(label="启动结果")

            def start_multiview_task(api: str, cfg_path: str, limit_sessions: float) -> str:
                payload = {"config": cfg_path, "limit_sessions": int(limit_sessions)}
                data = _post_json(api, "/jobs/multiview/prepare", payload)
                return f"任务已创建: {data['job_id']}"

            run_mv_btn.click(
                start_multiview_task,
                inputs=[api_base, mv_cfg, mv_limit],
                outputs=[mv_result],
            )

        with gr.Tab("模型测试"):
            eval_input = gr.Textbox(label="输入视频目录", value="data/raw/aistpp/videos")
            eval_style = gr.Textbox(label="舞种模板", value="gBR")
            eval_max = gr.Number(label="测试视频上限", value=10, precision=0)
            eval_csv = gr.Textbox(label="输出报告 CSV", value="outputs/eval/summary.csv")
            run_eval_btn = gr.Button("启动模型测试", variant="primary")
            eval_result = gr.Textbox(label="启动结果")

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
                return f"任务已创建: {data['job_id']}"

            run_eval_btn.click(
                start_eval_task,
                inputs=[api_base, eval_input, eval_style, eval_max, eval_csv],
                outputs=[eval_result],
            )

    return app


def main() -> None:
    args = parse_args()
    app = build_app(default_api=args.api)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

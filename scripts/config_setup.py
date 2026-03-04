#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from posementor.local_config import init_local_config, upsert_local_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="初始化 PoseMentor 本地配置")
    parser.add_argument("--config-path", default="configs/local.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--profile", default="quick")
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8787)
    parser.add_argument("--frontend-host", default="127.0.0.1")
    parser.add_argument("--frontend-port", type=int, default=7860)
    parser.add_argument("--dataset-id", default="aistpp")
    parser.add_argument("--standard-id", default="private_action_core")
    parser.add_argument("--aist-video-profile", default="mv3_quick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "profile": args.profile,
        "network.backend_host": args.backend_host,
        "network.backend_port": args.backend_port,
        "network.frontend_host": args.frontend_host,
        "network.frontend_port": args.frontend_port,
        "defaults.dataset_id": args.dataset_id,
        "defaults.standard_id": args.standard_id,
        "defaults.aist_video_profile": args.aist_video_profile,
    }
    if args.force:
        path, created = init_local_config(
            config_path=Path(args.config_path),
            force=True,
            overrides=overrides,
        )
    else:
        path, created = upsert_local_config(
            config_path=Path(args.config_path),
            overrides=overrides,
        )
    print(f"[DONE] 配置文件{'已创建' if created else '已存在'}: {path}")


if __name__ == "__main__":
    main()

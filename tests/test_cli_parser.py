from __future__ import annotations

from posementor.cli import build_parser


def test_cli_supports_config_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["config", "--backend-port", "9001"])
    assert args.command == "config"
    assert args.backend_port == 9001


def test_cli_supports_quickstart_up_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["quickstart", "--epochs", "2", "--up"])
    assert args.command == "quickstart"
    assert args.epochs == 2
    assert args.up is True


def test_cli_supports_config_profile_and_download_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["config", "--plain", "--aist-video-profile", "mv5_standard", "--download-now"])
    assert args.command == "config"
    assert args.aist_video_profile == "mv5_standard"
    assert args.download_now is True


def test_cli_supports_cleanup_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["cleanup"])
    assert args.command == "cleanup"


def test_cli_supports_start_stop_restart_commands() -> None:
    parser = build_parser()
    start_args = parser.parse_args(["start"])
    stop_args = parser.parse_args(["stop"])
    restart_args = parser.parse_args(["restart"])
    assert start_args.command == "start"
    assert stop_args.command == "stop"
    assert restart_args.command == "restart"

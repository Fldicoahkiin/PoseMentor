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

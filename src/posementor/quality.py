from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckItem:
    name: str
    cmd: list[str]
    required: bool


def _run_command(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    output = output.strip()
    if len(output) > 4000:
        output = f"{output[:4000]}\n...(输出已截断)"
    return result.returncode == 0, output


def _tool_exists(command: list[str]) -> bool:
    if not command:
        return False
    return shutil.which(command[0]) is not None


def run_quality_suite(
    project_root: Path,
    *,
    full: bool,
    strict: bool,
    skip_tests: bool,
    skip_mypy: bool,
) -> int:
    checks: list[CheckItem] = [
        CheckItem(
            name="Ruff 静态扫描",
            cmd=[
                "uv",
                "run",
                "ruff",
                "check",
                "--ignore",
                "E501",
                "backend_api.py",
                "download_and_prepare_aist.py",
                "train_3d_lift_demo.py",
                "extract_pose_aist2d.py",
                "extract_pose_yolo11.py",
                "inference_pipeline_demo.py",
                "scripts",
                "src/posementor",
                "tests",
            ],
            required=True,
        ),
        CheckItem(
            name="pip-audit 依赖漏洞扫描",
            cmd=["uv", "run", "--with", "pip-audit>=2.9.0", "pip-audit"],
            required=False,
        ),
        CheckItem(
            name="Deepchecks 数据完整性扫描",
            cmd=[
                "uv",
                "run",
                "--with",
                "deepchecks>=0.19.1",
                "--with",
                "pandas>=2.2.0",
                "--with",
                "setuptools<81",
                "--with",
                "scikit-learn<1.6",
                "--with",
                "numpy<2",
                "python",
                "scripts/run_deepchecks.py",
                "--input",
                "data/processed/aistpp/aist_metadata.csv",
                "--fallback-input",
                "artifacts/visualizations/training_history.csv",
                "--output",
                "artifacts/quality/deepchecks_report.html",
                "--allow-skip",
            ],
            required=True,
        ),
    ]

    print("=== 代码质量检查（Ruff + pip-audit + Deepchecks）===")
    all_ok = True
    for check in checks:
        print(f"\n[RUN] {check.name}")
        if not _tool_exists(check.cmd):
            all_ok = False
            print(f"[FAIL] 缺少命令: {check.cmd[0]}")
            continue
        ok, output = _run_command(check.cmd, cwd=project_root)
        print(output if output else "(无输出)")
        if ok:
            print(f"[OK] {check.name}")
            continue
        if check.required:
            all_ok = False
            print(f"[FAIL] {check.name} 未通过")
        else:
            print(f"[WARN] {check.name} 未通过")

    print("\n=== 结论 ===")
    if all_ok:
        print("[DONE] 代码质量检查通过")
        return 0
    print("[WARN] 代码质量检查未通过，请先修复 FAIL 项。")
    return 1

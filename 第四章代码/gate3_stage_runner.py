"""Gate 3 通用阶段执行器：隔离日志、进程树资源监控与硬超时。

除硬超时外，本执行器只记录资源异常，不自动终止实验。每次调用必须使用
全新 ``output_dir``，防止覆盖既有失败或通过证据。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil


GIB = 1024**3


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:replace"
    return env


def safe_stream_write(stream: Any, text: str) -> None:
    encoding = stream.encoding or "utf-8"
    safe_text = text.encode(encoding, errors="backslashreplace").decode(encoding)
    stream.write(safe_text)
    stream.flush()


def safe_console_write(text: str) -> None:
    safe_stream_write(sys.stdout, text)


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate 3 阶段执行与资源监控")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--gate_label", default="Gate3A")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--scope_dir", type=Path, required=True,
                        help="统计输出体积和磁盘空间的 Gate 3 根目录")
    parser.add_argument("--working_dir", type=Path, required=True)
    parser.add_argument(
        "--timeout_seconds", type=float, default=0.0,
        help="硬超时秒数；0表示禁用自动超时终止",
    )
    parser.add_argument("--check_interval_seconds", type=float, default=10.0)
    parser.add_argument(
        "--console_log_mode", choices=("full", "summary"), default="full",
        help="full转发全部子进程输出；summary仅写阶段日志并显示阶段摘要",
    )
    parser.add_argument("--scope_output_warning_gib", type=float, default=6.5)
    parser.add_argument("--scope_output_red_gib", type=float, default=8.0)
    parser.add_argument("--system_ram_warning_gib", type=float, default=10.0)
    parser.add_argument("--system_ram_red_gib", type=float, default=8.0)
    parser.add_argument("--process_rss_warning_gib", type=float, default=6.0)
    parser.add_argument("--process_rss_red_gib", type=float, default=8.0)
    parser.add_argument("--gpu_warning_gib", type=float, default=12.0)
    parser.add_argument("--gpu_red_gib", type=float, default=14.0)
    parser.add_argument("--disk_warning_gib", type=float, default=120.0)
    parser.add_argument("--disk_red_gib", type=float, default=100.0)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("必须在 -- 后提供待执行命令")
    if args.timeout_seconds < 0:
        parser.error("timeout_seconds 不得为负数")
    if not 0 < args.check_interval_seconds <= 300:
        parser.error("check_interval_seconds 必须在 (0,300] 秒")
    if not 0 < args.scope_output_warning_gib < args.scope_output_red_gib:
        parser.error("输出 warning 阈值必须为正且小于 red 阈值")
    if not 0 < args.system_ram_red_gib < args.system_ram_warning_gib:
        parser.error("可用RAM阈值必须满足 0 < red < warning")
    if not 0 < args.process_rss_warning_gib < args.process_rss_red_gib:
        parser.error("进程RSS阈值必须满足 0 < warning < red")
    if not 0 < args.gpu_warning_gib < args.gpu_red_gib:
        parser.error("GPU阈值必须满足 0 < warning < red")
    if not 0 < args.disk_red_gib < args.disk_warning_gib:
        parser.error("磁盘阈值必须满足 0 < red < warning")
    return args


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def existing_anchor(path: Path) -> Path:
    current = path.resolve()
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def gpu_used_mib() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return max(values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def process_tree_rss(pid: int) -> int:
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return int(total)


def classify(
    sample: dict[str, Any], args: argparse.Namespace,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    red_flags: list[str] = []

    available = sample["system_available_bytes"]
    rss = sample["process_tree_rss_bytes"]
    gpu = sample["gpu_used_mib"]
    disk = sample["disk_free_bytes"]
    output = sample["scope_output_bytes"]

    if available < args.system_ram_red_gib * GIB:
        red_flags.append(f"system_available_ram_below_{args.system_ram_red_gib:g}_gib")
    elif available < args.system_ram_warning_gib * GIB:
        warnings.append(f"system_available_ram_below_{args.system_ram_warning_gib:g}_gib")
    if rss > args.process_rss_red_gib * GIB:
        red_flags.append(f"process_tree_rss_above_{args.process_rss_red_gib:g}_gib")
    elif rss > args.process_rss_warning_gib * GIB:
        warnings.append(f"process_tree_rss_above_{args.process_rss_warning_gib:g}_gib")
    if gpu is not None:
        if gpu > args.gpu_red_gib * 1024:
            red_flags.append(f"gpu_used_above_{args.gpu_red_gib:g}_gib")
        elif gpu > args.gpu_warning_gib * 1024:
            warnings.append(f"gpu_used_above_{args.gpu_warning_gib:g}_gib")
    if disk < args.disk_red_gib * GIB:
        red_flags.append(f"disk_free_below_{args.disk_red_gib:g}_gib")
    elif disk < args.disk_warning_gib * GIB:
        warnings.append(f"disk_free_below_{args.disk_warning_gib:g}_gib")
    if output > args.scope_output_red_gib * GIB:
        red_flags.append(f"gate3_output_above_{args.scope_output_red_gib:g}_gib")
    elif output > args.scope_output_warning_gib * GIB:
        warnings.append(f"gate3_output_above_{args.scope_output_warning_gib:g}_gib")
    return warnings, red_flags


def sample_resources(pid: int | None, scope_dir: Path, started: float) -> dict[str, Any]:
    disk = shutil.disk_usage(existing_anchor(scope_dir))
    return {
        "timestamp_epoch": time.time(),
        "elapsed_seconds": time.perf_counter() - started,
        "system_available_bytes": int(psutil.virtual_memory().available),
        "process_tree_rss_bytes": process_tree_rss(pid) if pid is not None else 0,
        "gpu_used_mib": gpu_used_mib(),
        "disk_free_bytes": int(disk.free),
        "scope_output_bytes": directory_size(scope_dir),
    }


def stop_process_tree(process: subprocess.Popen[bytes]) -> None:
    try:
        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        root.terminate()
    except psutil.NoSuchProcess:
        pass
    _, alive = psutil.wait_procs([*children, root], timeout=10)
    for item in alive:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    requested_check_interval = args.check_interval_seconds
    check_interval = max(10.0, requested_check_interval)
    output_dir = args.output_dir.resolve()
    scope_dir = args.scope_dir.resolve()
    working_dir = args.working_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空阶段目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not working_dir.is_dir():
        raise NotADirectoryError(working_dir)

    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    all_warnings: set[str] = set()
    all_red_flags: set[str] = set()

    preflight = sample_resources(None, scope_dir, started)
    pre_warnings, pre_red = classify(preflight, args)
    samples.append(preflight)
    all_warnings.update(pre_warnings)
    all_red_flags.update(pre_red)
    if pre_red:
        report = {
            "stage": args.stage,
            "status": "PRECHECK_FAIL",
            "command": args.command,
            "working_dir": str(working_dir),
            "warnings": sorted(all_warnings),
            "red_flags": sorted(all_red_flags),
            "samples": samples,
        }
        write_json(output_dir / "stage_monitor_report.json", report)
        safe_stream_write(
            sys.stderr,
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )
        return 2

    log_path = output_dir / "stage.log"
    process = subprocess.Popen(
        args.command,
        cwd=working_dir,
        env=child_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"[{args.gate_label}] stage={args.stage} pid={process.pid}")
    if requested_check_interval < 10:
        print(
            f"[{args.gate_label}] check_interval={requested_check_interval:g}s "
            "自动提升为10s。",
            flush=True,
        )
    if args.console_log_mode == "full":
        safe_console_write(f"[{args.gate_label}] command={args.command}\n")

    def copy_output() -> None:
        assert process.stdout is not None
        with log_path.open("w", encoding="utf-8", newline="") as log_handle:
            for raw_line in iter(process.stdout.readline, b""):
                line = raw_line.decode(args.encoding, errors="replace")
                log_handle.write(line)
                log_handle.flush()
                if args.console_log_mode == "full":
                    safe_console_write(line)

    reader = threading.Thread(target=copy_output, daemon=True)
    reader.start()
    timeout_hit = False
    announced: set[str] = set()

    while process.poll() is None:
        elapsed = time.perf_counter() - started
        if args.timeout_seconds > 0 and elapsed >= args.timeout_seconds:
            timeout_hit = True
            print(f"[{args.gate_label}][HARD_TIMEOUT] {elapsed:.1f}s，终止进程树。", flush=True)
            stop_process_tree(process)
            break
        sample = sample_resources(process.pid, scope_dir, started)
        warnings, red_flags = classify(sample, args)
        samples.append(sample)
        all_warnings.update(warnings)
        all_red_flags.update(red_flags)
        for alert in [*warnings, *red_flags]:
            if alert not in announced:
                level = "RED_FLAG" if alert in red_flags else "WARNING"
                print(f"[{args.gate_label}][{level}] {alert}", flush=True)
                announced.add(alert)
        try:
            wait_seconds = check_interval
            if args.timeout_seconds > 0:
                wait_seconds = min(
                    check_interval,
                    max(0.1, args.timeout_seconds - elapsed),
                )
            process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            pass

    try:
        exit_code = process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        stop_process_tree(process)
        exit_code = process.wait(timeout=15)
    reader.join(timeout=15)
    final_sample = sample_resources(None, scope_dir, started)
    samples.append(final_sample)
    warnings, red_flags = classify(final_sample, args)
    all_warnings.update(warnings)
    all_red_flags.update(red_flags)

    resource_path = output_dir / "resource_samples.csv"
    with resource_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0].keys()))
        writer.writeheader()
        writer.writerows(samples)

    if timeout_hit:
        status = "TIMEOUT"
    elif exit_code != 0:
        status = "CRASHED"
    elif all_red_flags:
        status = "COMPLETED_WITH_RED_FLAGS"
    else:
        status = "PASS"
    report = {
        "stage": args.stage,
        "status": status,
        "command": args.command,
        "working_dir": str(working_dir),
        "duration_seconds": time.perf_counter() - started,
        "exit_code": int(exit_code),
        "timeout_seconds": args.timeout_seconds,
        "timeout_enabled": args.timeout_seconds > 0,
        "requested_check_interval_seconds": requested_check_interval,
        "effective_check_interval_seconds": check_interval,
        "console_log_mode": args.console_log_mode,
        "gate_label": args.gate_label,
        "scope_output_warning_gib": args.scope_output_warning_gib,
        "scope_output_red_gib": args.scope_output_red_gib,
        "resource_thresholds_gib": {
            "system_ram_warning": args.system_ram_warning_gib,
            "system_ram_red": args.system_ram_red_gib,
            "process_rss_warning": args.process_rss_warning_gib,
            "process_rss_red": args.process_rss_red_gib,
            "gpu_warning": args.gpu_warning_gib,
            "gpu_red": args.gpu_red_gib,
            "disk_warning": args.disk_warning_gib,
            "disk_red": args.disk_red_gib,
        },
        "warnings": sorted(all_warnings),
        "red_flags": sorted(all_red_flags),
        "minimum_system_available_bytes": min(s["system_available_bytes"] for s in samples),
        "maximum_process_tree_rss_bytes": max(s["process_tree_rss_bytes"] for s in samples),
        "maximum_gpu_used_mib": max(
            (s["gpu_used_mib"] for s in samples if s["gpu_used_mib"] is not None),
            default=None,
        ),
        "minimum_disk_free_bytes": min(s["disk_free_bytes"] for s in samples),
        "maximum_scope_output_bytes": max(s["scope_output_bytes"] for s in samples),
        "log_path": str(log_path.resolve()),
        "resource_csv": str(resource_path.resolve()),
    }
    write_json(output_dir / "stage_monitor_report.json", report)
    if args.console_log_mode == "full":
        safe_console_write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    else:
        safe_console_write(
            f"[{args.gate_label}] stage={args.stage} status={status} "
            f"elapsed={report['duration_seconds']:.1f}s "
            f"warnings={len(report['warnings'])} red_flags={len(report['red_flags'])}\n"
        )
    if status == "PASS":
        return 0
    if status == "COMPLETED_WITH_RED_FLAGS":
        return 3
    return exit_code if exit_code != 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())

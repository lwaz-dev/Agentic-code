"""run_command: execute shell commands inside the project root."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from agentic.permissions import Risk, classify_command
from agentic.redact import redact
from agentic.tools.registry import Tool, ToolContext, ToolError, truncate_middle


def detect_shell(config: Any = None) -> str:
    """powershell | cmd | bash | zsh | fish | sh"""
    override = getattr(config, "shell", "") if config is not None else ""
    if override:
        return override.lower()
    if os.name == "nt":
        if os.environ.get("PSModulePath") and not os.environ.get("PROMPT"):
            return "powershell"
        return "cmd"
    return os.path.basename(os.environ.get("SHELL", "sh")) or "sh"


def _output_encoding() -> str:
    if os.name == "nt":
        try:
            import ctypes

            return f"cp{ctypes.windll.kernel32.GetOEMCP()}"  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return "utf-8"
    return "utf-8"


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        proc.kill()


def execute_command(command: str, cwd: Path, timeout: int, shell: str) -> dict[str, Any]:
    """Run a command; returns exit code, stdout, stderr, duration. Kills the process tree on timeout/interrupt."""
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    kwargs: dict[str, Any] = dict(
        cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, env=env
    )
    if os.name == "nt":
        if shell == "powershell":
            exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
            args: Any = [exe, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command]
        else:
            args, kwargs["shell"] = command, True
    else:
        args, kwargs["shell"], kwargs["start_new_session"] = command, True, True
        if shell in ("bash", "zsh") and shutil.which(shell):
            kwargs["executable"] = shutil.which(shell)

    started = time.monotonic()
    proc = subprocess.Popen(args, **kwargs)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        out, err = proc.communicate()
        timed_out = True
    except KeyboardInterrupt:
        _kill_tree(proc)
        proc.wait()
        raise
    enc = _output_encoding()
    return {
        "exit_code": -1 if timed_out else proc.returncode,
        "stdout": (out or b"").decode(enc, errors="replace"),
        "stderr": (err or b"").decode(enc, errors="replace"),
        "duration_s": round(time.monotonic() - started, 2),
        "timed_out": timed_out,
    }


def run_command(ctx: ToolContext, command: str, timeout: int | None = None):
    if not command.strip():
        raise ToolError("command must not be empty.")
    if ctx.mode in ("review", "plan") and classify_command(command) != Risk.SAFE:
        raise ToolError(f"Only read-only/safe commands may run in {ctx.mode} mode.")
    decision = ctx.permissions.check_command(command)
    if not decision.allowed:
        raise ToolError(decision.reason)
    timeout = max(1, min(timeout or ctx.config.command_timeout, 900))
    shell = detect_shell(ctx.config)
    result = execute_command(command, ctx.root, timeout, shell)
    limit = ctx.config.max_output_chars
    data: dict[str, Any] = {
        "command": command,
        "exit_code": result["exit_code"],
        "ok": result["exit_code"] == 0,
        "duration_s": result["duration_s"],
        "timed_out": result["timed_out"],
        "shell": shell,
    }
    notes = []
    for stream in ("stdout", "stderr"):
        text, truncated, shown, total = truncate_middle(redact(result[stream]), limit)
        data[stream] = text
        if truncated:
            notes.append(f"{stream} truncated. Showing {shown} of {total:,} lines.")
    if result["timed_out"]:
        notes.append(f"Command timed out after {timeout}s and was killed.")
    if notes:
        data["note"] = " ".join(notes)
    return data


TOOLS = [
    Tool(
        "run_command",
        "Run a shell command in the project directory (use for tests, linters, builds, package managers). "
        "Returns exit code, stdout, stderr and duration. Risky commands require user approval. Commands must not be interactive.",
        {"type": "object", "properties": {
            "command": {"type": "string", "description": "The command line to run."},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default from config, max 900)."}},
         "required": ["command"]},
        run_command, "exec",
    )
]

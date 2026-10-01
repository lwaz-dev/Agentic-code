"""Git tools (argument-list subprocess calls, never a shell)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from agentic.redact import redact
from agentic.tools.filesystem import rel, resolve_path
from agentic.tools.registry import Tool, ToolContext, ToolError, truncate_middle

_REF = re.compile(r"^[\w./@{}\-]+$")


def run_git(root: Path, *args: str, timeout: int = 30) -> tuple[int, str, str]:
    git = shutil.which("git")
    if not git:
        raise ToolError("git is not installed or not on PATH.")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        proc = subprocess.run(
            [git, *args], cwd=str(root), capture_output=True, stdin=subprocess.DEVNULL,
            timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        raise ToolError("git command timed out.") from None
    dec = lambda b: b.decode("utf-8", errors="replace")  # noqa: E731
    return proc.returncode, dec(proc.stdout), dec(proc.stderr)


def git_summary(root: Path) -> str:
    """Short human status for /status."""
    if not shutil.which("git"):
        return "git not installed"
    try:
        code, out, _ = run_git(root, "status", "--porcelain")
    except ToolError:
        return "unavailable"
    if code != 0:
        return "Not a repository"
    n = len([ln for ln in out.splitlines() if ln.strip()])
    return "Clean" if n == 0 else f"{n} changed file(s)"


def _read(ctx: ToolContext, *args: str):
    code, out, err = run_git(ctx.root, *args)
    if code != 0:
        raise ToolError(redact(err.strip() or "git failed") )
    text, truncated, shown, total = truncate_middle(redact(out), ctx.config.max_output_chars)
    data = {"output": text}
    if truncated:
        data["note"] = f"Output truncated. Showing {shown} of {total:,} lines."
    return data


def git_status(ctx: ToolContext):
    return _read(ctx, "status", "--short", "--branch")


def git_diff(ctx: ToolContext, path: str | None = None, staged: bool = False):
    args = ["diff", "--no-color"] + (["--cached"] if staged else [])
    if path:
        args += ["--", rel(ctx.root, resolve_path(ctx.root, path))]
    return _read(ctx, *args)


def git_log(ctx: ToolContext, max_count: int = 10):
    return _read(ctx, "log", "--oneline", "--decorate", "-n", str(max(1, min(max_count, 100))))


def git_branch(ctx: ToolContext):
    return _read(ctx, "branch", "--all", "--no-color")


def _write(ctx: ToolContext, args: list[str], always_ask: bool = False):
    decision = ctx.permissions.check_git_write("git " + " ".join(args), always_ask)
    if not decision.allowed:
        raise ToolError(decision.reason)
    code, out, err = run_git(ctx.root, *args)
    if code != 0:
        raise ToolError(redact((err or out).strip() or "git failed"))
    return {"output": redact((out or err).strip())}


def git_add(ctx: ToolContext, paths: list):
    if not paths or not all(isinstance(p, str) for p in paths):
        raise ToolError("paths must be a non-empty list of strings.")
    rels = [rel(ctx.root, resolve_path(ctx.root, p)) for p in paths]
    return _write(ctx, ["add", "--", *rels])


def git_commit(ctx: ToolContext, message: str):
    if not message.strip():
        raise ToolError("Commit message must not be empty.")
    return _write(ctx, ["commit", "-m", message], always_ask=True)


def git_checkout(ctx: ToolContext, ref: str, create: bool = False):
    if not _REF.match(ref) or ref.startswith("-"):
        raise ToolError("Invalid branch/ref name.")
    return _write(ctx, ["checkout", *(["-b"] if create else []), ref])


_NOARGS = {"type": "object", "properties": {}}

TOOLS = [
    Tool("git_status", "Show git status (short format with branch).", _NOARGS, git_status, "read"),
    Tool("git_diff", "Show uncommitted changes, optionally for one path or staged only.",
         {"type": "object", "properties": {"path": {"type": "string"}, "staged": {"type": "boolean"}}},
         git_diff, "read"),
    Tool("git_log", "Show recent commits (one line each).",
         {"type": "object", "properties": {"max_count": {"type": "integer"}}}, git_log, "read"),
    Tool("git_branch", "List local and remote branches.", _NOARGS, git_branch, "read"),
    Tool("git_add", "Stage files. Requires user approval.",
         {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
          "required": ["paths"]}, git_add, "git_write"),
    Tool("git_commit", "Create a commit with the staged changes. Only when the user asked for a commit. Always requires approval.",
         {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
         git_commit, "git_write"),
    Tool("git_checkout", "Switch (or create with create=true) a branch. Requires user approval.",
         {"type": "object", "properties": {"ref": {"type": "string"}, "create": {"type": "boolean"}},
          "required": ["ref"]}, git_checkout, "git_write"),
]

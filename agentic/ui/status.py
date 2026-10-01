"""Data for /status and /doctor."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic.tools.filesystem import walk
from agentic.tools.git import git_summary
from agentic.tools.terminal import detect_shell


@dataclass
class Check:
    ok: bool | None  # None = warning
    label: str
    detail: str = ""


def collect_status(agent) -> list[tuple[str, str]]:
    from itertools import islice

    files = sum(1 for _, is_dir in islice(walk(agent.root, agent.root), 50_000) if not is_dir)
    return [
        ("Project", str(agent.root)),
        ("Model", agent.config.model),
        ("Mode", agent.config.mode.capitalize()),
        ("Approval", agent.config.approval_mode),
        ("Files", str(files)),
        ("Git", git_summary(agent.root)),
        ("Session", f"{len(agent.conversation.messages)} messages"),
        ("Iterations", f"{agent.last_iterations} / {agent.config.max_iterations}"),
    ]


def run_doctor(agent) -> list[Check]:
    cfg = agent.config
    checks: list[Check] = []
    v = sys.version_info
    checks.append(Check(v >= (3, 11), f"Python {v.major}.{v.minor}.{v.micro}", "" if v >= (3, 11) else "3.11+ required"))
    checks.append(Check(bool(cfg.api_key), "OpenRouter API key" if cfg.provider == "openrouter" else "API key",
                        "" if cfg.api_key else "set OPENROUTER_API_KEY in .env or the environment"))
    checks.append(Check(bool(cfg.model.strip()), f"Model configuration ({cfg.model})"))
    if cfg.api_key:
        try:
            ok, detail = agent.get_provider().check_connection()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, str(exc)
        checks.append(Check(ok, "Provider connection", detail))
    else:
        checks.append(Check(None, "Provider connection", "skipped (no API key)"))
    root: Path = agent.root
    access = os.access(root, os.R_OK | os.W_OK | os.X_OK)
    checks.append(Check(access, "Project access", str(root) if access else "no read/write permission"))
    git = shutil.which("git")
    checks.append(Check(bool(git), "Git", "" if git else "not installed (git tools disabled)"))
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    utf = "utf" in enc
    checks.append(Check(True if utf else None, f"Terminal ({detect_shell(cfg)}, {enc or 'unknown'})",
                        "" if utf else "non-UTF-8 output; some symbols may look odd"))
    return checks

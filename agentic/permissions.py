"""Command risk classification and approval policy.

This is a heuristic guard, not a sandbox: it reduces accidents, it does not stop a determined attacker.
"""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class Risk(str, Enum):
    SAFE = "safe"
    MODERATE = "moderate"
    DANGEROUS = "dangerous"


_ORDER = {Risk.SAFE: 0, Risk.MODERATE: 1, Risk.DANGEROUS: 2}

_SPLIT = re.compile(r"\|\||&&|;|\||\r?\n")

DANGEROUS_BINS = {
    "rm", "rmdir", "rd", "del", "erase", "format", "shutdown", "reboot", "halt", "poweroff",
    "diskpart", "mkfs", "fdisk", "dd", "sudo", "su", "doas", "taskkill", "kill", "killall",
    "pkill", "remove-item", "ri", "truncate", "shred", "wipefs", "stop-computer",
    "restart-computer", "clear-disk", "format-volume", "chown",
}
WRAPPERS = {"bash", "sh", "zsh", "fish", "cmd", "powershell", "pwsh", "env", "xargs", "nohup", "start"}
_DANGEROUS_WORDS = re.compile(r"\b(" + "|".join(sorted(DANGEROUS_BINS - {"su", "ri", "format"})) + r")\b", re.I)

DANGEROUS_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\bgit\s+(?:\S+\s+)*(reset\s+--hard|clean\b|push\b)",
        r"\bgit\b.*--force",
        r"\brmdir\s+/s", r"\brd\s+/s", r"\bdel\s+/", r"-recurse\b",
        r"\|\s*(sh|bash|zsh|iex|invoke-expression)\b",
        r":\(\)\s*\{",
        r"\bchmod\s+-r",
        r">\s*/dev/(sd|nvme|disk)",
        r"\breg\s+delete\b",
        r"\bfind\b.*(-delete|-exec)",
        r"\bxargs\b.*\b(rm|del)\b",
    )
]
_SENSITIVE_FILE = re.compile(r"(^|[\s/\\'\"])\.env(\.(?!example|sample|template|dist)\w+)?(?=$|[\s'\"])", re.I)

SAFE_BINS = {
    "ls", "dir", "pwd", "echo", "cat", "type", "head", "tail", "wc", "grep", "rg", "which", "where",
    "whoami", "date", "tree", "stat", "file", "get-childitem", "gci", "get-content", "gc",
    "select-string", "sls", "get-location", "hostname", "uname", "sort", "uniq", "diff",
    "pytest", "phpunit", "flake8", "mypy",
}
_SAFE_GIT = {"status", "diff", "log", "show", "blame", "ls-files", "rev-parse"}
_SAFE_BRANCH_FLAGS = {"-a", "-r", "-v", "-vv", "--list", "--all", "--show-current"}


def _bin(token: str) -> str:
    name = token.replace("\\", "/").rsplit("/", 1)[-1].lower().strip("\"'")
    for ext in (".exe", ".cmd", ".bat", ".ps1"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name


def _tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment, posix=(os.name != "nt"))
    except ValueError:
        return segment.split()


def _is_safe_segment(tokens: list[str]) -> bool:
    t0, rest = _bin(tokens[0]), [t.strip("\"'") for t in tokens[1:]]
    if t0 in SAFE_BINS:
        return True
    if t0 == "git":
        args = [a for a in rest if not a.startswith("-")]
        if not args:
            return False
        sub = args[0]
        if sub in _SAFE_GIT:
            return True
        if sub == "branch":
            tail = rest[rest.index("branch") + 1 :]
            return all(a in _SAFE_BRANCH_FLAGS for a in tail)
        return sub == "remote" and rest[-1:] == ["-v"]
    first = rest[0] if rest else ""
    if t0 == "php":
        return first in ("-l", "-v", "--version")
    if t0 in ("python", "python3", "py"):
        return first in ("--version", "-V") or rest[:2] in (
            ["-m", "pytest"], ["-m", "unittest"], ["-m", "py_compile"], ["-m", "compileall"],
        )
    if t0 == "node":
        return first in ("--version", "-v", "--check", "-c")
    if t0 == "npm":
        return first in ("--version", "-v", "test", "t") or (
            first == "run" and rest[1:2] in (["test"], ["lint"], ["typecheck"])
        )
    if t0 == "composer":
        return first in ("test", "validate", "--version")
    if t0 == "cargo":
        return first in ("check", "test", "--version")
    if t0 == "go":
        return first in ("test", "vet", "version")
    if t0 == "ruff":
        return first == "check" and "--fix" not in rest
    if t0 == "tsc":
        return "--noEmit" in rest
    return False


def classify_command(command: str) -> Risk:
    text = command.strip()
    if not text:
        return Risk.MODERATE
    if any(p.search(text) for p in DANGEROUS_PATTERNS):
        return Risk.DANGEROUS
    risk = Risk.SAFE
    for segment in _SPLIT.split(text):
        tokens = _tokens(segment.strip())
        if not tokens:
            continue
        t0 = _bin(tokens[0])
        if t0 in DANGEROUS_BINS or t0.startswith("mkfs"):
            return Risk.DANGEROUS
        if t0 in WRAPPERS:
            if _DANGEROUS_WORDS.search(text):
                return Risk.DANGEROUS
            risk = max(risk, Risk.MODERATE, key=_ORDER.get)
            continue
        if not _is_safe_segment(tokens):
            risk = max(risk, Risk.MODERATE, key=_ORDER.get)
    if "$(" in text or "`" in text or _SENSITIVE_FILE.search(text):
        risk = max(risk, Risk.MODERATE, key=_ORDER.get)
    if ">" in re.sub(r"\d?>\s*&\s*\d", "", text):
        risk = max(risk, Risk.MODERATE, key=_ORDER.get)
    return risk


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    risk: Risk | None = None


ConfirmFn = Callable[[str, str, str], bool]  # (title, detail, warning) -> allowed?


class PermissionManager:
    """Decides when the user must approve an action. Deleting and dangerous commands always ask."""

    def __init__(self, approval_mode: Callable[[], str], confirm: ConfirmFn | None = None) -> None:
        self.approval_mode = approval_mode
        self.confirm = confirm

    def _ask(self, title: str, detail: str, warning: str, denied: str, risk: Risk | None = None) -> Decision:
        if self.confirm is None:
            return Decision(False, "Approval required but no interactive user is available.", risk)
        if self.confirm(title, detail, warning):
            return Decision(True, "", risk)
        return Decision(False, denied, risk)

    def check_command(self, command: str) -> Decision:
        risk = classify_command(command)
        mode = self.approval_mode()
        needs = risk == Risk.DANGEROUS or mode == "strict" or (risk == Risk.MODERATE and mode == "normal")
        if not needs:
            return Decision(True, "", risk)
        warning = {
            Risk.DANGEROUS: "This may permanently delete files or affect your system.",
            Risk.MODERATE: "This command may modify your project or install software.",
            Risk.SAFE: "",
        }[risk]
        return self._ask("Agent wants to run:", command, warning, "User denied permission to run this command.", risk)

    def check_write(self, rel: str) -> Decision:
        if self.approval_mode() != "strict":
            return Decision(True)
        return self._ask("Agent wants to write:", rel, "", "User denied permission to write this file.")

    def check_edit(self, rel: str) -> Decision:
        if self.approval_mode() != "strict":
            return Decision(True)
        return self._ask("Agent wants to edit:", rel, "", "User denied permission to edit this file.")

    def check_overwrite(self, rel: str) -> Decision:
        if self.approval_mode() == "auto":
            return Decision(True)
        return self._ask(
            "Agent wants to overwrite:", rel, "This replaces the existing file contents.",
            "User denied permission to overwrite this file.",
        )

    def check_delete(self, rel: str) -> Decision:
        return self._ask(
            "Agent wants to delete:", rel, "This may permanently delete files.",
            "User denied permission to delete this file.",
        )

    def check_sensitive_write(self, rel: str) -> Decision:
        return self._ask(
            "Agent wants to modify a sensitive file:", rel, "This file may contain secrets.",
            "User denied permission to modify this sensitive file.",
        )

    def check_git_write(self, description: str, always_ask: bool = False) -> Decision:
        if self.approval_mode() == "auto" and not always_ask:
            return Decision(True)
        return self._ask(
            "Agent wants to run:", description, "This changes your repository.",
            "User denied permission for this git operation.",
        )

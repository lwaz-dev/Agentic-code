"""Project filesystem tools. All paths are confined to the project root."""
from __future__ import annotations

import difflib
import fnmatch
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

from agentic.redact import REDACTED, redact
from agentic.tools.registry import Change, Tool, ToolContext, ToolError

IGNORED_DIRS = {
    ".git", "node_modules", "vendor", "__pycache__", ".venv", "venv", ".idea", ".vscode",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".next",
}
MAX_WRITE_BYTES = 1_000_000
MAX_READ_BYTES = 5_000_000
_SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".jks"}
_SENSITIVE_NAMES = {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".netrc", ".pgpass"}
_EXAMPLE_SUFFIXES = (".example", ".sample", ".template", ".dist")


# ---------------------------------------------------------------- path safety
def is_sensitive_file(path: Path) -> bool:
    name = path.name.lower()
    if name == ".env" or (name.startswith(".env.") and not name.endswith(_EXAMPLE_SUFFIXES)):
        return True
    return name in _SENSITIVE_NAMES or path.suffix.lower() in _SENSITIVE_SUFFIXES


def resolve_path(root: Path, path: str) -> Path:
    """Resolve a model-supplied path; reject anything outside the project root."""
    if not isinstance(path, str) or not path.strip() or "\x00" in path:
        raise ToolError("Invalid path.")
    candidate = Path(path.strip())
    full = candidate if candidate.is_absolute() else Path(root) / candidate
    full = Path(os.path.realpath(full))
    root_real = Path(os.path.realpath(root))
    try:
        full.relative_to(root_real)
    except ValueError:
        raise ToolError("Path is outside project root.") from None
    return full


def rel(root: Path, full: Path) -> str:
    rel_path = full.relative_to(Path(os.path.realpath(root))).as_posix()
    return rel_path or "."


def _guard_write(root: Path, full: Path) -> str:
    rel_path = rel(root, full)
    if ".git" in rel_path.split("/"):
        raise ToolError("Modifying the .git directory is not allowed.")
    return rel_path


# ------------------------------------------------------------------- walking
def _load_gitignore(root: Path) -> list[str]:
    try:
        lines = (root / ".gitignore").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith(("#", "!"))]


def _ignored(rel_path: str, is_dir: bool, patterns: list[str]) -> bool:
    parts = rel_path.split("/")
    for pat in patterns:
        dir_only = pat.endswith("/")
        pat = pat.strip("/")
        if not pat:
            continue
        if "/" in pat:
            if fnmatch.fnmatch(rel_path, pat) or rel_path.startswith(pat + "/"):
                return True
            continue
        for i, part in enumerate(parts):
            part_is_dir = i < len(parts) - 1 or is_dir
            if dir_only and not part_is_dir:
                continue
            if fnmatch.fnmatch(part, pat):
                return True
    return False


def walk(
    root: Path, base: Path, include_ignored: bool = False, max_depth: int | None = None
) -> Iterator[tuple[str, bool]]:
    """Yield (relative_posix_path, is_dir) under base, honouring ignore rules."""
    patterns = [] if include_ignored else _load_gitignore(root)
    root_real = Path(os.path.realpath(root))
    for dirpath, dirnames, filenames in os.walk(base):
        dir_rel = Path(dirpath).relative_to(root_real).as_posix() if Path(dirpath) != root_real else ""
        depth = dir_rel.count("/") + 1 if dir_rel else 0
        kept: list[str] = []
        for d in sorted(dirnames):
            r = f"{dir_rel}/{d}" if dir_rel else d
            if not include_ignored and (d in IGNORED_DIRS or _ignored(r, True, patterns)):
                continue
            kept.append(d)
            yield r, True
        dirnames[:] = kept if max_depth is None or depth + 1 < max_depth else []
        for f in sorted(filenames):
            r = f"{dir_rel}/{f}" if dir_rel else f
            if not include_ignored and (f.lower().startswith(".env") and is_sensitive_file(Path(f))):
                continue
            if not include_ignored and _ignored(r, False, patterns):
                continue
            yield r, False


# ------------------------------------------------------------------- changes
def make_diff(path: str, old: str | None, new: str | None) -> tuple[str, int, int]:
    lines = list(
        difflib.unified_diff(
            (old or "").splitlines(), (new or "").splitlines(),
            fromfile="/dev/null" if old is None else f"a/{path}",
            tofile="/dev/null" if new is None else f"b/{path}",
            lineterm="", n=3,
        )
    )
    added = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
    return "\n".join(lines), added, removed


def _record(ctx: ToolContext, path: str, old: str | None, new: str | None) -> Change:
    ctx.backups.setdefault(path, old)  # in-memory backup of the original
    action = "created" if old is None else "deleted" if new is None else "modified"
    diff, added, removed = make_diff(path, old, new)
    change = Change(path, action, diff, added, removed)
    ctx.changes.append(change)
    return change


def session_changes(ctx: ToolContext) -> list[Change]:
    """Net changes versus the original content of every file touched this session."""
    result: list[Change] = []
    for path, original in ctx.backups.items():
        full = ctx.root / path
        current = full.read_text(encoding="utf-8", errors="replace") if full.is_file() else None
        if current == original:
            continue
        diff, added, removed = make_diff(path, original, current)
        action = "created" if original is None else "deleted" if current is None else "modified"
        result.append(Change(path, action, diff, added, removed))
    return result


def _atomic_write(full: Path, text: str, crlf: bool = False) -> None:
    normalized = text.replace("\r\n", "\n")
    data = (normalized.replace("\n", "\r\n") if crlf else normalized).encode("utf-8")
    full.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(full.parent), prefix=".agentic-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if full.exists():
            shutil.copymode(full, tmp)
        elif os.name != "nt":
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, full)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_text(full: Path) -> str:
    return full.read_bytes().decode("utf-8", errors="replace")


# -------------------------------------------------------------------- tools
def list_files(ctx: ToolContext, path: str = ".", max_depth: int | None = None, include_ignored: bool = False):
    base = resolve_path(ctx.root, path)
    if not base.is_dir():
        raise ToolError(f"Not a directory: {path}")
    limit = ctx.config.max_list_files
    entries: list[str] = []
    total = 0
    for rel_path, is_dir in walk(ctx.root, base, include_ignored, max_depth):
        total += 1
        if len(entries) < limit:
            entries.append(rel_path + ("/" if is_dir else ""))
        if total >= 50_000:
            break
    data = {"path": rel(ctx.root, base), "count": total, "entries": entries}
    if total > len(entries):
        data["note"] = f"Output truncated. Showing {len(entries)} of {total} entries. Narrow with 'path'."
    return data


def read_file(ctx: ToolContext, path: str, start_line: int | None = None, end_line: int | None = None):
    full = resolve_path(ctx.root, path)
    if not full.is_file():
        raise ToolError(f"File not found: {path}")
    if is_sensitive_file(full):
        raise ToolError("Reading secret/credential files is blocked to protect your secrets.")
    if full.stat().st_size > MAX_READ_BYTES:
        raise ToolError("File is too large to read (over 5 MB).")
    raw = full.read_bytes()
    if b"\x00" in raw[:4096]:
        raise ToolError("File appears to be binary.")
    lines = raw.decode("utf-8", errors="replace").splitlines()
    total = len(lines)
    start = max(1, start_line or 1)
    end = min(total, end_line or total)
    if start > max(total, 1):
        raise ToolError(f"start_line {start} is past the end of the file ({total} lines).")
    selected = lines[start - 1 : end]
    limit = ctx.config.max_read_chars
    used, count = 0, 0
    for line in selected:
        if used + len(line) + 1 > limit and count:
            break
        used += len(line) + 1
        count += 1
    truncated = count < len(selected)
    selected = selected[:count]
    end = start + count - 1 if count else start - 1
    content = "\n".join(selected)
    safe = redact(content)
    rel_path = rel(ctx.root, full)
    if safe != content:
        ctx.redacted_files.add(rel_path)
    ctx.read_files[rel_path] = full.stat().st_mtime_ns
    data = {"path": rel_path, "content": safe, "start_line": start, "end_line": end, "total_lines": total}
    if truncated or end < total:
        data["note"] = f"Showing lines {start}-{end} of {total}. Use start_line/end_line to read more."
    if safe != content:
        data["warning"] = f"Secrets were replaced with {REDACTED}; do not rewrite those lines."
    return data


def write_file(ctx: ToolContext, path: str, content: str):
    full = resolve_path(ctx.root, path)
    rel_path = _guard_write(ctx.root, full)
    if full.is_dir():
        raise ToolError(f"{path} is a directory.")
    if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
        raise ToolError("Content is too large (over 1 MB).")
    old: str | None = None
    crlf = False
    if full.exists():
        if rel_path in ctx.redacted_files:
            raise ToolError("This file contained redacted secrets; use edit_file instead of overwriting it.")
        old = _read_text(full)
        crlf = "\r\n" in old
        decision = ctx.permissions.check_overwrite(rel_path)
    else:
        decision = ctx.permissions.check_write(rel_path)
    if not decision.allowed:
        raise ToolError(decision.reason)
    if is_sensitive_file(full):
        sensitive = ctx.permissions.check_sensitive_write(rel_path)
        if not sensitive.allowed:
            raise ToolError(sensitive.reason)
    _atomic_write(full, content, crlf)
    if not full.is_file():
        raise ToolError("Write verification failed: file does not exist after writing.")
    _record(ctx, rel_path, old, _read_text(full))
    ctx.read_files[rel_path] = full.stat().st_mtime_ns
    return {"path": rel_path, "created": old is None, "bytes": full.stat().st_size}


def edit_file(ctx: ToolContext, path: str, old_text: str, new_text: str, replace_all: bool = False):
    full = resolve_path(ctx.root, path)
    rel_path = _guard_write(ctx.root, full)
    if not full.is_file():
        raise ToolError(f"File not found: {path}")
    if is_sensitive_file(full):
        raise ToolError("Editing secret/credential files is blocked.")
    if rel_path not in ctx.read_files:
        raise ToolError("Read the file with read_file before editing it.")
    if ctx.read_files[rel_path] != full.stat().st_mtime_ns:
        raise ToolError("The file changed on disk since you last read it. Read it again before editing.")
    if REDACTED in old_text:
        raise ToolError(f"old_text contains {REDACTED}; choose lines that do not contain secrets.")
    if not old_text:
        raise ToolError("old_text must not be empty. Use write_file to create files.")
    try:
        raw = full.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError("File is not valid UTF-8; refusing to edit.") from None
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    old = old_text.replace("\r\n", "\n")
    new = new_text.replace("\r\n", "\n")
    if old == new:
        raise ToolError("old_text and new_text are identical; nothing to change.")
    count = text.count(old)
    if count == 0:
        raise ToolError("old_text was not found. It must match exactly (including whitespace). Re-read the file.")
    if count > 1 and not replace_all:
        raise ToolError(f"old_text matches {count} places. Add surrounding context or set replace_all=true.")
    decision = ctx.permissions.check_edit(rel_path)
    if not decision.allowed:
        raise ToolError(decision.reason)
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    _atomic_write(full, updated, crlf)
    if not full.is_file():
        raise ToolError("Write verification failed: file does not exist after editing.")
    _record(ctx, rel_path, text, updated)
    ctx.read_files[rel_path] = full.stat().st_mtime_ns
    return {"path": rel_path, "replacements": count if replace_all else 1}


def delete_file(ctx: ToolContext, path: str):
    full = resolve_path(ctx.root, path)
    rel_path = _guard_write(ctx.root, full)
    if not full.is_file():
        raise ToolError(f"File not found (only files can be deleted): {path}")
    decision = ctx.permissions.check_delete(rel_path)
    if not decision.allowed:
        raise ToolError(decision.reason)
    old = _read_text(full)
    full.unlink()
    _record(ctx, rel_path, old, None)
    ctx.read_files.pop(rel_path, None)
    return {"path": rel_path, "deleted": True}


_PATH = {"type": "string", "description": "Path relative to the project root."}

TOOLS = [
    Tool(
        "list_files",
        "List files and directories in the project (respects .gitignore; hides .git, .env, node_modules, vendor, caches).",
        {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory to list (default '.')."},
            "max_depth": {"type": "integer", "description": "Limit directory depth."},
            "include_ignored": {"type": "boolean", "description": "Also show ignored/hidden entries."}}},
        list_files, "read",
    ),
    Tool(
        "read_file",
        "Read a text file from the project. Large files are truncated; use start_line/end_line for ranges.",
        {"type": "object", "properties": {
            "path": _PATH,
            "start_line": {"type": "integer", "description": "1-based first line."},
            "end_line": {"type": "integer", "description": "1-based last line (inclusive)."}},
         "required": ["path"]},
        read_file, "read",
    ),
    Tool(
        "write_file",
        "Create a new file (or fully replace an existing one, which needs user approval). Prefer edit_file for changes.",
        {"type": "object", "properties": {"path": _PATH, "content": {"type": "string"}},
         "required": ["path", "content"]},
        write_file, "write",
    ),
    Tool(
        "edit_file",
        "Edit an existing file by exact text replacement. The file must have been read first. old_text must match exactly once (or set replace_all).",
        {"type": "object", "properties": {
            "path": _PATH,
            "old_text": {"type": "string", "description": "Exact text to replace."},
            "new_text": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence."}},
         "required": ["path", "old_text", "new_text"]},
        edit_file, "write",
    ),
    Tool(
        "delete_file",
        "Delete a single file. Always requires user confirmation.",
        {"type": "object", "properties": {"path": _PATH}, "required": ["path"]},
        delete_file, "write",
    ),
]

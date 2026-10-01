"""Project-wide text search."""
from __future__ import annotations

import fnmatch
import re

from agentic.redact import redact
from agentic.tools.filesystem import is_sensitive_file, rel, resolve_path, walk
from agentic.tools.registry import Tool, ToolContext, ToolError

MAX_FILE_BYTES = 1_000_000


def search_files(
    ctx: ToolContext,
    pattern: str,
    path: str = ".",
    regex: bool = False,
    case_sensitive: bool = False,
    glob: str | None = None,
    include_ignored: bool = False,
):
    if not pattern:
        raise ToolError("pattern must not be empty.")
    base = resolve_path(ctx.root, path)
    if not base.exists():
        raise ToolError(f"Path not found: {path}")
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        rx = re.compile(pattern if regex else re.escape(pattern), flags)
    except re.error as exc:
        raise ToolError(f"Invalid regular expression: {exc}") from exc

    limit = ctx.config.max_search_results
    results: list[str] = []
    truncated = False
    if base.is_file():
        files = [rel(ctx.root, base)]
    else:
        files = (p for p, is_dir in walk(ctx.root, base, include_ignored) if not is_dir)
    for rel_path in files:
        if glob and not (fnmatch.fnmatch(rel_path, glob) or fnmatch.fnmatch(rel_path.rsplit("/", 1)[-1], glob)):
            continue
        full = ctx.root / rel_path
        try:
            if is_sensitive_file(full) or full.stat().st_size > MAX_FILE_BYTES:
                continue
            raw = full.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:2048]:
            continue
        for number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
            if rx.search(line):
                if len(results) >= limit:
                    truncated = True
                    break
                results.append(f"{rel_path}:{number}: {redact(line.strip())[:200]}")
        if truncated:
            break
    data = {"pattern": pattern, "count": len(results), "matches": results}
    if truncated:
        data["note"] = f"Output truncated. Showing the first {limit} matches; refine the pattern or path."
    return data


TOOLS = [
    Tool(
        "search_files",
        "Search project files for text (case-insensitive by default). Returns 'path:line: text'. Skips .git, node_modules, vendor, __pycache__ and binary files.",
        {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Text or regex to look for."},
            "path": {"type": "string", "description": "File or directory to search (default '.')."},
            "regex": {"type": "boolean", "description": "Treat pattern as a regular expression."},
            "case_sensitive": {"type": "boolean"},
            "glob": {"type": "string", "description": "Only files matching this glob, e.g. '*.php'."},
            "include_ignored": {"type": "boolean"}},
         "required": ["pattern"]},
        search_files, "read",
    )
]

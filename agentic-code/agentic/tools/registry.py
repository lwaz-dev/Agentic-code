"""Tool definitions, argument validation and safe execution."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agentic.log import get_logger

# Which tool permission levels each agent mode may use.
MODE_PERMISSIONS: dict[str, set[str]] = {
    "chat": set(),
    "plan": {"read"},
    "review": {"read", "exec"},
    "code": {"read", "write", "exec", "git_write"},
    "auto": {"read", "write", "exec", "git_write"},
}


class ToolError(Exception):
    """A recoverable tool failure; the message is returned to the model."""


@dataclass
class Change:
    path: str
    action: str  # created | modified | deleted
    diff: str = ""
    added: int = 0
    removed: int = 0


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    permission: str = "read"  # read | write | exec | git_write

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class ToolContext:
    root: Path
    config: Any
    permissions: Any
    ui: Any = None
    mode: str = "code"
    planner: Any = None
    read_files: dict[str, int] = field(default_factory=dict)  # rel path -> mtime_ns when last read/written
    redacted_files: set[str] = field(default_factory=set)
    backups: dict[str, str | None] = field(default_factory=dict)  # original content (None = did not exist)
    changes: list[Change] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.ui is None:
            from agentic.ui import NullUI

            self.ui = NullUI()


def ok(tool: str, data: Any) -> dict[str, Any]:
    return {"success": True, "tool": tool, "data": data}


def fail(tool: str, error: str) -> dict[str, Any]:
    return {"success": False, "tool": tool, "error": error}


_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
    "array": (list,), "object": (dict,),
}


def validate_args(schema: dict[str, Any], args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ToolError("Tool arguments must be a JSON object.")
    props = schema.get("properties", {})
    clean: dict[str, Any] = {}
    for key, value in args.items():
        if key not in props:
            continue  # ignore unknown keys
        expected = props[key].get("type")
        if expected == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value)
        if expected == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
            value = value.lower() == "true"
        if value is None:
            continue
        if expected in _TYPES:
            bad_bool = expected in ("integer", "number") and isinstance(value, bool)
            if bad_bool or not isinstance(value, _TYPES[expected]):
                raise ToolError(f"Parameter '{key}' must be of type {expected}.")
        clean[key] = value
    missing = [k for k in schema.get("required", []) if k not in clean]
    if missing:
        raise ToolError(f"Missing required parameter(s): {', '.join(missing)}")
    return clean


def truncate_middle(text: str, max_chars: int) -> tuple[str, bool, int, int]:
    """Trim long output keeping head and tail. Returns (text, truncated, shown_lines, total_lines)."""
    lines = text.splitlines()
    if len(text) <= max_chars:
        return text, False, len(lines), len(lines)
    head_budget, tail_budget = max_chars // 3, max_chars - max_chars // 3
    head: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > head_budget:
            break
        head.append(line)
        used += len(line) + 1
    tail: list[str] = []
    used = 0
    for line in reversed(lines[len(head):]):
        if used + len(line) + 1 > tail_budget:
            break
        tail.append(line)
        used += len(line) + 1
    tail.reverse()
    omitted = len(lines) - len(head) - len(tail)
    marker = f"... [{omitted} lines omitted] ..."
    return "\n".join(head + [marker] + tail), True, len(head) + len(tail), len(lines)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values() if allowed is None or t.permission in allowed]

    def execute(self, name: str, raw_args: Any, ctx: ToolContext) -> dict[str, Any]:
        """Validate and run one model-requested tool call. Never raises (except KeyboardInterrupt)."""
        tool = self._tools.get(name)
        if tool is None:
            return fail(name, f"Unknown tool '{name}'. Available tools: {', '.join(self._tools)}")
        if tool.permission not in MODE_PERMISSIONS.get(ctx.mode, set()):
            return fail(name, f"Tool '{name}' is not available in {ctx.mode} mode.")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            if args is None or args == "":
                args = {}
        except json.JSONDecodeError as exc:
            return fail(name, f"Tool arguments are not valid JSON: {exc}")
        try:
            args = validate_args(tool.parameters, args)
            data = tool.handler(ctx, **args)
        except ToolError as exc:
            return fail(name, str(exc))
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - tool crashes must not kill the agent
            get_logger().exception("tool %s crashed", name)
            return fail(name, f"{type(exc).__name__}: {exc}")
        return ok(name, data)

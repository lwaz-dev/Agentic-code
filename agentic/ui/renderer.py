"""Rich renderables: banner, tool activity, diffs, plan, summaries."""
from __future__ import annotations

import json
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DIFF_LINES = 30


def banner(console: Console, root: str, model: str, mode: str) -> None:
    title = Group(
        Text(""),
        Text("AGENTIC CODE", style="bold", justify="center"),
        Text(""),
        Text("Autonomous AI coding agent", style="dim", justify="center"),
        Text(""),
    )
    console.print(Align.center(Panel(title, box=box.ROUNDED, width=52, border_style="dim")))
    console.print()
    for label, value in (("Project", root), ("Model", model), ("Mode", mode.capitalize())):
        console.print(Text.assemble((f"{label:<8}", "dim"), (value, "")))
    console.print()
    console.print(Text("Type /help for commands.", style="dim"))
    console.rule(style="dim")


def describe_args(name: str, args: dict[str, Any]) -> str:
    if name in ("read_file",):
        rng = f":{args.get('start_line', '')}-{args.get('end_line', '')}" if "start_line" in args or "end_line" in args else ""
        return f"{args.get('path', '?')}{rng}"
    if name in ("write_file", "edit_file", "delete_file"):
        return str(args.get("path", "?"))
    if name == "list_files":
        return str(args.get("path", "."))
    if name == "search_files":
        where = f"  in {args['path']}" if args.get("path") not in (None, ".") else ""
        return f'"{args.get("pattern", "")}"{where}'
    if name == "run_command":
        return str(args.get("command", "")).splitlines()[0][:120] if args.get("command") else ""
    if name == "git_add":
        return " ".join(map(str, args.get("paths", [])))[:100]
    if name in ("git_commit",):
        return str(args.get("message", ""))[:100]
    if name == "git_checkout":
        return str(args.get("ref", ""))
    return json.dumps(args, ensure_ascii=False)[:100] if args else ""


def tool_start(console: Console, name: str, args: dict[str, Any]) -> None:
    console.print()
    console.print(Text.assemble(("● ", "cyan"), (name, "bold")))
    detail = describe_args(name, args)
    if detail:
        console.print(Text.assemble(("  └─ ", "dim"), (detail, "dim")))


def _result_line(name: str, data: dict[str, Any]) -> tuple[str, str] | None:
    if name == "list_files":
        return f"{data.get('count', 0)} entries", "dim"
    if name == "read_file":
        return f"{data.get('end_line', 0) - data.get('start_line', 1) + 1} of {data.get('total_lines', 0)} lines", "dim"
    if name == "search_files":
        return f"{data.get('count', 0)} matches", "dim"
    if name == "run_command":
        mark, style = ("✓", "green") if data.get("ok") else ("✗", "red")
        label = "timed out" if data.get("timed_out") else f"exit {data.get('exit_code')}"
        return f"{mark} {label} · {data.get('duration_s')}s", style
    if name == "project_info":
        return f"{data.get('language')} / {data.get('framework')}", "dim"
    return None


def render_diff(console: Console, diff: str, limit: int = DIFF_LINES) -> None:
    lines = diff.splitlines()
    for line in lines[:limit]:
        if line.startswith(("+++", "---")):
            continue
        style = "green" if line.startswith("+") else "red" if line.startswith("-") else "cyan dim" if line.startswith("@@") else "dim"
        console.print(Text("     " + line, style=style, no_wrap=True, overflow="ellipsis"))
    if len(lines) > limit:
        console.print(Text(f"     … {len(lines) - limit} more diff lines (use /diff)", style="dim"))


def tool_result(console: Console, name: str, result: dict[str, Any], changes: list) -> None:
    if not result.get("success"):
        console.print(Text(f"     ✗ {result.get('error', 'failed')}", style="red"))
        return
    data = result.get("data") or {}
    for change in changes:
        console.print(Text(f"     {change.action}  +{change.added} -{change.removed}", style="dim"))
        render_diff(console, change.diff)
    line = _result_line(name, data) if isinstance(data, dict) else None
    if line:
        console.print(Text("     " + line[0], style=line[1]))
    if name == "run_command" and not data.get("ok"):
        err = (data.get("stderr") or data.get("stdout") or "").strip().splitlines()[-6:]
        for ln in err:
            console.print(Text("     " + ln[:140], style="red dim"))


def plan(console: Console, steps: list) -> None:
    console.print()
    console.print(Text.assemble(("● ", "cyan"), ("Plan", "bold")))
    for i, step in enumerate(steps, 1):
        mark, style = {"done": ("✓", "green"), "in_progress": ("›", "yellow")}.get(step.status, (" ", "dim"))
        console.print(Text.assemble(("  ", ""), (mark, style), (f" {i}. {step.title}", "" if step.status != "pending" else "dim")))


def run_summary(console: Console, result) -> None:
    """End-of-task report. Printed only when tools were used."""
    console.print()
    console.rule(style="dim")
    if result.status == "completed":
        console.print(Text("✓ Task completed", style="bold green"))
    if result.changed:
        console.print()
        console.print(Text("Changed:", style="bold"))
        for path in result.created:
            console.print(Text(f"  + {path}", style="green"))
        for path in result.modified:
            console.print(Text(f"  ~ {path}", style="yellow"))
        for path in result.deleted:
            console.print(Text(f"  - {path}", style="red"))
    if result.validations:
        console.print()
        console.print(Text("Validation:", style="bold"))
        for command, passed in result.validations[-6:]:
            console.print(Text.assemble(("  ✓ " if passed else "  ✗ ", "green" if passed else "red"), (command[:90], "dim")))
    counts = f"{len(result.created)} created · {len(result.modified)} modified · {len(result.deleted)} deleted"
    console.print()
    console.print(Text(counts, style="dim"))
    console.rule(style="dim")


def table(console: Console, rows: list[tuple[str, str]], title: str | None = None) -> None:
    grid = Table(box=None, show_header=False, padding=(0, 2), title=title, title_justify="left")
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    for key, value in rows:
        grid.add_row(key, Text(str(value)))
    console.print(grid)

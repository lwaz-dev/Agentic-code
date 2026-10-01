"""Command-line entry point: `agentic [project_path]`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from agentic import __version__
from agentic.agent import Agent
from agentic.commands import CommandHandler
from agentic.config import ConfigError, load_config
from agentic.log import setup_logging
from agentic.ui import renderer
from agentic.ui.terminal import TerminalUI


def _configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", **({"encoding": "utf-8"} if sys.platform == "win32" else {}))
        except (AttributeError, ValueError):
            pass


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="agentic", description="Agentic Code - an autonomous AI coding agent for your terminal.")
    p.add_argument("project", nargs="?", default=".", help="project directory (default: current directory)")
    p.add_argument("-m", "--model", help="model id (overrides config)")
    p.add_argument("--mode", choices=["chat", "code", "plan", "review", "auto"], help="agent mode")
    p.add_argument("--approval", choices=["strict", "normal", "auto"], help="approval mode")
    p.add_argument("-p", "--prompt", help="run a single task non-interactively, then exit")
    p.add_argument("--debug", action="store_true", help="enable debug logging to .agentic/logs/")
    p.add_argument("--version", action="version", version=f"agentic-code {__version__}")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_streams()
    args = parse_args(argv)
    console = Console()
    root = Path(args.project).expanduser().resolve()
    if not root.is_dir():
        console.print(f"[red]✗[/] Project directory not found: {root}", markup=True, highlight=False)
        return 2
    try:
        config = load_config(root, {"model": args.model, "mode": args.mode, "approval_mode": args.approval,
                                    "debug": True if args.debug else None})
    except ConfigError as exc:
        console.print(f"✗ {exc}", style="red", markup=False)
        return 2
    setup_logging(config.debug, root)

    ui = TerminalUI(console, Path.home() / ".agentic" / "history")
    agent = Agent(config, root, ui=ui)
    commands = CommandHandler(agent, ui)

    if args.prompt:
        result = agent.run(args.prompt)
        ui.show_result(result)
        return 0 if result.status == "completed" else 1

    renderer.banner(console, str(root), config.model, config.mode)
    if not config.api_key:
        ui.warning("OPENROUTER_API_KEY is not set. Add it to .env (or ~/.agentic/.env), then restart. Run /doctor for help.")
    while True:
        try:
            text = ui.ask()
        except KeyboardInterrupt:
            continue
        if text is None:
            break
        text = text.strip()
        if not text:
            continue
        if text.startswith("/"):
            if commands.handle(text) == "exit":
                break
            continue
        ui.show_result(agent.run(text))
        console.rule(style="dim")
    console.print("Goodbye.", style="dim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

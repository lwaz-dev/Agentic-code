"""Slash commands."""
from __future__ import annotations

import os
from itertools import islice
from typing import Callable

from rich.markdown import Markdown
from rich.text import Text

from agentic.config import AGENT_MODES, APPROVAL_MODES, ConfigError, save_project_config
from agentic.providers import ProviderError
from agentic.tools.filesystem import session_changes, walk
from agentic.tools.git import run_git
from agentic.tools.project import detect_project, format_project_info, init_project
from agentic.tools.registry import ToolError
from agentic.ui import renderer
from agentic.ui.status import collect_status, run_doctor

HELP = """\
| Command | What it does |
|---|---|
| `/help` | Show this help |
| `/status` | Project, model, mode, git and session info |
| `/mode [name]` | Show or switch mode: chat, code, plan, review, auto |
| `/model [name]` | Show or change the model for this session |
| `/models [filter]` | List models available from the provider |
| `/project [path]` | Show project info, or switch the project root |
| `/files` | List project files |
| `/diff` | Show changes made in this session (or `git diff`) |
| `/history` | Show the conversation history |
| `/compact` | Summarise older context to free space |
| `/reset` | Clear the conversation and session changes |
| `/clear` | Clear the screen |
| `/config [set key value]` | Show or persist settings (.agentic/config.json) |
| `/init` | Create `.agentic/` project files |
| `/doctor` | Check your setup |
| `/exit`, `/quit` | Leave |

Tip: Alt+Enter (or a trailing `\\`) adds a new line; Enter sends. Ctrl+C stops the agent."""


class CommandHandler:
    def __init__(self, agent, ui) -> None:
        self.agent, self.ui, self.console = agent, ui, ui.console
        self.table: dict[str, Callable[[str], str | None]] = {
            "help": self.help, "?": self.help, "clear": self.clear, "exit": self.exit, "quit": self.exit,
            "status": self.status, "model": self.model, "models": self.models, "project": self.project,
            "files": self.files, "diff": self.diff, "reset": self.reset, "history": self.history,
            "config": self.config, "mode": self.mode, "init": self.init, "doctor": self.doctor,
            "compact": self.compact,
        }

    def handle(self, line: str) -> str | None:
        """Run a slash command. Returns 'exit' to quit."""
        name, _, arg = line.strip()[1:].partition(" ")
        handler = self.table.get(name.lower())
        if handler is None:
            self.ui.warning(f"Unknown command /{name}. Type /help.")
            return None
        try:
            return handler(arg.strip())
        except (ValueError, ConfigError, ProviderError, ToolError) as exc:
            self.ui.error(str(exc))
        return None

    # ---------------------------------------------------------------- commands
    def help(self, _: str) -> None:
        self.console.print(Markdown(HELP))

    def clear(self, _: str) -> None:
        self.console.clear()

    def exit(self, _: str) -> str:
        return "exit"

    def status(self, _: str) -> None:
        self.console.print(Text("Agentic Code", style="bold"))
        self.console.print()
        renderer.table(self.console, collect_status(self.agent))

    def mode(self, arg: str) -> None:
        if arg:
            self.agent.set_mode(arg)
            self.ui.info(f"Mode set to {self.agent.config.mode.capitalize()}.")
            return
        self.console.print(f"Current mode: {self.agent.config.mode.capitalize()}\n\nAvailable:")
        for name in AGENT_MODES:
            self.console.print(Text(f"- {name.capitalize()}"))

    def model(self, arg: str) -> None:
        if arg:
            self.agent.config.model = arg
            self.ui.info(f"Model set to {arg} for this session. Use /config set model {arg} to save it.")
        else:
            self.ui.info(f"Model: {self.agent.config.model}")

    def models(self, arg: str) -> None:
        with self.console.status("Fetching models..."):
            names = self.agent.get_provider().list_models()
        if arg:
            names = [n for n in names if arg.lower() in n.lower()]
        if not names:
            self.ui.info("No models found. Browse https://openrouter.ai/models")
            return
        for name in names[:60]:
            self.console.print(Text(name))
        if len(names) > 60:
            self.ui.info(f"... {len(names) - 60} more. Use /models <filter> to narrow.")

    def project(self, arg: str) -> None:
        if arg:
            self.agent.set_root(arg)
            os.chdir(self.agent.root)
            self.ui.info(f"Project root changed to {self.agent.root} (conversation reset).")
        info = detect_project(self.agent.root)
        self.console.print(Text(f"Project: {self.agent.root}", style="bold"))
        self.console.print(Text(format_project_info(info)))

    def files(self, _: str) -> None:
        shown = 0
        for path, is_dir in islice(walk(self.agent.root, self.agent.root), 400):
            self.console.print(Text(path + ("/" if is_dir else ""), style="dim" if is_dir else ""))
            shown += 1
        if shown >= 400:
            self.ui.info("... list truncated at 400 entries.")

    def diff(self, _: str) -> None:
        changes = session_changes(self.agent.ctx)
        if changes:
            for change in changes:
                self.console.print(Text(f"{change.path}  ({change.action})", style="bold"))
                renderer.render_diff(self.console, change.diff, limit=400)
            added, removed = sum(c.added for c in changes), sum(c.removed for c in changes)
            self.console.print(Text(f"\n{len(changes)} files changed  +{added}  -{removed}", style="dim"))
            return
        try:
            code, out, _err = run_git(self.agent.root, "diff", "--no-color")
        except ToolError:
            code, out = 1, ""
        if code == 0 and out.strip():
            renderer.render_diff(self.console, out, limit=400)
        else:
            self.ui.info("No changes.")

    def reset(self, _: str) -> None:
        self.agent.reset()
        self.ui.info("Conversation and session changes reset.")

    def history(self, _: str) -> None:
        rows = self.agent.conversation.history()
        if not rows:
            self.ui.info("History is empty.")
            return
        for i, (role, text) in enumerate(rows, 1):
            self.console.print(Text.assemble((f"{i:>3} {role:<9}", "dim"), (text, "")))

    def compact(self, _: str) -> None:
        before = self.agent.conversation.total_chars()
        self.agent.conversation.compact(force=True)
        self.ui.info(f"Context compacted: {before:,} -> {self.agent.conversation.total_chars():,} characters.")

    def config(self, arg: str) -> None:
        parts = arg.split(None, 2)
        if parts[:1] == ["set"]:
            if len(parts) < 3:
                raise ValueError("Usage: /config set <key> <value>")
            key, value = parts[1], parts[2]
            if key == "mode":
                self.agent.set_mode(value)
            else:
                value_cast: object = value
                if key in ("max_iterations", "command_timeout"):
                    value_cast = int(value)
                elif key in ("stream", "debug"):
                    value_cast = value.lower() in ("1", "true", "yes", "on")
                setattr(self.agent.config, key, value_cast)  # validated by pydantic
                value = value_cast  # type: ignore[assignment]
            path = save_project_config(self.agent.root, key, getattr(self.agent.config, key))
            self.ui.info(f"Saved {key} to {path}")
            return
        renderer.table(self.console, [(k, str(v)) for k, v in self.agent.config.public_dict().items()
                                      if not k.startswith("max_") or k == "max_iterations"])
        self.ui.info("\nChange with: /config set <key> <value>   (approval_mode: " + ", ".join(APPROVAL_MODES) + ")")

    def init(self, _: str) -> None:
        created = init_project(self.agent.root, self.agent.config)
        self.agent.project_info(refresh=True)
        if created:
            self.ui.info("Created: " + ", ".join(created))
        else:
            self.ui.info(".agentic/ already initialised.")
        self.ui.info("Edit .agentic/rules.md to give the agent project rules. Consider adding .agentic/logs/ to .gitignore.")

    def doctor(self, _: str) -> None:
        self.console.print(Text("Agentic Code Doctor", style="bold"))
        self.console.print()
        with self.console.status("Running checks..."):
            checks = run_doctor(self.agent)
        for c in checks:
            mark, style = {True: ("✓", "green"), False: ("✗", "red"), None: ("⚠", "yellow")}[c.ok]
            self.console.print(Text.assemble((f"{mark} ", style), (c.label, ""), (f"  {c.detail}" if c.detail else "", "dim")))
        self.console.print()
        self.console.print("Everything looks good." if all(c.ok is not False for c in checks) else "Some checks failed - see above.")

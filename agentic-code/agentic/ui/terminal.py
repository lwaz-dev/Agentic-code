"""TerminalUI: streaming markdown, spinner and tool activity on top of Rich."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from agentic.ui import NullUI, prompts, renderer


class TerminalUI(NullUI):
    def __init__(self, console: Console | None = None, history_path: Path | None = None) -> None:
        self.console = console or Console()
        self._status = None
        self._live: Live | None = None
        self._buffer = ""
        self._session = None
        self._history_path = history_path

    # ------------------------------------------------------------------ input
    def ask(self) -> str | None:
        if self._session is None and self.console.is_terminal:
            try:
                self._session = prompts.build_session(self._history_path)
            except Exception:  # noqa: BLE001 - fall back to plain input()
                self._session = None
        self.console.print()
        return prompts.read_line(self._session, self.console)

    def confirm(self, title: str, detail: str, warning: str = "") -> bool:
        self._stop_all()
        return prompts.ask_confirm(self.console, title, detail, warning)

    # -------------------------------------------------------------- streaming
    def thinking_start(self) -> None:
        if self.console.is_terminal and self._status is None and self._live is None:
            self._status = self.console.status("[cyan]●[/] Thinking", spinner="dots")
            self._status.start()

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def stream_text(self, chunk: str) -> None:
        if not chunk:
            return
        self._stop_status()
        if not self.console.is_terminal:
            self.console.print(chunk, end="", markup=False, highlight=False)
            self._buffer += chunk
            return
        if self._live is None:
            self.console.print()
            self._buffer = ""
            self._live = Live(Markdown(""), console=self.console, refresh_per_second=12, vertical_overflow="visible")
            self._live.start()
        self._buffer += chunk
        self._live.update(Markdown(self._buffer))

    def thinking_stop(self) -> None:
        self._stop_all()

    def _stop_all(self) -> None:
        self._stop_status()
        if self._live is not None:
            self._live.update(Markdown(self._buffer), refresh=True)
            self._live.stop()
            self._live = None
        elif self._buffer:
            self.console.print()
        self._buffer = ""

    # --------------------------------------------------------------- activity
    def tool_start(self, name: str, args: dict[str, Any]) -> None:
        self._stop_all()
        renderer.tool_start(self.console, name, args)

    def tool_result(self, name: str, result: dict[str, Any], changes: list) -> None:
        renderer.tool_result(self.console, name, result, changes)

    def plan(self, steps: list) -> None:
        self._stop_all()
        renderer.plan(self.console, steps)

    def info(self, message: str) -> None:
        self.console.print(Text(message))

    def warning(self, message: str) -> None:
        self.console.print(Text(f"⚠ {message}", style="yellow"))

    def error(self, message: str) -> None:
        self.console.print(Text(f"✗ {message}", style="red"))

    def show_result(self, result) -> None:
        """Final status block after an agent run."""
        if result.status == "interrupted":
            self.console.print()
            self.warning("Agent stopped by user.")
        elif result.status == "max_iterations":
            self.console.print()
            self.warning("Agent reached the maximum number of iterations.\n\n  The task may require additional input.")
            if result.changed or result.validations:
                renderer.run_summary(self.console, result)
        elif result.status == "error":
            self.console.print()
            self.error("Agent error")
            self.console.print(Text("\n" + (result.error or "Unknown error")))
            self.console.print(Text("\nCheck:\n- OPENROUTER_API_KEY\n- internet connection\n- model name (/model)\n- run /doctor for diagnostics", style="dim"))
        elif result.tool_calls:
            renderer.run_summary(self.console, result)

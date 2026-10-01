"""Interactive input (prompt-toolkit) and approval prompts."""
from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.text import Text


def build_session(history_path: Path | None = None):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        buf = event.current_buffer
        if buf.text.endswith("\\"):  # trailing backslash continues the line
            buf.delete_before_cursor(1)
            buf.insert_text("\n")
        else:
            buf.validate_and_handle()

    @kb.add("escape", "enter")  # Alt+Enter
    @kb.add("c-j")  # Ctrl+J
    def _newline(event):
        event.current_buffer.insert_text("\n")

    history = InMemoryHistory()
    if history_path:
        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(history_path))
        except OSError:
            pass
    return PromptSession(
        message=[("ansicyan bold", "❯ ")],
        multiline=True,
        key_bindings=kb,
        history=history,
        prompt_continuation=lambda width, line_number, wrap_count: "  ",
        enable_history_search=True,
    )


def read_line(session, console: Console) -> str | None:
    """Return the entered text, '' if cancelled with Ctrl+C, None on EOF (Ctrl+D)."""
    try:
        if session is None or not sys.stdin.isatty():
            return input("❯ ")
        return session.prompt()
    except KeyboardInterrupt:
        console.print(Text("(Ctrl+C) Type /exit or press Ctrl+D to quit.", style="dim"))
        return ""
    except EOFError:
        return None


def ask_confirm(console: Console, title: str, detail: str, warning: str = "") -> bool:
    console.print()
    console.print(Text(f"⚠ {title}", style="bold yellow"))
    console.print()
    console.print(Text("  " + detail.replace("\n", "\n  ")))
    if warning:
        console.print()
        console.print(Text("  " + warning, style="dim"))
    try:
        answer = console.input("\nAllow? [y/N] ", markup=False)
    except (KeyboardInterrupt, EOFError):
        console.print()
        return False
    return answer.strip().lower() in ("y", "yes")

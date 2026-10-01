"""The agent loop: model -> tool calls -> observe -> repeat until the task is done."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from agentic.context import Conversation
from agentic.log import get_logger
from agentic.providers.base import LLMProvider, ProviderError, ProviderResponse
from agentic.redact import redact
from agentic.tools.registry import MODE_PERMISSIONS, ToolContext, ToolRegistry

_VALIDATION = re.compile(
    r"php\s+-l|pytest|unittest|py_compile|compileall|npm\s+(run\s+)?(test|lint|build|typecheck)|npm\s+t\b|"
    r"composer\s+(test|validate)|phpunit|\btsc\b|eslint|ruff|flake8|mypy|go\s+(test|vet|build)|"
    r"cargo\s+(test|check|build)|node\s+(--check|-c)\b|jest|vitest|mvn\s+test|gradle\s+test|"
    r"dotnet\s+(test|build)|make\s+test",
    re.I,
)
_DOC_EXTS = {".md", ".txt", ".rst", ".csv", ".example", ".gitignore"}
NUDGE = (
    "[Agentic Code reminder] You changed files but have not run any validation since. "
    "Run an appropriate check now (syntax check, tests or linter), or briefly explain why none is possible."
)


@dataclass
class RunResult:
    status: str = "completed"  # completed | max_iterations | interrupted | error
    iterations: int = 0
    final_text: str = ""
    tool_calls: int = 0
    created: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    validations: list[tuple[str, bool]] = field(default_factory=list)
    error: str | None = None

    @property
    def changed(self) -> list[str]:
        return self.created + self.modified + self.deleted


def looks_like_validation(command: str) -> bool:
    return bool(_VALIDATION.search(command))


def serialize_result(result: dict[str, Any], limit: int = 40_000) -> str:
    text = json.dumps(result, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return json.dumps(
        {"success": result.get("success"), "tool": result.get("tool"), "truncated": True, "data": text[:limit]},
        ensure_ascii=False,
    )


class AgentLoop:
    def __init__(
        self,
        provider_factory: Callable[[], LLMProvider],
        registry: ToolRegistry,
        conversation: Conversation,
        ctx: ToolContext,
        system_prompt: Callable[[], str],
    ) -> None:
        self.provider_factory = provider_factory
        self.registry = registry
        self.conversation = conversation
        self.ctx = ctx
        self.system_prompt = system_prompt

    def run(self, user_text: str) -> RunResult:
        ctx, conv, ui = self.ctx, self.conversation, self.ctx.ui
        result = RunResult()
        start = len(ctx.changes)
        conv.add_user(user_text)
        conv.current_task = user_text[:500]
        allowed = MODE_PERMISSIONS.get(ctx.mode, set())
        tools = self.registry.schemas(allowed) or None
        dirty, nudged = False, False
        log = get_logger()
        try:
            provider = self.provider_factory()
            for iteration in range(1, ctx.config.max_iterations + 1):
                result.iterations = iteration
                conv.compact()
                messages = [{"role": "system", "content": self.system_prompt()}] + conv.messages
                ui.thinking_start()
                try:
                    response: ProviderResponse = provider.chat(messages, tools, on_text=ui.stream_text)
                finally:
                    ui.thinking_stop()
                conv.add_assistant(response)
                if response.tool_calls:
                    for validated, edited in self._run_tools(response, result):
                        if validated:
                            dirty = False
                        elif edited:
                            dirty = True
                    continue
                if dirty and not nudged and ctx.mode in ("code", "auto") and self._needs_validation(ctx, start):
                    nudged = True
                    conv.add_user(NUDGE)
                    continue
                result.final_text = response.content
                break
            else:
                result.status = "max_iterations"
        except KeyboardInterrupt:
            result.status = "interrupted"
        except ProviderError as exc:
            result.status, result.error = "error", str(exc)
            log.error("provider error: %s", exc)
        self._finalize_changes(result, start)
        return result

    # ------------------------------------------------------------------ tools
    def _run_tools(self, response: ProviderResponse, result: RunResult):
        ctx, conv, ui = self.ctx, self.conversation, self.ctx.ui
        log = get_logger()
        for index, call in enumerate(response.tool_calls):
            try:
                args = json.loads(call.arguments) if call.arguments else {}
            except json.JSONDecodeError:
                args = {}
            before = len(ctx.changes)
            try:
                if call.name != "update_plan":
                    ui.tool_start(call.name, args if isinstance(args, dict) else {})
                log.debug("tool call %s %s", call.name, redact(call.arguments))
                outcome = self.registry.execute(call.name, call.arguments, ctx)
            except KeyboardInterrupt:
                interrupted = {"success": False, "tool": "", "error": "Interrupted by user."}
                for pending in response.tool_calls[index:]:
                    conv.add_tool(pending.id, pending.name, json.dumps({**interrupted, "tool": pending.name}))
                raise
            result.tool_calls += 1
            if call.name != "update_plan":
                ui.tool_result(call.name, outcome, ctx.changes[before:])
            conv.add_tool(call.id, call.name, serialize_result(outcome))
            validated = False
            if call.name == "run_command" and outcome.get("success"):
                command = str(args.get("command", ""))
                if looks_like_validation(command):
                    validated = True
                    result.validations.append((command, bool(outcome["data"].get("ok"))))
            yield validated, len(ctx.changes) > before

    @staticmethod
    def _needs_validation(ctx: ToolContext, start: int) -> bool:
        from pathlib import PurePosixPath

        for change in ctx.changes[start:]:
            name = PurePosixPath(change.path)
            if name.suffix.lower() not in _DOC_EXTS and name.name.upper() not in ("LICENSE", "README"):
                return True
        return False

    def _finalize_changes(self, result: RunResult, start: int) -> None:
        net: dict[str, list[str]] = {}
        for change in self.ctx.changes[start:]:
            net.setdefault(change.path, []).append(change.action)
        for path, actions in net.items():
            if actions[-1] == "deleted":
                if actions[0] != "created":
                    result.deleted.append(path)
            elif actions[0] == "created":
                result.created.append(path)
            else:
                result.modified.append(path)

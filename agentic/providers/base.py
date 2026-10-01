"""Provider-agnostic LLM interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class ProviderError(Exception):
    """A user-presentable provider failure."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model


@dataclass
class ProviderResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None

    def to_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return msg


class LLMProvider:
    """Implement this to add a new backend (OpenAI, Anthropic, Ollama, ...)."""

    name = "base"

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        """Send messages (+ tool schemas). Stream text through on_text when supported."""
        raise NotImplementedError

    def list_models(self) -> list[str]:
        return []

    def check_connection(self) -> tuple[bool, str]:
        return True, "not checked"

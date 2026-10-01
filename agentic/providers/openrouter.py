"""OpenRouter provider (OpenAI-compatible API)."""
from __future__ import annotations

import uuid
from typing import Any, Callable

import openai

from agentic.providers.base import LLMProvider, ProviderError, ProviderResponse, ToolCall


class OpenRouterProvider(LLMProvider):
    name = "openrouter"

    def __init__(self, config: Any, client: Any = None) -> None:
        self.config = config  # read model/stream live so /model takes effect immediately
        if client is None:
            if not config.api_key:
                raise ProviderError("OPENROUTER_API_KEY is not set.")
            client = openai.OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=120.0,
                max_retries=2,
                default_headers={"HTTP-Referer": "https://github.com/agentic-code", "X-Title": "Agentic Code"},
            )
        self.client = client

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        kwargs: dict[str, Any] = {"model": self.config.model, "messages": messages}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = "auto"
        try:
            if self.config.stream:
                return self._stream(kwargs, on_text)
            return self._complete(kwargs, on_text)
        except ProviderError:
            raise
        except openai.AuthenticationError as exc:
            raise ProviderError("Authentication failed. Check OPENROUTER_API_KEY.") from exc
        except openai.NotFoundError as exc:
            raise ProviderError(f"Model or endpoint not found. Check the model name: {self.config.model}") from exc
        except openai.RateLimitError as exc:
            raise ProviderError("Rate limited by the provider. Wait a moment or switch models.") from exc
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            raise ProviderError("Unable to connect to OpenRouter.") from exc
        except openai.APIStatusError as exc:
            detail = getattr(exc, "message", "") or str(exc)
            raise ProviderError(f"Provider error ({exc.status_code}): {detail[:300]}") from exc

    def _stream(self, kwargs: dict[str, Any], on_text: Callable[[str], None] | None) -> ProviderResponse:
        stream = self.client.chat.completions.create(stream=True, **kwargs)
        text: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish = None
        try:
            for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.delta
                if getattr(delta, "content", None):
                    text.append(delta.content)
                    if on_text:
                        on_text(delta.content)
                for tc in getattr(delta, "tool_calls", None) or []:
                    slot = calls.setdefault(tc.index or 0, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn:
                        if fn.name and not slot["name"]:
                            slot["name"] = fn.name
                        if fn.arguments:
                            slot["arguments"] += fn.arguments
                if choice.finish_reason:
                    finish = choice.finish_reason
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()
        return ProviderResponse("".join(text), self._calls(calls), finish)

    def _complete(self, kwargs: dict[str, Any], on_text: Callable[[str], None] | None) -> ProviderResponse:
        resp = self.client.chat.completions.create(**kwargs)
        if not resp.choices:
            raise ProviderError("The provider returned an empty response. Try another model.")
        choice = resp.choices[0]
        content = choice.message.content or ""
        if content and on_text:
            on_text(content)
        calls = {
            i: {"id": tc.id or "", "name": tc.function.name, "arguments": tc.function.arguments or ""}
            for i, tc in enumerate(choice.message.tool_calls or [])
        }
        return ProviderResponse(content, self._calls(calls), choice.finish_reason)

    @staticmethod
    def _calls(calls: dict[int, dict[str, str]]) -> list[ToolCall]:
        return [
            ToolCall(c["id"] or f"call_{uuid.uuid4().hex[:12]}", c["name"], c["arguments"] or "{}")
            for _, c in sorted(calls.items())
        ]

    # ------------------------------------------------------------- utilities
    def list_models(self) -> list[str]:
        try:
            page = self.client.with_options(timeout=15).models.list()
            return sorted(m.id for m in page)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Could not fetch the model list: {exc}") from exc

    def check_connection(self) -> tuple[bool, str]:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/auth/key",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10):
                return True, "connected, API key accepted"
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return False, "API key rejected"
            return False, f"unexpected status {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError):
            return False, "cannot reach the provider (check your internet connection)"

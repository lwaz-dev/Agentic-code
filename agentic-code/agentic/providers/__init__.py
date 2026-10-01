from __future__ import annotations

from typing import Any

from agentic.providers.base import LLMProvider, ProviderError, ProviderResponse, ToolCall
from agentic.providers.openrouter import OpenRouterProvider

PROVIDERS = {"openrouter": OpenRouterProvider}


def create_provider(config: Any) -> LLMProvider:
    try:
        return PROVIDERS[config.provider](config)
    except KeyError:
        raise ProviderError(
            f"Unknown provider '{config.provider}'. Available: {', '.join(PROVIDERS)}"
        ) from None


__all__ = ["LLMProvider", "ProviderError", "ProviderResponse", "ToolCall", "create_provider", "PROVIDERS"]

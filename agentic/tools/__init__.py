from __future__ import annotations

from agentic.tools import filesystem, git, project, search, terminal
from agentic.tools.registry import MODE_PERMISSIONS, Tool, ToolContext, ToolError, ToolRegistry


def build_registry() -> ToolRegistry:
    from agentic import planner  # local import: planner depends on tools.registry

    registry = ToolRegistry()
    for module in (filesystem, search, terminal, git, project, planner):
        for tool in module.TOOLS:
            registry.register(tool)
    return registry


__all__ = ["build_registry", "ToolRegistry", "ToolContext", "ToolError", "Tool", "MODE_PERMISSIONS"]

"""Concise, user-visible task plan (the agent's action plan, not its private reasoning)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic.tools.registry import Tool, ToolContext, ToolError

STATUSES = ("pending", "in_progress", "done")


@dataclass
class PlanStep:
    title: str
    status: str = "pending"


@dataclass
class Planner:
    steps: list[PlanStep] = field(default_factory=list)

    def update(self, items: list[Any]) -> list[PlanStep]:
        steps: list[PlanStep] = []
        for item in items[:12]:
            if isinstance(item, str):
                steps.append(PlanStep(item.strip()[:120]))
            elif isinstance(item, dict) and str(item.get("title", "")).strip():
                status = item.get("status", "pending")
                steps.append(PlanStep(str(item["title"]).strip()[:120], status if status in STATUSES else "pending"))
        if not steps:
            raise ToolError("Provide at least one plan step.")
        self.steps = steps
        return steps

    def clear(self) -> None:
        self.steps = []


def update_plan(ctx: ToolContext, steps: list):
    planner = ctx.planner or Planner()
    ctx.planner = planner
    result = planner.update(steps)
    ctx.ui.plan(result)
    return {"steps": len(result), "done": sum(s.status == "done" for s in result)}


TOOLS = [
    Tool(
        "update_plan",
        "Show/refresh a short action plan (3-8 concise steps) for multi-step tasks. Call it before starting and again as steps complete.",
        {"type": "object", "properties": {"steps": {"type": "array", "items": {
            "type": "object",
            "properties": {"title": {"type": "string"},
                           "status": {"type": "string", "enum": list(STATUSES)}},
            "required": ["title"]}}},
         "required": ["steps"]},
        update_plan, "read",
    )
]

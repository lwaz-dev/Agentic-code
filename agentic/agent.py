"""Agent facade: wires config, provider, tools, permissions, context and the loop together."""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from agentic.config import AGENT_MODES, Config
from agentic.context import Conversation
from agentic.log import get_logger
from agentic.loop import AgentLoop, RunResult
from agentic.permissions import PermissionManager
from agentic.planner import Planner
from agentic.prompts import build_system_prompt
from agentic.providers import LLMProvider, ProviderError, create_provider
from agentic.tools import ToolContext, build_registry
from agentic.tools.project import detect_project, format_project_info, load_project_rules
from agentic.tools.terminal import detect_shell
from agentic.ui import NullUI


class Agent:
    def __init__(self, config: Config, root: str | Path, provider: LLMProvider | None = None, ui: Any = None) -> None:
        self.config = config
        self.root = Path(root).resolve()
        self.ui = ui or NullUI()
        self._provider = provider
        self.registry = build_registry()
        self.permissions = PermissionManager(self._effective_approval, self.ui.confirm)
        self.planner = Planner()
        self.conversation = Conversation(config.max_context_chars)
        self.ctx = ToolContext(
            root=self.root, config=config, permissions=self.permissions, ui=self.ui,
            mode=config.mode, planner=self.planner,
        )
        self._project_info: dict[str, Any] | None = None
        self.loop = AgentLoop(self.get_provider, self.registry, self.conversation, self.ctx, self.system_prompt)
        self.last_iterations = 0

    # --------------------------------------------------------------- helpers
    def _effective_approval(self) -> str:
        return "auto" if self.config.mode == "auto" else self.config.approval_mode

    def get_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = create_provider(self.config)
        return self._provider

    def reset_provider(self) -> None:
        self._provider = None

    def project_info(self, refresh: bool = False) -> dict[str, Any]:
        if self._project_info is None or refresh:
            self._project_info = detect_project(self.root)
        return self._project_info

    def system_prompt(self) -> str:
        return build_system_prompt(
            mode=self.config.mode,
            root=str(self.root),
            shell=detect_shell(self.config),
            project_info=format_project_info(self.project_info()),
            rules=load_project_rules(self.root),
            summary=self.conversation.summary,
            current_task=self.conversation.current_task,
        )

    # ------------------------------------------------------------- operations
    def set_mode(self, mode: str) -> None:
        mode = mode.lower()
        if mode not in AGENT_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Available: {', '.join(AGENT_MODES)}")
        self.config.mode = mode
        self.ctx.mode = mode

    def set_root(self, path: str | Path) -> None:
        new_root = Path(path).expanduser().resolve()
        if not new_root.is_dir():
            raise ValueError(f"Not a directory: {new_root}")
        self.root = self.ctx.root = new_root
        self.reset()
        self._project_info = None

    def reset(self) -> None:
        self.conversation.reset()
        self.planner.clear()
        self.ctx.read_files.clear()
        self.ctx.redacted_files.clear()
        self.ctx.backups.clear()
        self.ctx.changes.clear()
        self.last_iterations = 0

    def run(self, text: str) -> RunResult:
        """Handle one user task. Never raises; failures are returned in RunResult."""
        try:
            result = self.loop.run(text)
        except Exception as exc:  # noqa: BLE001
            get_logger().error("unexpected error: %s\n%s", exc, traceback.format_exc())
            result = RunResult(status="error", error=f"Unexpected error: {exc}")
            if self.config.debug:
                result.error += "\n" + traceback.format_exc()
        self.last_iterations = result.iterations
        return result

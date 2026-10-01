import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic.config import Config
from agentic.permissions import PermissionManager
from agentic.providers.base import LLMProvider, ProviderResponse, ToolCall
from agentic.tools import ToolContext, build_registry
from agentic.ui import NullUI


class FakeProvider(LLMProvider):
    """Scripted provider: each chat() pops the next ProviderResponse."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools=None, on_text=None):
        self.calls.append((list(messages), tools))
        response = self.script.pop(0)
        if on_text and response.content:
            on_text(response.content)
        return response


def call(name, call_id="c1", **args):
    return ToolCall(call_id, name, json.dumps(args))


def reply(text="", *calls):
    return ProviderResponse(text, list(calls), "tool_calls" if calls else "stop")


class AnswerUI(NullUI):
    """Records what was shown and answers approval prompts with a fixed value."""

    def __init__(self, answer=True):
        self.answer, self.prompts = answer, []

    def confirm(self, title, detail, warning=""):
        self.prompts.append((title, detail))
        return self.answer


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    return root


def make_ctx(root, approval="normal", mode="code", answer=True):
    config = Config(approval_mode=approval, mode=mode)
    ui = AnswerUI(answer)
    perms = PermissionManager(lambda: approval, ui.confirm)
    return ToolContext(root=root, config=config, permissions=perms, ui=ui, mode=mode)


@pytest.fixture
def ctx(project):
    return make_ctx(project)


@pytest.fixture
def registry():
    return build_registry()

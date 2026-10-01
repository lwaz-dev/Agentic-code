import json
from types import SimpleNamespace

import pytest

from agentic.agent import Agent
from agentic.config import Config, ConfigError, load_config, save_project_config
from agentic.context import Conversation
from agentic.providers.base import ProviderError, ProviderResponse
from agentic.providers.openrouter import OpenRouterProvider
from agentic.redact import redact
from conftest import AnswerUI, FakeProvider, call, reply


def make_agent(project, script, mode="code", approval="normal", answer=True, **cfg):
    config = Config(mode=mode, approval_mode=approval, **cfg)
    return Agent(config, project, provider=FakeProvider(script), ui=AnswerUI(answer))


def test_multi_step_task_inspect_modify_validate(project):
    (project / "app.py").write_text("def add(a, b):\n    return a - b\n")
    script = [
        reply("Inspecting.", call("list_files", "1")),
        reply("", call("read_file", "2", path="app.py")),
        reply("", call("edit_file", "3", path="app.py", old_text="a - b", new_text="a + b")),
        reply("", call("run_command", "4", command="python -m py_compile app.py")),
        reply("Fixed the bug; py_compile passes."),
    ]
    agent = make_agent(project, script, approval="auto")
    result = agent.run("fix add()")
    assert result.status == "completed" and result.iterations == 5 and result.tool_calls == 4
    assert "a + b" in (project / "app.py").read_text()
    assert result.modified == ["app.py"] and result.validations == [("python -m py_compile app.py", True)]
    # tool results were fed back to the model
    last_messages = agent.get_provider().calls[-1][0]
    assert [m["role"] for m in last_messages].count("tool") == 4


def test_failed_validation_is_observed_and_fixed(project):
    script = [
        reply("", call("write_file", "1", path="bad.py", content="def f(:\n")),
        reply("", call("run_command", "2", command="python -m py_compile bad.py")),
        reply("", call("write_file", "3", path="bad.py", content="def f():\n    pass\n")),
        reply("", call("run_command", "4", command="python -m py_compile bad.py")),
        reply("Done."),
    ]
    agent = make_agent(project, script, approval="auto", answer=True)
    result = agent.run("create bad.py")
    assert [ok for _, ok in result.validations] == [False, True]
    tool_msgs = [m for m in agent.get_provider().calls[2][0] if m["role"] == "tool"]
    assert json.loads(tool_msgs[-1]["content"])["data"]["exit_code"] != 0


def test_unvalidated_changes_trigger_one_nudge(project):
    script = [
        reply("", call("write_file", "1", path="x.py", content="x = 1\n")),
        reply("All done."),  # no validation -> nudged
        reply("Could not validate; no test runner here."),
    ]
    agent = make_agent(project, script)
    result = agent.run("make x.py")
    assert result.iterations == 3
    nudges = [m for m in agent.conversation.messages if m["role"] == "user" and "reminder" in m["content"]]
    assert len(nudges) == 1


def test_docs_only_changes_are_not_nudged(project):
    script = [reply("", call("write_file", "1", path="NOTES.md", content="# hi\n")), reply("Done.")]
    assert make_agent(project, script).run("notes").iterations == 2


def test_max_iterations(project):
    script = [reply("", call("list_files", f"c{i}")) for i in range(5)]
    agent = make_agent(project, script, max_iterations=3)
    result = agent.run("loop forever")
    assert result.status == "max_iterations" and result.iterations == 3


def test_rejected_tool_call_is_returned_to_model(project):
    script = [reply("", call("read_file", "1", path="../../etc/passwd")), reply("Understood.")]
    agent = make_agent(project, script)
    agent.run("read secrets")
    tool_msg = [m for m in agent.get_provider().calls[1][0] if m["role"] == "tool"][0]
    payload = json.loads(tool_msg["content"])
    assert payload["success"] is False and "outside project root" in payload["error"]


def test_malformed_tool_call_does_not_crash(project):
    bad = ProviderResponse("", [SimpleNamespace(id="1", name="read_file", arguments="{oops")], "tool_calls")
    agent = make_agent(project, [bad, reply("ok")])
    assert agent.run("x").status == "completed"


def test_denied_dangerous_command_is_not_executed(project):
    (project / "keep.txt").write_text("x")
    script = [reply("", call("run_command", "1", command="rm -rf .")), reply("I was not allowed to.")]
    agent = make_agent(project, script, approval="auto", answer=False)
    agent.run("clean up")
    assert (project / "keep.txt").exists()
    assert agent.ui.prompts and "rm -rf" in agent.ui.prompts[0][1]


def test_plan_mode_is_read_only_and_exposes_only_read_tools(project):
    script = [reply("", call("write_file", "1", path="a.txt", content="x")), reply("Plan.")]
    agent = make_agent(project, script, mode="plan")
    agent.run("plan it")
    assert not (project / "a.txt").exists()
    tool_names = {t["name"] for t in agent.get_provider().calls[0][1]}
    assert "read_file" in tool_names and not tool_names & {"write_file", "edit_file", "run_command", "git_commit"}
    assert "not available in plan mode" in agent.conversation.messages[2]["content"]


def test_chat_mode_has_no_tools(project):
    agent = make_agent(project, [reply("Hello!")], mode="chat")
    agent.run("hi")
    assert agent.get_provider().calls[0][1] is None


def test_system_prompt_contains_rules_and_project_info(project):
    (project / ".agentic").mkdir()
    (project / ".agentic" / "rules.md").write_text("Use PDO for database access.")
    (project / "index.php").write_text("<?php // mysqli\n")
    agent = make_agent(project, [reply("ok")])
    agent.run("hi")
    system = agent.get_provider().calls[0][0][0]["content"]
    assert "Use PDO for database access." in system and "Agentic Code" in system
    assert "Language: PHP" in system and "MySQL" in system


def test_provider_error_is_reported_not_raised(project):
    class Boom(FakeProvider):
        def chat(self, *a, **k):
            raise ProviderError("Unable to connect to OpenRouter.")

    agent = Agent(Config(), project, provider=Boom([]), ui=AnswerUI())
    result = agent.run("hi")
    assert result.status == "error" and "connect" in result.error


def test_keyboard_interrupt_leaves_valid_history(project):
    class Slow(FakeProvider):
        def chat(self, messages, tools=None, on_text=None):
            self.calls.append(1)
            if len(self.calls) == 1:
                return reply("", call("list_files", "1"), call("read_file", "2", path="a"))
            raise KeyboardInterrupt

    agent = Agent(Config(), project, provider=Slow([]), ui=AnswerUI())
    result = agent.run("go")
    assert result.status == "interrupted"
    roles = [m["role"] for m in agent.conversation.messages]
    assert roles == ["user", "assistant", "tool", "tool"]  # every tool call has a result


def test_set_mode_validation_and_reset(project):
    agent = make_agent(project, [])
    agent.set_mode("review")
    assert agent.ctx.mode == "review"
    with pytest.raises(ValueError):
        agent.set_mode("nope")
    agent.conversation.add_user("x")
    agent.reset()
    assert agent.conversation.messages == []


# ------------------------------------------------------------- context / config
def test_context_compaction_keeps_valid_structure():
    conv = Conversation(max_chars=3000)
    for i in range(30):
        conv.add_user(f"task {i}")
        conv.add_assistant(reply("", call("read_file", f"id{i}", path=f"f{i}.py")))
        conv.add_tool(f"id{i}", "read_file", json.dumps({"success": True, "data": "x" * 400}))
    assert conv.compact()
    assert conv.total_chars() < 3500 + 500
    assert conv.summary and "read_file" in conv.summary
    msgs = conv.messages
    assert msgs[0]["role"] in ("user",)
    for i, m in enumerate(msgs):  # no orphaned tool results
        if m["role"] == "tool":
            assert msgs[i - 1]["role"] in ("assistant", "tool")


def test_force_compact_and_reset():
    conv = Conversation()
    for i in range(10):
        conv.add_user("hello " * 50)
        conv.add_assistant(ProviderResponse("answer " * 50))
    before = conv.total_chars()
    conv.compact(force=True)
    assert conv.total_chars() < before
    conv.reset()
    assert conv.messages == [] and conv.summary == ""


def test_config_precedence(project, monkeypatch):
    monkeypatch.delenv("AGENTIC_MODEL", raising=False)
    monkeypatch.setenv("HOME", str(project))
    assert load_config(project).model == "openai/gpt-oss-20b:free"
    (project / ".agentic").mkdir()
    (project / ".agentic" / "config.json").write_text(json.dumps({"model": "from/file", "max_iterations": 7}))
    assert load_config(project).model == "from/file" and load_config(project).max_iterations == 7
    monkeypatch.setenv("AGENTIC_MODEL", "from/env")
    assert load_config(project).model == "from/env"
    assert load_config(project, {"model": "from/cli"}).model == "from/cli"


def test_config_errors_and_api_key(project, monkeypatch):
    monkeypatch.setenv("HOME", str(project))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abcdefghijklmnop")
    cfg = load_config(project)
    assert cfg.api_key.startswith("sk-or") and "sk-or" not in repr(cfg)
    assert "sk-or" not in str(cfg.public_dict())
    assert "[REDACTED]" in redact("key is sk-or-v1-abcdefghijklmnop")
    (project / ".agentic" / "config.json").parent.mkdir(exist_ok=True)
    (project / ".agentic" / "config.json").write_text("{bad json")
    with pytest.raises(ConfigError):
        load_config(project)
    (project / ".agentic" / "config.json").write_text(json.dumps({"approval_mode": "yolo"}))
    with pytest.raises(ConfigError):
        load_config(project)


def test_save_project_config_never_saves_api_key(project):
    save_project_config(project, "model", "a/b")
    assert json.loads((project / ".agentic" / "config.json").read_text()) == {"model": "a/b"}
    with pytest.raises(ConfigError):
        save_project_config(project, "api_key", "x")


@pytest.mark.parametrize("text,hidden", [
    ("DB_PASSWORD=hunter2hunter2", "hunter2hunter2"),
    ('"api_key": "abcd1234efgh"', "abcd1234efgh"),
    ("mysql://root:s3cretpass@localhost/db", "s3cretpass"),
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----", "MIIabc"),
    ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz"),
])
def test_redaction(text, hidden):
    assert hidden not in redact(text)


def test_redaction_leaves_ordinary_code_alone():
    code = "$password = $_POST['password'];\nself.api_key = api_key\ntoken = get_token()\nif (!password_verify($p, $h)) {}"
    assert redact(code) == code


# ------------------------------------------------------------------- provider
def _chunk(content=None, tool_calls=None, finish=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


def _tc(index, id=None, name=None, args=None):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=args))


class FakeClient:
    def __init__(self, chunks):
        self.chunks, self.kwargs = chunks, None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.chunks)


def test_openrouter_streams_text_and_assembles_tool_calls():
    chunks = [
        _chunk("I'll "), _chunk("look."),
        _chunk(tool_calls=[_tc(0, "call_1", "read_file", '{"pa')]),
        _chunk(tool_calls=[_tc(0, None, None, 'th": "a.py"}')]),
        _chunk(tool_calls=[_tc(1, "call_2", "list_files", "{}")], finish="tool_calls"),
        SimpleNamespace(choices=[]),  # usage-only chunk
    ]
    client = FakeClient(chunks)
    provider = OpenRouterProvider(Config(model="some/model"), client=client)
    streamed = []
    resp = provider.chat([{"role": "user", "content": "x"}], [{"name": "read_file", "description": "d", "parameters": {}}], streamed.append)
    assert "".join(streamed) == "I'll look." == resp.content
    assert [(c.id, c.name, c.arguments) for c in resp.tool_calls] == [
        ("call_1", "read_file", '{"path": "a.py"}'), ("call_2", "list_files", "{}")]
    assert client.kwargs["model"] == "some/model" and client.kwargs["stream"] is True
    assert client.kwargs["tools"][0]["type"] == "function"
    provider.config.model = "other/model"  # model changes apply live
    provider.chat([], None)
    assert client.kwargs["model"] == "other/model" and "tools" not in client.kwargs


def test_openrouter_requires_api_key():
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider(Config(api_key=""))


def test_openrouter_maps_errors():
    import openai

    try:  # newer openai releases use httpx2
        import httpx2 as httpx
    except ImportError:
        import httpx

    class Failing:
        def __init__(self, exc):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **k: (_ for _ in ()).throw(exc)))

    req = httpx.Request("POST", "https://x")
    auth = openai.AuthenticationError("no", response=httpx.Response(401, request=req), body=None)
    conn = openai.APIConnectionError(request=req)
    for exc, text in ((auth, "Authentication"), (conn, "Unable to connect")):
        provider = OpenRouterProvider(Config(), client=Failing(exc))
        with pytest.raises(ProviderError, match=text):
            provider.chat([], None)

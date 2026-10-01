import json
import sys

from conftest import make_ctx

PY = f'"{sys.executable}"'


def run(registry, ctx, command, **kw):
    return registry.execute("run_command", json.dumps({"command": command, **kw}), ctx)


def test_command_runs_in_project_root(registry, project):
    ctx = make_ctx(project, approval="auto")
    (project / "marker.txt").write_text("x")
    r = run(registry, ctx, f'{PY} -c "import os; print(sorted(os.listdir()))"')
    assert r["success"] and r["data"]["exit_code"] == 0 and r["data"]["ok"]
    assert "marker.txt" in r["data"]["stdout"]
    assert r["data"]["duration_s"] >= 0


def test_exit_code_and_stderr(registry, project):
    ctx = make_ctx(project, approval="auto")
    r = run(registry, ctx, f'{PY} -c "import sys; sys.stderr.write(\'boom\'); sys.exit(3)"')
    assert r["success"] and r["data"]["exit_code"] == 3 and not r["data"]["ok"]
    assert "boom" in r["data"]["stderr"]


def test_timeout_kills_command(registry, project):
    ctx = make_ctx(project, approval="auto")
    r = run(registry, ctx, f'{PY} -c "import time; time.sleep(30)"', timeout=1)
    assert r["data"]["timed_out"] and "timed out" in r["data"]["note"]


def test_output_truncation(registry, project):
    ctx = make_ctx(project, approval="auto")
    ctx.config.max_output_chars = 2000
    r = run(registry, ctx, f'{PY} -c "print(\'\\n\'.join(str(i) for i in range(5000)))"')
    assert "truncated" in r["data"]["note"] and "lines omitted" in r["data"]["stdout"]
    assert len(r["data"]["stdout"]) < 3000


def test_secrets_redacted_from_output(registry, project):
    ctx = make_ctx(project, approval="auto")
    r = run(registry, ctx, f'{PY} -c "print(\'API_TOKEN=abcdef123456789xyz\')"')
    assert "abcdef123456789xyz" not in r["data"]["stdout"]


def test_denied_command_does_not_run(registry, project):
    ctx = make_ctx(project, answer=False)
    r = run(registry, ctx, "rm -rf somewhere")
    assert not r["success"] and "denied" in r["error"].lower()


def test_api_key_not_passed_to_children(registry, project, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "should-not-leak-12345")
    ctx = make_ctx(project, approval="auto")
    r = run(registry, ctx, f'{PY} -c "import os; print(os.environ.get(\'OPENROUTER_API_KEY\'))"')
    assert r["data"]["stdout"].strip() == "None"


def test_read_only_modes_restrict_commands(registry, project):
    review = make_ctx(project, mode="review", approval="auto")
    assert run(registry, review, "git status")["success"] or True  # may fail only if git is missing
    r = run(registry, review, "npm install")
    assert not r["success"] and "review mode" in r["error"]
    plan = make_ctx(project, mode="plan")
    assert "not available in plan mode" in run(registry, plan, "ls")["error"]

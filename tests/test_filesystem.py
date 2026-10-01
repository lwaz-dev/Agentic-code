import os

import pytest

from agentic.tools.filesystem import resolve_path
from agentic.tools.registry import ToolError
from conftest import make_ctx


def run(registry, ctx, name, **args):
    import json

    return registry.execute(name, json.dumps(args), ctx)


def test_path_traversal_rejected(ctx):
    for bad in ("../../important.txt", "/etc/passwd", "a/../../x"):
        with pytest.raises(ToolError, match="outside project root"):
            resolve_path(ctx.root, bad)
    with pytest.raises(ToolError):
        resolve_path(ctx.root, "")


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlink_escape_rejected(ctx, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    (ctx.root / "link").symlink_to(outside)
    with pytest.raises(ToolError, match="outside project root"):
        resolve_path(ctx.root, "link")


def test_write_then_read(registry, ctx):
    r = run(registry, ctx, "write_file", path="src/app.py", content="print('hi')\n")
    assert r["success"] and r["data"]["created"]
    r = run(registry, ctx, "read_file", path="src/app.py")
    assert r["data"]["content"] == "print('hi')"
    assert (ctx.root / "src" / "app.py").read_text() == "print('hi')\n"


def test_read_errors(registry, ctx):
    assert not run(registry, ctx, "read_file", path="missing.txt")["success"]
    assert "outside" in run(registry, ctx, "read_file", path="../x")["error"]
    (ctx.root / "bin.dat").write_bytes(b"\x00\x01\x02")
    assert "binary" in run(registry, ctx, "read_file", path="bin.dat")["error"]


def test_env_file_blocked_and_hidden(registry, ctx):
    (ctx.root / ".env").write_text("OPENROUTER_API_KEY=sk-or-v1-abcdefghijklmnopqrstuv\n")
    (ctx.root / "a.txt").write_text("x")
    assert "blocked" in run(registry, ctx, "read_file", path=".env")["error"]
    listing = run(registry, ctx, "list_files")["data"]["entries"]
    assert "a.txt" in listing and ".env" not in listing


def test_secrets_redacted_when_reading(registry, ctx):
    (ctx.root / "config.php").write_text("<?php\n$db_password = 'hunter2hunter2';\n$user = 'root';\n")
    data = run(registry, ctx, "read_file", path="config.php")["data"]
    assert "hunter2hunter2" not in data["content"] and "[REDACTED]" in data["content"]
    assert "'root'" in data["content"]
    # cannot clobber a file whose secrets were hidden from the model
    r = run(registry, ctx, "write_file", path="config.php", content="x")
    assert not r["success"] and "redacted" in r["error"]


def test_overwrite_requires_confirmation(registry, project):
    denied = make_ctx(project, answer=False)
    (project / "f.txt").write_text("old")
    r = run(registry, denied, "write_file", path="f.txt", content="new")
    assert not r["success"] and "denied" in r["error"].lower()
    assert (project / "f.txt").read_text() == "old"
    allowed = make_ctx(project, answer=True)
    assert run(registry, allowed, "write_file", path="f.txt", content="new")["success"]
    assert (project / "f.txt").read_text() == "new"
    assert allowed.permissions.confirm.__self__.prompts  # user was asked


def test_new_file_needs_no_prompt_in_normal_mode(registry, project):
    ctx = make_ctx(project, answer=False)
    assert run(registry, ctx, "write_file", path="new.txt", content="x")["success"]


def test_edit_requires_prior_read(registry, ctx):
    (ctx.root / "a.py").write_text("x = 1\n")
    r = run(registry, ctx, "edit_file", path="a.py", old_text="x = 1", new_text="x = 2")
    assert not r["success"] and "read_file" in r["error"]


def test_edit_exact_unique_replacement(registry, ctx):
    (ctx.root / "a.py").write_text("a = 1\nb = 2\nb = 2\n")
    run(registry, ctx, "read_file", path="a.py")
    r = run(registry, ctx, "edit_file", path="a.py", old_text="a = 1", new_text="a = 10")
    assert r["success"]
    assert (ctx.root / "a.py").read_text() == "a = 10\nb = 2\nb = 2\n"
    # ambiguous
    r = run(registry, ctx, "edit_file", path="a.py", old_text="b = 2", new_text="b = 3")
    assert not r["success"] and "2 places" in r["error"]
    # replace_all
    assert run(registry, ctx, "edit_file", path="a.py", old_text="b = 2", new_text="b = 3", replace_all=True)["success"]
    # missing text
    r = run(registry, ctx, "edit_file", path="a.py", old_text="zzz", new_text="y")
    assert not r["success"] and "not found" in r["error"]


def test_edit_detects_external_change(registry, ctx):
    p = ctx.root / "a.py"
    p.write_text("one\n")
    run(registry, ctx, "read_file", path="a.py")
    p.write_text("two!\n")
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 5_000_000))
    r = run(registry, ctx, "edit_file", path="a.py", old_text="two!", new_text="x")
    assert not r["success"] and "changed on disk" in r["error"]


def test_edit_preserves_crlf(registry, ctx):
    p = ctx.root / "w.txt"
    p.write_bytes(b"a\r\nb\r\n")
    run(registry, ctx, "read_file", path="w.txt")
    assert run(registry, ctx, "edit_file", path="w.txt", old_text="a\nb", new_text="a\nc")["success"]
    assert p.read_bytes() == b"a\r\nc\r\n"


def test_delete_requires_confirmation(registry, project):
    (project / "d.txt").write_text("x")
    assert not run(registry, make_ctx(project, answer=False), "delete_file", path="d.txt")["success"]
    assert (project / "d.txt").exists()
    # auto approval mode still asks before deleting
    auto = make_ctx(project, approval="auto", answer=True)
    assert run(registry, auto, "delete_file", path="d.txt")["success"]
    assert auto.permissions.confirm.__self__.prompts
    assert not (project / "d.txt").exists()


def test_git_directory_is_protected(registry, ctx):
    (ctx.root / ".git").mkdir()
    r = run(registry, ctx, "write_file", path=".git/config", content="x")
    assert not r["success"] and ".git" in r["error"]


def test_list_files_respects_ignores(registry, ctx):
    for d in ("node_modules", "vendor", "__pycache__", "src", "build_out"):
        (ctx.root / d).mkdir()
        (ctx.root / d / "f.txt").write_text("x")
    (ctx.root / ".gitignore").write_text("build_out/\n*.log\n")
    (ctx.root / "debug.log").write_text("x")
    entries = run(registry, ctx, "list_files")["data"]["entries"]
    assert "src/f.txt" in entries
    assert not any(e.startswith(("node_modules", "vendor", "__pycache__", "build_out")) for e in entries)
    assert "debug.log" not in entries
    everything = run(registry, ctx, "list_files", include_ignored=True)["data"]["entries"]
    assert "node_modules/f.txt" in everything


def test_large_read_is_truncated(registry, ctx):
    ctx.config.max_read_chars = 500
    (ctx.root / "big.txt").write_text("\n".join(f"line {i}" for i in range(1000)))
    data = run(registry, ctx, "read_file", path="big.txt")["data"]
    assert data["end_line"] < 1000 and "Showing lines" in data["note"]
    rng = run(registry, ctx, "read_file", path="big.txt", start_line=10, end_line=12)["data"]
    assert rng["content"] == "line 9\nline 10\nline 11"


def test_no_temp_files_left_behind(registry, ctx):
    run(registry, ctx, "write_file", path="x.txt", content="1")
    assert [p.name for p in ctx.root.iterdir()] == ["x.txt"]


def test_invalid_arguments_rejected(registry, ctx):
    assert "valid JSON" in registry.execute("read_file", "{not json", ctx)["error"]
    assert "Missing required" in registry.execute("read_file", "{}", ctx)["error"]
    assert "must be of type" in registry.execute("read_file", '{"path": 5}', ctx)["error"]
    assert "Unknown tool" in registry.execute("rm_rf", "{}", ctx)["error"]

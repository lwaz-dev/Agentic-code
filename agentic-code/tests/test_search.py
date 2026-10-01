import json

from conftest import make_ctx


def search(registry, ctx, **args):
    return registry.execute("search_files", json.dumps(args), ctx)


def test_finds_matches_with_line_numbers(registry, ctx):
    (ctx.root / "config").mkdir()
    (ctx.root / "config" / "database.php").write_text("<?php\n\n$pdo = new PDO($dsn);\n")
    (ctx.root / "api.php").write_text("a\nb\nuse PDO;\n")
    data = search(registry, ctx, pattern="PDO")["data"]
    assert "api.php:3: use PDO;" in data["matches"]
    assert any(m.startswith("config/database.php:3:") for m in data["matches"])


def test_ignores_vendor_and_node_modules(registry, ctx):
    for d in ("vendor", "node_modules", ".git", "src"):
        (ctx.root / d).mkdir()
        (ctx.root / d / "f.txt").write_text("needle")
    matches = search(registry, ctx, pattern="needle")["data"]["matches"]
    assert matches == ["src/f.txt:1: needle"]
    assert len(search(registry, ctx, pattern="needle", include_ignored=True)["data"]["matches"]) == 4


def test_regex_glob_and_case(registry, ctx):
    (ctx.root / "a.py").write_text("def Foo(): pass\n")
    (ctx.root / "b.txt").write_text("def Foo(): pass\n")
    assert len(search(registry, ctx, pattern=r"def\s+foo", regex=True)["data"]["matches"]) == 2
    assert len(search(registry, ctx, pattern="foo", case_sensitive=True)["data"]["matches"]) == 0
    assert len(search(registry, ctx, pattern="foo", glob="*.py")["data"]["matches"]) == 1
    assert "Invalid regular expression" in search(registry, ctx, pattern="(", regex=True)["error"]


def test_result_limit_and_binary_and_secrets(registry, ctx):
    ctx.config.max_search_results = 5
    (ctx.root / "many.txt").write_text("hit\n" * 50)
    (ctx.root / "bin.dat").write_bytes(b"\x00hit")
    (ctx.root / ".env").write_text("SECRET_TOKEN=hit\n")
    data = search(registry, ctx, pattern="hit")["data"]
    assert data["count"] == 5 and "truncated" in data["note"]
    assert all(m.startswith("many.txt") for m in data["matches"])


def test_search_outside_root_rejected(registry, ctx):
    assert "outside" in search(registry, ctx, pattern="x", path="../")["error"]

"""Project understanding and .agentic/ project files."""
from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Any

from agentic.redact import redact
from agentic.tools.filesystem import walk
from agentic.tools.registry import Tool, ToolContext

LANGS = {
    ".php": "PHP", ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".go": "Go",
    ".rs": "Rust", ".java": "Java", ".cs": "C#", ".rb": "Ruby",
}
ENTRY_POINTS = [
    "index.php", "public/index.php", "main.py", "app.py", "manage.py", "src/main.py", "index.js",
    "server.js", "src/index.js", "src/index.ts", "main.go", "src/main.rs",
]
DB_HINTS = [
    ("MySQL", ("mysqli", "pdo_mysql", "mysql:host", "mysql://", "pymysql", "mysql2")),
    ("PostgreSQL", ("pgsql", "postgres", "psycopg")),
    ("SQLite", ("sqlite",)),
    ("MongoDB", ("mongodb", "mongoose", "pymongo")),
]


def _json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return ""


def detect_project(root: Path) -> dict[str, Any]:
    """Cheap, bounded inspection of language, framework, database, entry point, package manager, tests."""
    info: dict[str, Any] = {
        "language": "Unknown", "framework": "None", "database": "Unknown", "entry_point": None,
        "package_manager": None, "test_command": None, "git": (root / ".git").exists(), "files": 0,
    }
    files = [p for p, d in islice(walk(root, root), 2000) if not d]
    info["files"] = len(files)
    counts: dict[str, int] = {}
    for f in files:
        lang = LANGS.get(Path(f).suffix.lower())
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    if counts:
        info["language"] = max(counts, key=counts.get)

    pkg, composer = _json(root / "package.json"), _json(root / "composer.json")
    py_text = _text(root / "requirements.txt") + _text(root / "pyproject.toml")
    if composer or (root / "composer.json").exists():
        info["package_manager"] = "Composer"
        require = " ".join(composer.get("require", {}))
        info["framework"] = "Laravel" if "laravel/framework" in require else "Symfony" if "symfony/" in require else info["framework"]
        if "test" in composer.get("scripts", {}):
            info["test_command"] = "composer test"
    if (root / "wp-config.php").exists():
        info["framework"] = "WordPress"
    if pkg:
        info["package_manager"] = info["package_manager"] or (
            "pnpm" if (root / "pnpm-lock.yaml").exists() else "yarn" if (root / "yarn.lock").exists() else "npm"
        )
        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        for dep, name in (("next", "Next.js"), ("react", "React"), ("vue", "Vue"), ("@angular/core", "Angular"),
                          ("svelte", "Svelte"), ("express", "Express")):
            if dep in deps:
                info["framework"] = name
                break
        if "test" in pkg.get("scripts", {}):
            info["test_command"] = info["test_command"] or "npm test"
    if py_text or (root / "setup.py").exists():
        info["package_manager"] = info["package_manager"] or "pip"
        for dep, name in (("django", "Django"), ("fastapi", "FastAPI"), ("flask", "Flask")):
            if dep in py_text:
                info["framework"] = name
                break
        if (root / "tests").is_dir() or "pytest" in py_text:
            info["test_command"] = info["test_command"] or "pytest"
    for manifest, manager, test in (("Cargo.toml", "Cargo", "cargo test"), ("go.mod", "Go modules", "go test ./..."),
                                    ("pom.xml", "Maven", "mvn test")):
        if (root / manifest).exists():
            info["package_manager"] = info["package_manager"] or manager
            info["test_command"] = info["test_command"] or test
    if (root / "phpunit.xml").exists() or (root / "phpunit.xml.dist").exists():
        info["test_command"] = info["test_command"] or "vendor/bin/phpunit"

    for candidate in ENTRY_POINTS:
        if (root / candidate).exists():
            info["entry_point"] = candidate
            break

    source_exts = set(LANGS) | {".env", ".ini", ".json", ".yml", ".yaml", ".toml"}
    blob = ""
    for f in [f for f in files if Path(f).suffix.lower() in source_exts][:60]:
        full = root / f
        try:
            if full.stat().st_size < 100_000 and not full.name.lower().startswith(".env"):
                blob += _text(full)
        except OSError:
            continue
    for name, hints in DB_HINTS:
        if any(h in blob for h in hints):
            info["database"] = name
            break
    return info


def format_project_info(info: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Language: {info['language']}",
            f"Framework: {info['framework']}",
            f"Database: {info['database']}",
            f"Entry point: {info['entry_point'] or 'unknown'}",
            f"Package manager: {info['package_manager'] or 'none detected'}",
            f"Test command: {info['test_command'] or 'none detected'}",
            f"Git repository: {'yes' if info['git'] else 'no'}",
            f"Files (non-ignored): {info['files']}",
        ]
    )


def load_project_rules(root: Path) -> str:
    """Contents of .agentic/rules.md and .agentic/project.md, if present."""
    parts = []
    for name in ("rules.md", "project.md"):
        path = root / ".agentic" / name
        try:
            if path.is_file():
                parts.append(f"## .agentic/{name}\n{redact(path.read_text(encoding='utf-8', errors='replace'))[:8000]}")
        except OSError:
            continue
    return "\n\n".join(parts)


RULES_TEMPLATE = """# Project Rules

<!-- Agentic Code follows these rules in this project. Examples: -->
<!-- Use PHP 8.2. Use PDO with prepared statements. Keep API endpoints inside /api. -->
"""


def init_project(root: Path, config: Any) -> list[str]:
    """Create .agentic/project.md, rules.md and config.json if missing. Returns created paths."""
    folder = root / ".agentic"
    folder.mkdir(exist_ok=True)
    info = detect_project(root)
    created: list[str] = []
    files = {
        "project.md": f"# Project\n\n{format_project_info(info).replace(chr(10), chr(10) + chr(10))}\n",
        "rules.md": RULES_TEMPLATE,
        "config.json": json.dumps(
            {"model": config.model, "approval_mode": config.approval_mode,
             "max_iterations": config.max_iterations, "stream": config.stream}, indent=4) + "\n",
    }
    for name, content in files.items():
        path = folder / name
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            created.append(f".agentic/{name}")
    return created


def project_info_tool(ctx: ToolContext):
    info = detect_project(ctx.root)
    info["summary"] = format_project_info(info)
    return info


TOOLS = [
    Tool("project_info",
         "Detect the project's language, framework, database, entry point, package manager and test command.",
         {"type": "object", "properties": {}}, project_info_tool, "read"),
]

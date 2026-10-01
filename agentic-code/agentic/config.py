"""Configuration: defaults < ~/.agentic/.env < project .env < .agentic/config.json < env vars < CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic.redact import register_secret

DEFAULT_MODEL = "openai/gpt-oss-20b:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
APPROVAL_MODES = ("strict", "normal", "auto")
AGENT_MODES = ("chat", "code", "plan", "review", "auto")

PERSISTABLE = {
    "provider", "model", "base_url", "approval_mode", "mode", "max_iterations",
    "stream", "debug", "command_timeout", "shell",
}
ENV_MAP = {
    "AGENTIC_PROVIDER": "provider",
    "AGENTIC_MODEL": "model",
    "AGENTIC_BASE_URL": "base_url",
    "AGENTIC_APPROVAL_MODE": "approval_mode",
    "AGENTIC_MODE": "mode",
    "AGENTIC_MAX_ITERATIONS": "max_iterations",
    "AGENTIC_STREAM": "stream",
    "AGENTIC_DEBUG": "debug",
    "AGENTIC_SHELL": "shell",
    "AGENTIC_COMMAND_TIMEOUT": "command_timeout",
}


class ConfigError(Exception):
    """Raised for invalid configuration (message is user-friendly)."""


class Config(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    provider: str = "openrouter"
    model: str = DEFAULT_MODEL
    api_key: str = Field(default="", repr=False)
    base_url: str = DEFAULT_BASE_URL
    approval_mode: Literal["strict", "normal", "auto"] = "normal"
    mode: Literal["chat", "code", "plan", "review", "auto"] = "code"
    max_iterations: int = Field(default=30, ge=1, le=500)
    stream: bool = True
    debug: bool = False
    shell: str = ""
    command_timeout: int = Field(default=120, ge=1, le=3600)
    max_read_chars: int = 40_000
    max_output_chars: int = 20_000
    max_search_results: int = 100
    max_list_files: int = 500
    max_context_chars: int = 120_000

    def public_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        d["api_key"] = "set" if self.api_key else "missing"
        return d


def _read_env(root: Path) -> dict[str, str]:
    merged: dict[str, str] = {}
    candidates = [Path.home() / ".agentic" / ".env", Path.cwd() / ".env", root / ".env"]
    for path in candidates:  # later files win
        try:
            if path.is_file():
                merged.update({k: v for k, v in dotenv_values(path).items() if v is not None})
        except OSError:
            continue
    merged.update(os.environ)
    return merged


def load_config(root: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    root = Path(root)
    env = _read_env(root)
    data: dict[str, Any] = {}

    cfg_file = root / ".agentic" / "config.json"
    if cfg_file.is_file():
        try:
            raw = json.loads(cfg_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Could not read {cfg_file}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{cfg_file} must contain a JSON object.")
        data.update({k: v for k, v in raw.items() if k in PERSISTABLE})

    for env_name, key in ENV_MAP.items():
        if env.get(env_name):
            data[key] = env[env_name]
    data["api_key"] = env.get("OPENROUTER_API_KEY") or env.get("AGENTIC_API_KEY") or ""

    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value

    try:
        config = Config(**data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ConfigError(f"Invalid configuration: {problems}") from exc
    register_secret(config.api_key)
    return config


def save_project_config(root: str | Path, key: str, value: Any) -> Path:
    """Persist one setting to .agentic/config.json (never the API key)."""
    if key not in PERSISTABLE:
        raise ConfigError(f"'{key}' cannot be saved. Settable: {', '.join(sorted(PERSISTABLE))}")
    path = Path(root) / ".agentic" / "config.json"
    current: dict[str, Any] = {}
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
    current[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=4) + "\n", encoding="utf-8")
    return path

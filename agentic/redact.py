"""Secret detection and redaction. Best-effort; applied to anything shown or sent to the model."""
from __future__ import annotations

import re

REDACTED = "[REDACTED]"

_known_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Remember an exact secret (e.g. the API key) so it is always masked."""
    if value and len(value) >= 8:
        _known_secrets.add(value)


_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.S
)
_TOKENS = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{30,}|xox[baprs]-[A-Za-z0-9\-]{10,}"
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})"
)
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}")
_URL_CRED = re.compile(r"(\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:)([^\s@/]+)(@)", re.I)
_KEY = r"[\w\-]*(?:api[_\-]?key|secret|token|passw(?:or)?d|pwd|private[_\-]?key|access[_\-]?key|credential)[\w\-]*"
_ASSIGN = re.compile(
    r"(?i)(\b" + _KEY + r"""["']?\s*(?:=>|=|:)\s*)("[^"\n]{4,}"|'[^'\n]{4,}'|"""
    r"""(?!\$)(?=[A-Za-z0-9_\-+/=.@!#%^*~]*\d)[A-Za-z0-9_\-+/=.@!#%^*~]{8,}(?=\s|$|[,;)]))""",
    re.M,
)
_ENV_LINE = re.compile(
    r"^(\s*(?:export\s+)?[A-Z][A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|PWD)[A-Z0-9_]*\s*=\s*)(\S.*)$",
    re.M,
)


def _assign_sub(m: re.Match) -> str:
    value = m.group(2)
    if value[0] in "\"'":
        return f"{m.group(1)}{value[0]}{REDACTED}{value[0]}"
    return f"{m.group(1)}{REDACTED}"


def redact(text: str) -> str:
    """Return text with likely secrets masked."""
    if not text:
        return text
    for secret in _known_secrets:
        text = text.replace(secret, REDACTED)
    text = _PRIVATE_KEY.sub(REDACTED, text)
    text = _TOKENS.sub(REDACTED, text)
    text = _BEARER.sub(lambda m: m.group(1) + REDACTED, text)
    text = _URL_CRED.sub(lambda m: m.group(1) + REDACTED + m.group(3), text)
    text = _ENV_LINE.sub(lambda m: m.group(1) + REDACTED, text)
    text = _ASSIGN.sub(_assign_sub, text)
    return text

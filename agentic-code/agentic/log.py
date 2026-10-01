"""Optional debug logging to .agentic/logs/agentic.log (secrets are redacted)."""
from __future__ import annotations

import logging
from pathlib import Path

from agentic.redact import redact


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def get_logger() -> logging.Logger:
    return logging.getLogger("agentic")


def setup_logging(debug: bool, root: str | Path) -> logging.Logger:
    logger = get_logger()
    for handler in list(logger.handlers):
        if getattr(handler, "_agentic", False):
            logger.removeHandler(handler)
            handler.close()
    if not debug:
        logger.addHandler(logging.NullHandler())
        return logger
    try:
        log_dir = Path(root) / ".agentic" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / "agentic.log", encoding="utf-8")
    except OSError:
        return logger
    handler._agentic = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handler.addFilter(_RedactFilter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

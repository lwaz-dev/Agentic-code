"""Conversation history with bounded size (older turns are summarised, not kept verbatim)."""
from __future__ import annotations

import json
from typing import Any

from agentic.providers.base import ProviderResponse

TOOL_RESULT_KEEP = 400  # chars kept from old tool results when shrinking


def _size(msg: dict[str, Any]) -> int:
    total = len(msg.get("content") or "")
    for call in msg.get("tool_calls") or []:
        total += len(call["function"]["arguments"]) + len(call["function"]["name"])
    return total


class Conversation:
    def __init__(self, max_chars: int = 120_000) -> None:
        self.max_chars = max_chars
        self.messages: list[dict[str, Any]] = []
        self.summary = ""
        self.current_task = ""

    # ------------------------------------------------------------- mutation
    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, response: ProviderResponse) -> None:
        self.messages.append(response.to_message())

    def add_tool(self, call_id: str, name: str, content: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": content})

    def reset(self) -> None:
        self.messages.clear()
        self.summary = ""
        self.current_task = ""

    # ---------------------------------------------------------------- stats
    def total_chars(self) -> int:
        return sum(_size(m) for m in self.messages) + len(self.summary)

    def history(self) -> list[tuple[str, str]]:
        rows = []
        for m in self.messages:
            text = (m.get("content") or "").replace("\n", " ")
            if m.get("tool_calls"):
                calls = ", ".join(c["function"]["name"] for c in m["tool_calls"])
                text = f"{text[:60]} [calls: {calls}]".strip()
            rows.append((m["role"], text[:100]))
        return rows

    # ----------------------------------------------------------- compaction
    def compact(self, force: bool = False) -> bool:
        """Shrink context if it exceeds the budget. Returns True if anything changed."""
        limit = 0 if force else self.max_chars
        if self.total_chars() <= limit and not force:
            return False
        changed = self._shrink_old_tool_results()
        if self.total_chars() <= limit and not force:
            return changed
        target = self.total_chars() // 3 if force else self.max_chars // 2
        return self._summarise_head(target) or changed

    def _shrink_old_tool_results(self) -> bool:
        changed = False
        for msg in self.messages[:-8]:
            content = msg.get("content") or ""
            if msg["role"] == "tool" and len(content) > TOOL_RESULT_KEEP:
                msg["content"] = json.dumps(
                    {"truncated": True, "preview": content[:TOOL_RESULT_KEEP],
                     "note": "Older tool output was trimmed to save context; re-run the tool if needed."}
                )
                changed = True
        return changed

    def _summarise_head(self, target_chars: int) -> bool:
        # Find the earliest safe cut (a user/assistant message, never a tool result) whose tail fits.
        sizes = [_size(m) for m in self.messages]
        cut, tail = len(self.messages), 0
        for i in range(len(self.messages) - 1, -1, -1):
            tail += sizes[i]
            if tail > target_chars:
                break
            if self.messages[i]["role"] in ("user", "assistant"):
                cut = i
        if cut >= len(self.messages) or cut == 0:
            return False
        dropped, kept = self.messages[:cut], self.messages[cut:]
        self.summary = self._merge_summary(dropped)
        self.messages = kept
        if kept[0]["role"] == "assistant":
            self.messages.insert(0, {"role": "user", "content": "[Earlier context was summarised. Continue the current task.]"})
        return True

    def _merge_summary(self, dropped: list[dict[str, Any]]) -> str:
        lines = [self.summary] if self.summary else []
        actions: list[str] = []
        for m in dropped:
            if m["role"] == "user" and not (m.get("content") or "").startswith("["):
                lines.append(f"- User asked: {m['content'][:200]}")
            elif m["role"] == "assistant":
                for c in m.get("tool_calls") or []:
                    try:
                        args = json.loads(c["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    target = args.get("path") or args.get("command") or args.get("pattern") or ""
                    actions.append(f"{c['function']['name']}({str(target)[:60]})")
                if m.get("content"):
                    lines.append(f"- Assistant said: {m['content'][:150]}")
        if actions:
            seen = list(dict.fromkeys(actions))[-40:]
            lines.append("- Actions taken: " + ", ".join(seen))
        return "\n".join(lines)[-4000:]

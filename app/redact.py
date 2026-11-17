"""C1 Redactor (owner B): mask sensitive data before indexing, LLM calls, storage and export.

Only the redaction *types* are kept, never the original values.
"""
from __future__ import annotations

import re

# Order matters: more specific patterns first.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("url_token", re.compile(r"(?i)https?://\S*?[?&](token|key|sig|code|auth|session)=\S+")),
    ("secret", re.compile(r"(?i)\b(password|passwd|pwd|pin|secret|api[_ -]?key|token)\s*(is|=|:)\s*\S+")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("ip", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("ipv6", re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("phone", re.compile(r"(?<![\w-])\+?\d{1,3}[ -]?\(?\d{2,5}\)?[ -]?\d{3,5}[ -]?\d{3,5}\b")),
    ("employee_id", re.compile(r"\b(?:EMP|E)-?\d{4,8}\b", re.I)),
    ("asset_id", re.compile(r"\b(?:AST|ASSET)-?\d{4,8}\b", re.I)),
]


def redact(text: str | None) -> tuple[str, list[str]]:
    """Return (clean_text, sorted list of redaction types found)."""
    if not text:
        return text or "", []
    found: set[str] = set()
    out = text
    for kind, pat in PATTERNS:
        def _sub(m: re.Match, kind: str = kind) -> str:
            found.add(kind)
            if kind == "secret":            # keep the label, hide the value
                return f"{m.group(1)}: [SECRET]"
            return f"[{kind.upper()}]"
        out = pat.sub(_sub, out)
    return out, sorted(found)

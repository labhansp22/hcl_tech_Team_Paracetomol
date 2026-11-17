"""C0 Dataset adapter (owner A): any CSV / XLSX / JSON -> list[Ticket].

Columns are matched case-insensitively against candidate names, so most service-desk exports load
without code changes. Edit COLUMN_MAP on the day if the supplied dataset uses other names.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from app import config
from app.schema import KBArticle, Ticket
from app.taxonomy import rule_priority

COLUMN_MAP: dict[str, list[str]] = {
    "id": ["id", "ticket_id", "number", "incident_id", "ticket"],
    "title": ["title", "subject", "short_description", "summary"],
    "text": ["text", "description", "body", "details", "long_description", "ticket_text"],
    "category": ["category", "queue", "type"],
    "subcategory": ["subcategory", "sub_category", "subcat", "topic"],
    "priority": ["priority", "severity", "urgency"],
    "status": ["status", "state"],
    "application": ["application", "app", "service", "configuration_item", "ci"],
    "environment": ["environment", "env"],
    "assignment_group": ["assignment_group", "group", "team", "resolver_group"],
    "created_at": ["created_at", "opened_at", "created", "date", "open_date"],
    "resolution": ["resolution", "resolution_notes", "close_notes", "solution", "answer"],
}

_PRIO = {"1": "P1", "2": "P2", "3": "P3", "4": "P4", "critical": "P1", "high": "P2", "medium": "P3",
         "moderate": "P3", "low": "P4", "p1": "P1", "p2": "P2", "p3": "P3", "p4": "P4"}
_STATUS = {"open": "open", "new": "open", "in progress": "open", "resolved": "resolved", "closed": "closed"}
_ENV = {"prod": "prod", "production": "prod", "uat": "uat", "test": "uat", "dev": "dev", "development": "dev"}


def _read(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suf == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if suf == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def _clean(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


def from_dataframe(df: pd.DataFrame) -> list[Ticket]:
    lower = {c.lower().strip(): c for c in df.columns}
    cols = {field: next((lower[c] for c in cands if c in lower), None) for field, cands in COLUMN_MAP.items()}
    if not cols["text"]:
        raise ValueError(f"No text/description column found in {list(df.columns)}")
    out: list[Ticket] = []
    for i, row in df.iterrows():
        g = lambda f: _clean(row[cols[f]]) if cols[f] else None  # noqa: E731
        synthetic: list[str] = []
        text = g("text") or ""
        title = g("title")
        env = _ENV.get((g("environment") or "").lower())
        prio = _PRIO.get((g("priority") or "").lower())
        if not prio:
            prio = rule_priority(f"{title or ''} {text}", env)
            synthetic.append("priority")
        created = pd.to_datetime(g("created_at"), errors="coerce") if g("created_at") else None
        out.append(Ticket(
            id=g("id") or f"TKT-{i + 1:05d}",
            title=title,
            text=text,
            category=g("category"),
            subcategory=g("subcategory") or g("category"),
            priority=prio,
            status=_STATUS.get((g("status") or "").lower()),
            application=g("application"),
            environment=env,
            assignment_group=g("assignment_group"),
            created_at=None if created is None or pd.isna(created) else created.to_pydatetime(),
            resolution=g("resolution"),
            synthetic_fields=synthetic,
        ))
    return out


def load_tickets(path: str | Path | None = None) -> list[Ticket]:
    """Supplied dataset if DATASET_PATH / path is set, else the synthetic dataset (generated on demand)."""
    path = path or config.DATASET_PATH
    if path:
        return from_dataframe(_read(Path(path)))
    if not config.TICKETS_JSON.exists():
        subprocess.run([sys.executable, str(config.ROOT / "data" / "generate.py")], check=True)
    raw = json.loads(config.TICKETS_JSON.read_text(encoding="utf-8"))
    return [Ticket(**t) for t in raw]


def load_kb(path: Path | None = None) -> list[KBArticle]:
    path = path or config.KB_JSON
    return [KBArticle(**a) for a in json.loads(Path(path).read_text(encoding="utf-8"))]

"""C9 Audit store (owner C): SQLite tables analyses, feedback, kb_gaps (Section 5.4). Redacted text only."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from datetime import datetime, timezone

from app import config
from app.schema import AnalysisResult, FeedbackIn, TicketIn

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
  analysis_id TEXT PRIMARY KEY, created_at TEXT, input_redacted TEXT, classification TEXT, retrieved TEXT,
  recommendation TEXT, status TEXT, confidence REAL, degraded_mode INTEGER, provider TEXT, model TEXT,
  latency_ms INTEGER);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT REFERENCES analyses(analysis_id), item_type TEXT,
  item_id TEXT, verdict TEXT, corrected_label TEXT, note TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS kb_gaps (
  id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT, topic TEXT, created_at TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AuditDB:
    def __init__(self, path=config.DB_PATH):
        self.path = str(path)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def save_analysis(self, t: TicketIn, r: AnalysisResult) -> None:
        from app.redact import redact
        inp = {"title": redact(t.title)[0] if t.title else None, "description": redact(t.description)[0],
               "application": t.application, "environment": t.environment,
               "filters": t.filters.model_dump(mode="json")}
        retrieved = {"tickets": [(h.source_id, h.score) for h in r.similar_tickets],
                     "kb": [(h.source_id, h.score) for h in r.kb_matches]}
        rec = {"paths": [p.model_dump() for p in r.resolution_paths],
               "escalation": r.escalation.model_dump() if r.escalation else None,
               "reasons": r.confidence_reasons, "dropped_claims": r.dropped_claims, "redactions": r.redactions}
        with self.lock:
            self.conn.execute(
                "INSERT INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (r.analysis_id, _now(), json.dumps(inp), r.classification.model_dump_json(), json.dumps(retrieved),
                 json.dumps(rec), r.status, r.confidence, int(r.degraded_mode), r.provider, r.model,
                 sum(r.timings_ms.values())))
            if r.kb_gap:
                self.conn.execute("INSERT INTO kb_gaps (analysis_id, topic, created_at) VALUES (?,?,?)",
                                  (r.analysis_id, f"{r.classification.category} / {r.classification.subcategory}: "
                                   f"{inp['description'][:120]}", _now()))
            self.conn.commit()

    def exists(self, analysis_id: str) -> bool:
        with self.lock:
            return self.conn.execute("SELECT 1 FROM analyses WHERE analysis_id=?", (analysis_id,)).fetchone() is not None

    def save_feedback(self, f: FeedbackIn) -> int:
        from app.redact import redact
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO feedback (analysis_id,item_type,item_id,verdict,corrected_label,note,created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (f.analysis_id, f.item_type, f.item_id, f.verdict, f.corrected_label,
                 redact(f.note)[0] if f.note else None, _now()))
            self.conn.commit()
            return int(cur.lastrowid)

    def recent(self, limit: int = 20) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT analysis_id, created_at, input_redacted, classification, status, confidence, degraded_mode, "
                "provider, model, latency_ms FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["input_redacted"] = json.loads(d["input_redacted"])
            d["classification"] = json.loads(d["classification"])
            out.append(d)
        return out

    def stats(self) -> dict:
        with self.lock:
            q = lambda s: [dict(r) for r in self.conn.execute(s).fetchall()]  # noqa: E731
            return {
                "by_status": q("SELECT status, COUNT(*) n FROM analyses GROUP BY status"),
                "feedback": q("SELECT item_type, verdict, COUNT(*) n FROM feedback GROUP BY item_type, verdict"),
                "kb_gaps": q("SELECT analysis_id, topic, created_at FROM kb_gaps ORDER BY id DESC LIMIT 20"),
                "total": self.conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0],
                "avg_latency_ms": self.conn.execute("SELECT AVG(latency_ms) FROM analyses").fetchone()[0],
            }

    def export_rows(self) -> list[dict]:
        sql = """SELECT a.analysis_id, a.created_at, a.input_redacted, a.classification, a.status, a.confidence,
                        a.degraded_mode, a.provider, a.model, a.latency_ms, a.recommendation,
                        f.item_type AS fb_item_type, f.item_id AS fb_item_id, f.verdict AS fb_verdict,
                        f.corrected_label AS fb_corrected_label, f.note AS fb_note, f.created_at AS fb_created_at
                 FROM analyses a LEFT JOIN feedback f ON f.analysis_id = a.analysis_id
                 ORDER BY a.created_at"""
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql).fetchall()]

    def export_csv(self) -> str:
        rows = self.export_rows()
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return buf.getvalue()

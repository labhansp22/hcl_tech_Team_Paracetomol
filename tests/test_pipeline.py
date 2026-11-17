"""Integration tests on the built index (run `python -m app.build_index` first).

The LLM is replaced by a fake provider, so these tests never need an API key or network.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import config, llm_provider

pytestmark = pytest.mark.skipif(not config.INDEX_META.exists(), reason="index not built")


@pytest.fixture(scope="module")
def pipe():
    from app.pipeline import Pipeline
    return Pipeline()


def fake_llm(payload: dict):
    def _chat(messages, overrides=None, max_tokens=0):
        return json.dumps(payload), "fake", "fake-model"
    return _chat


def test_degraded_vpn(pipe):
    from app.schema import TicketIn
    r = pipe.analyze(TicketIn(description="GlobalProtect VPN stuck on connecting, error 809"), use_llm=False)
    assert r.degraded_mode and r.classification.subcategory == "VPN"
    assert r.status == "NEEDS_REVIEW" and r.resolution_paths
    assert r.resolution_paths[0].evidence[0].source_id.startswith("KB-")


def test_out_of_scope_escalates(pipe):
    from app.schema import TicketIn
    r = pipe.analyze(TicketIn(description="The coffee machine on floor 3 is leaking water"), use_llm=False)
    assert r.status == "ESCALATE" and r.kb_gap and r.escalation


def test_redaction_reaches_result(pipe):
    from app.schema import TicketIn
    r = pipe.analyze(TicketIn(description="I forgot my password, mail me at a.b@corp.example.com"), use_llm=False)
    assert "email" in r.redactions


def test_filters_apply(pipe):
    from app.schema import SearchFilters
    hits = pipe.retriever.search_tickets("printer offline", SearchFilters(status="open", environment="prod"), k=5)
    assert hits and all(h.metadata["status"] == "open" and h.metadata["environment"] == "prod" for h in hits)


def test_llm_path_with_validation(pipe, monkeypatch):
    """A fake LLM cites one real KB quote, one invented KB id and tries to lower the priority."""
    from app.schema import TicketIn
    monkeypatch.setattr(llm_provider, "available", lambda: ["fake"])
    monkeypatch.setattr("app.pipeline.available", lambda: ["fake"])
    monkeypatch.setattr("app.llm.chat_json", fake_llm({
        "category": "Network", "subcategory": "VPN", "priority": "P4", "confidence": 0.9,
        "rationale": "VPN error 809", "duplicate_ids": [],
        "resolution_paths": [
            {"title": "Restart VPN", "steps": ["Restart PanGPS"],
             "evidence": [{"source_id": "KB-0005", "excerpt": "Restart the PanGPS service or reboot the laptop", "reason": "kb"}]},
            {"title": "Invented", "steps": ["x"], "evidence": [{"source_id": "KB-9999", "excerpt": "made up", "reason": "?"}]}],
        "insufficient_evidence": False, "clarifying_questions": []}))
    r = pipe.analyze(TicketIn(description="VPN down for all users in the office, error 809"), use_cache=False)
    assert not r.degraded_mode and r.provider == "fake"
    assert [p.title for p in r.resolution_paths] == ["Restart VPN"]
    assert r.dropped_claims == 1
    assert r.classification.priority == "P1"          # LLM can't lower below the rule floor
    assert r.status == "NEEDS_REVIEW"                  # P1 + dropped claim


def test_api_roundtrip():
    from app.api import app
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/analyze", params={"use_llm": False}, json={"description": "Printer on floor 3 shows offline"}).json()
        assert r["classification"]["subcategory"] == "Printer"
        assert c.post("/feedback", json={"analysis_id": r["analysis_id"], "item_type": "classification",
                                         "item_id": "Printer", "verdict": "useful"}).status_code == 200
        assert c.post("/feedback", json={"analysis_id": "nope", "item_type": "classification",
                                         "item_id": "x", "verdict": "useful"}).status_code == 404
        s = c.get("/search", params={"q": "error 0x80070005", "mode": "bm25"}).json()
        assert s["tickets"]
        assert "analysis_id" in c.get("/audit/export", params={"format": "csv"}).text

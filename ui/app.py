"""C10 Streamlit UI (owner C). Talks to the FastAPI service; falls back to the in-process pipeline if the API is down.

Run:  streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.demo import DEMO_TICKETS  # noqa: E402

st.set_page_config(page_title="IT Ticket Assistant", page_icon="🎫", layout="wide")

STATUS_STYLE = {"OK": ("✅", "green"), "NEEDS_REVIEW": ("🟡", "orange"), "ESCALATE": ("🔺", "red")}


# ---------------------------------------------------------------- backend
class Backend:
    """HTTP client for the API, or the pipeline in-process when the API is not running."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.mode = "api"
        try:
            httpx.get(f"{self.url}/health", timeout=2).raise_for_status()
        except Exception:
            self.mode = "in-process"

    def _local(self):
        return _local_services()

    def health(self) -> dict:
        if self.mode == "api":
            return httpx.get(f"{self.url}/health", timeout=10).json()
        from app import llm_provider
        p, _ = self._local()
        return {"status": "ok", "index": p.retriever.stats(), "llm_chain": llm_provider.available(),
                "llm_pool": llm_provider.pool_status(), "baseline": p.baseline is not None}

    def analyze(self, payload: dict, use_llm: bool) -> dict:
        if self.mode == "api":
            r = httpx.post(f"{self.url}/analyze", params={"use_llm": use_llm}, json=payload, timeout=90)
            r.raise_for_status()
            return r.json()
        from app.schema import TicketIn
        p, db = self._local()
        t = TicketIn(**payload)
        res = p.analyze(t, use_llm=use_llm)
        db.save_analysis(t, res)
        return res.model_dump(mode="json")

    def search(self, params: dict) -> dict:
        if self.mode == "api":
            r = httpx.get(f"{self.url}/search", params={k: v for k, v in params.items() if v}, timeout=30)
            r.raise_for_status()
            return r.json()
        from app.schema import SearchFilters
        p, _ = self._local()
        f = SearchFilters(**{k: v for k, v in params.items() if k in SearchFilters.model_fields and v})
        return {"tickets": [h.model_dump() for h in p.retriever.search_tickets(params["q"], f, k=params["k"], mode=params["mode"])],
                "kb": [h.model_dump() for h in p.retriever.search_kb(params["q"], k=5, mode=params["mode"])]}

    def feedback(self, payload: dict) -> None:
        if self.mode == "api":
            httpx.post(f"{self.url}/feedback", json=payload, timeout=10).raise_for_status()
            return
        from app.schema import FeedbackIn
        self._local()[1].save_feedback(FeedbackIn(**payload))

    def stats(self) -> dict:
        return httpx.get(f"{self.url}/stats", timeout=10).json() if self.mode == "api" else self._local()[1].stats()

    def recent(self) -> list:
        return httpx.get(f"{self.url}/analyses", timeout=10).json() if self.mode == "api" else self._local()[1].recent(50)

    def export(self, fmt: str) -> bytes:
        if self.mode == "api":
            return httpx.get(f"{self.url}/audit/export", params={"format": fmt}, timeout=30).content
        db = self._local()[1]
        import json
        return db.export_csv().encode() if fmt == "csv" else json.dumps(db.export_rows(), indent=1).encode()

    def taxonomy(self) -> dict:
        return httpx.get(f"{self.url}/taxonomy", timeout=10).json() if self.mode == "api" else self._local()[0].tax


@st.cache_resource(show_spinner="Loading models and index…")
def _local_services():
    from app.db import AuditDB
    from app.pipeline import Pipeline
    return Pipeline(), AuditDB()


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🎫 IT Ticket Assistant")
api_url = st.sidebar.text_input("API URL", config.API_URL)
pool: dict = {}
be = Backend(api_url)          # re-checked every run: switches to the API as soon as it is up
try:
    h = be.health()
    st.sidebar.caption(f"Mode: **{be.mode}** · index: {h['index']['tickets']} tickets, "
                       f"{h['index']['kb_articles']} KB · {h['index']['backend']}")
    pool = h.get("llm_pool") or {}
    st.sidebar.caption("LLM chain: " + (" → ".join(
        f"{p} ({v['keys']} key{'s' if v['keys'] != 1 else ''}, round robin)" if v["keys"] > 1 else p
        for p, v in pool.items()) or ", ".join(h["llm_chain"]) or "⚠ no API key → degraded mode"))
except Exception as e:  # pragma: no cover
    st.sidebar.error(f"Backend error: {e}")
    st.stop()

use_llm = st.sidebar.toggle("Use LLM", value=True, help="Off = baseline + KB evidence only (degraded mode)")
tax = be.taxonomy()
st.sidebar.subheader("Similar-ticket filters")
f_cat = st.sidebar.selectbox("Category", [""] + list(tax))
f_status = st.sidebar.selectbox("Status", ["", "resolved", "closed", "open"])
f_env = st.sidebar.selectbox("Environment (history)", ["", "prod", "uat", "dev"])
f_app = st.sidebar.text_input("Application (exact)")
f_from = st.sidebar.date_input("Created from", value=None)
f_to = st.sidebar.date_input("Created to", value=None)
filters = {"category": f_cat or None, "status": f_status or None, "environment": f_env or None,
           "application": f_app or None,
           "date_from": f_from.isoformat() if isinstance(f_from, date) else None,
           "date_to": f_to.isoformat() if isinstance(f_to, date) else None}


def send_fb(item_type: str, item_id: str, verdict: str, corrected: str | None = None) -> None:
    try:
        be.feedback({"analysis_id": st.session_state["result"]["analysis_id"], "item_type": item_type,
                     "item_id": item_id, "verdict": verdict, "corrected_label": corrected})
        st.toast(f"Feedback recorded: {verdict}")
    except Exception as e:
        st.toast(f"Feedback failed: {e}")


def fb_buttons(item_type: str, item_id: str, key: str) -> None:
    c1, c2, c3 = st.columns(3)
    if c1.button("👍 Useful", key=f"u{key}"):
        send_fb(item_type, item_id, "useful")
    if c2.button("👎 Incorrect", key=f"i{key}"):
        send_fb(item_type, item_id, "incorrect")
    if c3.button("❓ Unresolved", key=f"n{key}"):
        send_fb(item_type, item_id, "unresolved")


def answered_by(provider: str | None, model: str | None, pool: dict) -> str:
    """'groq#2 (cache)' -> 'Groq · key slot 2 of 4 · model · cached answer'. Keys are never shown, only the slot."""
    if not provider:
        return "none"
    cached = provider.endswith(" (cache)")
    name = provider.replace(" (cache)", "")
    prov, _, slot = name.partition("#")
    n_keys = (pool.get(prov) or {}).get("keys")
    parts = [prov.capitalize()]
    parts.append(f"key slot {slot} of {n_keys}" if slot and n_keys else f"key slot {slot}" if slot
                 else "key slot 1 of 1" if n_keys == 1 else "key slot not recorded")
    parts.append(model or "?")
    if cached:
        parts.append("cached answer (no new LLM call)")
    return " · ".join(parts)


def hits_df(hits: list[dict]) -> pd.DataFrame:
    rows = []
    for x in hits:
        m, b = x.get("metadata", {}), x["score_breakdown"]
        rows.append({"id": x["source_id"], "score": x["score"], "semantic": b.get("semantic"), "bm25": b.get("bm25"),
                     "meta": b.get("meta"), "dup": "✔" if x.get("is_duplicate") else "",
                     "note": "keyword match only" if b.get("keyword_only") else "",
                     "subcategory": m.get("subcategory"), "status": m.get("status"),
                     "group": m.get("assignment_group"), "created": (m.get("created_at") or "")[:10],
                     "text": x["snippet"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tabs
tab_an, tab_search, tab_audit = st.tabs(["Analyze ticket", "Search history", "Audit & feedback"])

with tab_an:
    labels = ["(write your own)"] + [d["label"] for d in DEMO_TICKETS]
    pick = st.selectbox("Load a demo ticket", labels)
    demo = next((d for d in DEMO_TICKETS if d["label"] == pick), {})
    with st.form("ticket"):
        c1, c2, c3 = st.columns([3, 2, 1])
        title = c1.text_input("Title", demo.get("title") or "", key=f"t{pick}")
        app_in = c2.text_input("Application", demo.get("application") or "", key=f"a{pick}")
        envs = ["", "prod", "uat", "dev"]
        env_in = c3.selectbox("Environment", envs, index=envs.index(demo.get("environment") or ""), key=f"e{pick}")
        desc = st.text_area("Description", demo.get("description", ""), height=110, key=f"d{pick}")
        go = st.form_submit_button("Analyze", type="primary", use_container_width=True)
    if go:
        if len(desc.strip()) < 3:
            st.warning("Please describe the issue.")
        else:
            payload = {"title": title or None, "description": desc, "application": app_in or None,
                       "environment": env_in or None, "filters": filters}
            with st.spinner("Retrieving evidence and reasoning…"):
                try:
                    st.session_state["result"] = be.analyze(payload, use_llm)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    r = st.session_state.get("result")
    if r:
        icon, colour = STATUS_STYLE[r["status"]]
        c = r["classification"]
        st.divider()
        m1, m2, m3, m4 = st.columns([1.3, 1, 2, 1])
        m1.markdown(f"### {icon} :{colour}[{r['status'].replace('_', ' ')}]")
        m2.metric("Confidence", f"{r['confidence']:.2f}")
        m3.caption(f"Classification ({c['source']}, {c['confidence']:.2f})")
        m3.markdown(f"**{c['subcategory']}**  \n{c['category']}")
        m4.metric("Priority", c["priority"])
        if r["degraded_mode"]:
            st.warning("⚠ LLM unavailable or off: showing baseline classification and retrieved KB evidence only.")
        else:
            st.caption(f"Answered by: **{answered_by(r['provider'], r['model'], pool)}** · "
                       f"analysis `{r['analysis_id']}`")
        if c.get("secondary"):
            st.caption(f"Secondary domain: {c['secondary'][0]} / {c['secondary'][1]} ({c['secondary'][2]:.2f})")
        st.caption(f"Rationale: {c['rationale']}")
        if r["redactions"]:
            st.info("🔒 Redacted before search and LLM: " + ", ".join(r["redactions"]))

        with st.expander("Why this status? (confidence breakdown)", expanded=r["status"] != "OK"):
            for reason in r["confidence_reasons"]:
                st.write("• " + reason)
            s = r.get("signals", {})
            if s:
                st.bar_chart(pd.DataFrame({"signal": list(s), "value": list(s.values())}).set_index("signal"))
            if r["dropped_claims"]:
                st.write(f"🧹 Validator removed **{r['dropped_claims']}** unsupported claim(s).")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Recommended resolution")
            if r["escalation"]:
                e = r["escalation"]
                st.error(f"Escalate to **{e['assignment_group']}**")
                st.write("**Ask the user:**")
                for q in e["clarifying_questions"]:
                    st.write("• " + q)
                with st.expander("Diagnostic bundle"):
                    st.json(e["diagnostic_bundle"])
            if not r["resolution_paths"]:
                st.caption("No evidence-backed resolution path.")
            for p in r["resolution_paths"]:
                with st.container(border=True):
                    st.markdown(f"**{p['rank']}. {p['title']}**")
                    for i, step in enumerate(p["steps"], 1):
                        st.write(f"{i}. {step}")
                    for ev in p["evidence"]:
                        st.markdown(f"> “{ev['excerpt']}”  \n> — `{ev['source_id']}` ({ev['source_type']}): {ev['reason']}")
                    fb_buttons("resolution_path", f"path-{p['rank']}", f"p{p['rank']}")
            with st.container(border=True):
                st.markdown("**Was the classification right?**")
                subs = [s for v in tax.values() for s in v]
                corr = st.selectbox("Correct subcategory (if wrong)", [""] + subs, key="corr")
                b1, b2 = st.columns(2)
                if b1.button("👍 Correct", key="cls_ok"):
                    send_fb("classification", c["subcategory"], "useful")
                if b2.button("👎 Wrong, save correction", key="cls_bad"):
                    send_fb("classification", c["subcategory"], "incorrect", corr or None)

        with right:
            st.subheader("Knowledge base matches")
            for k in r["kb_matches"]:
                b = k["score_breakdown"]
                st.markdown(f"`{k['source_id']}` **{k.get('title') or ''}** · {k['score']:.2f}  \n"
                            f"<small>semantic {b['semantic']:.2f} · bm25 {b['bm25']:.2f} · meta {b['meta']:.2f}</small>",
                            unsafe_allow_html=True)

        st.subheader("Similar historical tickets")
        if r["similar_tickets"]:
            st.dataframe(hits_df(r["similar_tickets"]), use_container_width=True, hide_index=True,
                         column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=1)})
            pick_t = st.selectbox("Give feedback on a similar ticket", [x["source_id"] for x in r["similar_tickets"]])
            fb_buttons("similar_ticket", pick_t, f"t{pick_t}")
        else:
            st.caption("No matching history.")
        st.caption("Timings (ms): " + ", ".join(f"{k} {v}" for k, v in r["timings_ms"].items()))

with tab_search:
    st.subheader("Search past tickets and KB")
    q = st.text_input("Query", "error 0x80070005")
    c1, c2 = st.columns([2, 1])
    mode = c1.radio("Retrieval mode", ["hybrid", "semantic", "bm25"], horizontal=True,
                    help="Compare keyword (BM25) vs meaning (semantic) vs both")
    k = c2.slider("Results", 3, 20, 8)
    if q:
        res = be.search({"q": q, "k": k, "mode": mode, **filters})
        st.dataframe(hits_df(res["tickets"]), use_container_width=True, hide_index=True,
                     column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=1)})
        st.markdown("**KB:** " + " · ".join(f"`{x['source_id']}` {x.get('title')} ({x['score']:.2f})" for x in res["kb"]))

with tab_audit:
    st.subheader("Audit log")
    s = be.stats()
    a1, a2, a3 = st.columns(3)
    a1.metric("Analyses", s["total"])
    a2.metric("Avg latency (ms)", int(s["avg_latency_ms"] or 0))
    a3.metric("Knowledge gaps", len(s["kb_gaps"]))
    c1, c2 = st.columns(2)
    if s["by_status"]:
        c1.bar_chart(pd.DataFrame(s["by_status"]).set_index("status"))
    if s["feedback"]:
        c2.dataframe(pd.DataFrame(s["feedback"]), hide_index=True, use_container_width=True)
    rec = be.recent()
    if rec:
        st.dataframe(pd.DataFrame([{
            "id": x["analysis_id"], "when": x["created_at"], "status": x["status"], "confidence": x["confidence"],
            "class": f"{x['classification']['category']} / {x['classification']['subcategory']}",
            "priority": x["classification"]["priority"], "llm": x["model"] or "degraded",
            "ms": x["latency_ms"], "ticket": x["input_redacted"]["description"][:80]} for x in rec]),
            use_container_width=True, hide_index=True)
    if s["kb_gaps"]:
        st.markdown("**Knowledge gaps (write KB articles for these):**")
        st.dataframe(pd.DataFrame(s["kb_gaps"]), hide_index=True, use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Export CSV", be.export("csv"), "audit_export.csv", "text/csv", use_container_width=True)
    d2.download_button("⬇ Export JSON", be.export("json"), "audit_export.json", "application/json",
                       use_container_width=True)

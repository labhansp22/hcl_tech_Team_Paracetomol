"""Generate docs/WORKED_EXAMPLES.md from real runs of the demo tickets (submission requirement:
"at least three worked examples showing the input ticket, classification, similar cases,
recommended resolution and source evidence").

Uses the running API (http://127.0.0.1:8000) if it is up, otherwise runs the pipeline in-process.
    python scripts/worked_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.demo import DEMO_TICKETS  # noqa: E402

API = "http://127.0.0.1:8000"

WHY = {
    "1": "A clear, common ticket: shows the normal path end to end with quoted evidence.",
    "2": "Duplicate detection: a past ticket is flagged as a duplicate only when meaning is very close and the category matches.",
    "3": "Not an IT problem: weak evidence must lead to escalation, not an invented answer.",
    "4": "Prompt injection inside the ticket: the model must ignore it, and the priority rules keep P1.",
    "5": "Italian text with personal data: multilingual retrieval, and redaction before search and the LLM.",
    "6": "Two problems in one ticket: the answer gives a separate fix for each, each backed by its own source.",
}


def analyze_all() -> list[tuple[dict, dict]]:
    payloads = [{k: d[k] for k in ("title", "description", "application", "environment")} for d in DEMO_TICKETS]
    try:
        import httpx

        httpx.get(f"{API}/health", timeout=3).raise_for_status()
        with httpx.Client(timeout=180) as c:
            return [(d, c.post(f"{API}/analyze", params={"use_llm": "true"}, json=p).json())
                    for d, p in zip(DEMO_TICKETS, payloads)]
    except Exception:
        from app.pipeline import Pipeline
        from app.schema import TicketIn

        pipe = Pipeline()
        return [(d, pipe.analyze(TicketIn(**p), use_llm=True).model_dump(mode="json"))
                for d, p in zip(DEMO_TICKETS, payloads)]


def esc(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render(d: dict, r: dict) -> list[str]:
    num = d["label"].split(" ")[0]
    c = r["classification"]
    out = [f"## Example {d['label']}", "", f"*Why this example:* {WHY.get(num, '')}", "", "**Input ticket**", ""]
    out += [f"- **Title:** {d.get('title') or '—'}",
            f"- **Description:** {d['description']}",
            f"- **Application / environment:** {d.get('application') or '—'} / {d.get('environment') or '—'}", ""]
    if r.get("redactions"):
        out += [f"Redacted before search and the LLM: {', '.join(r['redactions'])}", ""]
    out += ["**Result**", "",
            "| Status | Confidence | Category / subcategory | Priority | Labelled by | Answered by |",
            "|---|---|---|---|---|---|",
            f"| **{r['status']}** | {r['confidence']:.2f} | {c['category']} / {c['subcategory']} | {c['priority']} | "
            f"{c['source']} ({c['confidence']:.2f}) | {r.get('provider') or 'none (degraded mode)'} "
            f"{('· ' + r['model']) if r.get('model') else ''} |", ""]
    out += [f"Rationale: {c['rationale']}", ""]
    out += ["Why this status:", ""] + [f"- {x}" for x in r.get("confidence_reasons", [])] + [""]

    out += ["**Similar historical tickets (top 3)**", "",
            "| Ticket | Score | Semantic | BM25 | Metadata | Duplicate | Subcategory | Snippet |",
            "|---|---|---|---|---|---|---|---|"]
    for h in r["similar_tickets"][:3]:
        b = h["score_breakdown"]
        out.append(f"| {h['source_id']} | {h['score']:.2f} | {b.get('semantic', 0):.2f} | {b.get('bm25', 0):.2f} | "
                   f"{b.get('meta', 0):.2f} | {'✔' if h['is_duplicate'] else ''} | "
                   f"{esc(h['metadata'].get('subcategory', ''))} | {esc(h['snippet'])[:90]} |")
    kb = r["kb_matches"][:1]
    if kb:
        out += ["", f"Best knowledge-base match: **{kb[0]['source_id']}**, {kb[0].get('title') or ''} "
                    f"(score {kb[0]['score']:.2f})"]
    out.append("")

    if r["resolution_paths"]:
        out += ["**Recommended resolution**", ""]
        for p in r["resolution_paths"]:
            out += [f"{p['rank']}. **{p['title']}**"]
            out += [f"   - {s}" for s in p["steps"]]
            for e in p["evidence"]:
                out += [f"   - *Evidence:* {e['source_id']} ({e['source_type']}): \"{esc(e['excerpt'])}\". {esc(e['reason'])}"]
        out.append("")
    if r.get("dropped_claims"):
        out += [f"The validator removed **{r['dropped_claims']}** unsupported claim(s) (a quote that was not "
                "word for word in its cited source, or a source that was not retrieved).", ""]
    if r.get("escalation"):
        e = r["escalation"]
        out += ["**Escalation**", "", f"- Route to: {e['assignment_group']}"]
        out += [f"- Ask the user: {q}" for q in e["clarifying_questions"]]
        out.append("")
    if r.get("kb_gap"):
        out += ["Logged as a knowledge gap for the KB team.", ""]
    return out + ["---", ""]


def main() -> None:
    results = analyze_all()
    lines = ["# Worked examples", "",
             "Generated by `python scripts/worked_examples.py` from real runs of the six demo tickets "
             "(`app/demo.py`). Each shows the input ticket, its classification, similar historical cases, the "
             "recommended resolution and the source evidence behind it. Scores: `0.6·semantic + 0.3·BM25 + "
             "0.1·metadata`. Confidence: `0.35·LLM + 0.40·retrieval + 0.25·agreement`.", "",
             "| # | Ticket | Status | Category / subcategory | Priority |", "|---|---|---|---|---|"]
    for d, r in results:
        c = r["classification"]
        lines.append(f"| {d['label'].split(' ')[0]} | {esc(d['label'].split('· ', 1)[-1])} | {r['status']} | "
                     f"{c['category']} / {c['subcategory']} | {c['priority']} |")
    lines.append("")
    for d, r in results:
        lines += render(d, r)
    path = ROOT / "docs" / "WORKED_EXAMPLES.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

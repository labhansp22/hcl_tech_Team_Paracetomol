"""C11 Evaluation harness (owner A). Writes eval/results/latest.json and EVALUATION.md.

Run:  python -m eval.run_eval                 # no LLM calls (baseline, retrieval, gate, latency)
      python -m eval.run_eval --llm 30        # + LLM on a stratified sample (cached, throttled)
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from app import config, taxonomy
from app.build_index import ticket_doc
from app.llm_provider import available
from app.pipeline import Pipeline
from app.schema import Ticket, TicketIn

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def p95(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))] if xs else 0.0


def retrieval_eval(p: Pipeline, test: list[Ticket]) -> dict:
    out = {}
    r = p.retriever
    for mode in ("bm25", "semantic", "hybrid"):
        rec5, prec5, rr, kb1 = [], [], [], []
        for t in test:
            q = ticket_doc(t)
            hits = r.search_tickets(q, k=10, mode=mode)
            rel = [h.metadata.get("subcategory") == t.subcategory for h in hits]
            rec5.append(any(rel[:5]))
            prec5.append(sum(rel[:5]) / 5)
            rr.append(next((1 / (i + 1) for i, x in enumerate(rel) if x), 0.0))
            kb = r.search_kb(q, k=1, mode=mode)            # no hints: pure retrieval quality
            kb1.append(bool(kb) and kb[0].metadata.get("subcategory") == t.subcategory)
        out[mode] = {"recall@5": statistics.mean(rec5), "precision@5": statistics.mean(prec5),
                     "mrr@10": statistics.mean(rr), "kb_top1": statistics.mean(kb1)}
    return out


def trap_eval(p: Pipeline, traps: list[dict]) -> dict:
    out = {}
    for mode in ("bm25", "semantic", "hybrid"):
        ok = [bool(h := p.retriever.search_tickets(t["text"], k=1, mode=mode)) and h[0].metadata.get("subcategory") == t["subcategory"]
              for t in traps]
        out[f"top1_{mode}"] = statistics.mean(ok)
    if p.baseline:
        out["baseline_cls"] = statistics.mean(p.baseline.predict(t["text"]).subcategory == t["subcategory"] for t in traps)
    return out


def gate_eval(p: Pipeline, texts: list[str], use_llm: bool, overrides, sleep: float) -> dict:
    st = Counter()
    for x in texts:
        st[p.analyze(TicketIn(description=x), use_llm=use_llm, llm_overrides=overrides).status] += 1
        if use_llm and sleep:
            time.sleep(sleep)
    n = len(texts)
    return {"n": n, "escalate": st["ESCALATE"] / n, "needs_review": st["NEEDS_REVIEW"] / n, "ok": st["OK"] / n}


def pipeline_eval(p: Pipeline, tickets: list[Ticket], use_llm: bool, overrides, sleep: float) -> dict:
    y, yhat_cat, ysub, yhat_sub, prio_ok, p1_rec = [], [], [], [], [], []
    conf_bins = defaultdict(lambda: [0, 0])
    status = Counter()
    lat, paths_total, paths_cited, dropped, degraded = [], 0, 0, 0, 0
    for t in tickets:
        tin = TicketIn(title=t.title, description=t.text, application=t.application, environment=t.environment)
        t0 = time.perf_counter()
        r = p.analyze(tin, use_llm=use_llm, llm_overrides=overrides)
        lat.append((time.perf_counter() - t0) * 1000)
        c = r.classification
        y.append(t.category); yhat_cat.append(c.category)
        ysub.append(t.subcategory); yhat_sub.append(c.subcategory)
        if t.priority:
            prio_ok.append(c.priority == t.priority)
            if t.priority == "P1":
                p1_rec.append(c.priority == "P1")
        b = min(int(r.confidence * 10), 9) / 10
        conf_bins[b][0] += 1
        conf_bins[b][1] += int(c.subcategory == t.subcategory)
        status[r.status] += 1
        paths_total += len(r.resolution_paths)
        paths_cited += sum(1 for x in r.resolution_paths if x.evidence)
        dropped += r.dropped_claims
        degraded += int(r.degraded_mode)
        if use_llm and sleep and not (r.provider or "").endswith("(cache)"):
            time.sleep(sleep)
    n = len(tickets)
    return {
        "n": n, "category_acc": accuracy_score(y, yhat_cat), "subcategory_acc": accuracy_score(ysub, yhat_sub),
        "subcategory_macro_f1": f1_score(ysub, yhat_sub, average="macro", zero_division=0),
        "priority_acc": statistics.mean(prio_ok) if prio_ok else None,
        "p1_recall": statistics.mean(p1_rec) if p1_rec else None,
        "status": {k: v / n for k, v in status.items()},
        "calibration": {f"{k:.1f}-{k + 0.1:.1f}": {"n": v[0], "acc": v[1] / v[0]} for k, v in sorted(conf_bins.items())},
        "paths_with_citation": paths_cited / paths_total if paths_total else None,
        "dropped_claims_per_ticket": dropped / n, "degraded_share": degraded / n,
        "latency_p50_ms": statistics.median(lat), "latency_p95_ms": p95(lat),
        "confusions": Counter(f"{a} -> {b}" for a, b in zip(ysub, yhat_sub) if a != b).most_common(6),
    }


def stratified_sample(tickets: list[Ticket], n: int) -> list[Ticket]:
    rng = random.Random(config.SEED)
    by = defaultdict(list)
    for t in tickets:
        by[t.subcategory].append(t)
    out, i = [], 0
    pools = [rng.sample(v, len(v)) for v in by.values()]
    while len(out) < n and any(pools):
        pool = pools[i % len(pools)]
        if pool:
            out.append(pool.pop())
        i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=int, default=0, help="LLM sample size (0 = no LLM calls)")
    ap.add_argument("--model", default="openai/gpt-oss-20b", help="Groq model for the LLM sample")
    ap.add_argument("--sleep", type=float, default=2.0)
    a = ap.parse_args()

    p = Pipeline()
    split = json.loads(config.SPLITS_JSON.read_text(encoding="utf-8"))
    test = [Ticket(**t) for t in split["test"]]
    sets = json.loads((HERE / "traps.json").read_text(encoding="utf-8"))
    res: dict = {"index": p.retriever.stats(), "test_size": len(test), "weights": {
        "W_SEM": config.W_SEM, "W_BM25": config.W_BM25, "W_META": config.W_META,
        "C_LLM": config.C_LLM, "C_RET": config.C_RET, "C_AGR": config.C_AGR, "T_OK": config.T_OK, "T_ESC": config.T_ESC}}

    print("retrieval…"); res["retrieval"] = retrieval_eval(p, test)
    print("traps…"); res["traps"] = trap_eval(p, sets["traps"])
    print("pipeline (no LLM) on the full test split…")
    res["pipeline_baseline"] = pipeline_eval(p, test, False, None, 0)
    res["gate_baseline"] = {"out_of_scope": gate_eval(p, sets["out_of_scope"], False, None, 0),
                            "ambiguous": gate_eval(p, sets["ambiguous"], False, None, 0)}
    if a.llm and available():
        ov = {"groq": a.model}
        sample = stratified_sample(test, a.llm)
        print(f"LLM on {len(sample)} tickets with {a.model}…")
        res["pipeline_llm"] = pipeline_eval(p, sample, True, ov, a.sleep)
        res["pipeline_llm"]["model"] = a.model
        res["pipeline_baseline_same_sample"] = pipeline_eval(p, sample, False, None, 0)
        res["gate_llm"] = {"out_of_scope": gate_eval(p, sets["out_of_scope"], True, ov, a.sleep),
                           "ambiguous": gate_eval(p, sets["ambiguous"], True, ov, a.sleep)}
    elif a.llm:
        print("No LLM key set: skipping the LLM sample.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (config.ROOT / "EVALUATION.md").write_text(report(res), encoding="utf-8")
    print("Wrote eval/results/latest.json and EVALUATION.md")


def report(r: dict) -> str:
    L = ["# Evaluation report", "",
         f"Generated by `python -m eval.run_eval`. Index: {r['index']['tickets']} historical tickets, "
         f"{r['index']['kb_articles']} KB articles, embedder `{r['index']['embedder']}`, backend {r['index']['backend']}. "
         f"Test split: {r['test_size']} held-out tickets that are **not** in the index.", "",
         "## 1. Retrieval: BM25 vs semantic vs hybrid", "",
         "Relevance = same subcategory as the query ticket. KB top-1 = the best KB article is for the right subcategory (no hints).", "",
         "| Mode | Recall@5 | Precision@5 | MRR@10 | KB top-1 |", "|---|---|---|---|---|"]
    for m, v in r["retrieval"].items():
        L.append(f"| {m} | {pct(v['recall@5'])} | {pct(v['precision@5'])} | {v['mrr@10']:.3f} | {pct(v['kb_top1'])} |")
    t = r["traps"]
    L += ["", "## 2. Keyword traps (20 hand-written tickets that share words with the wrong topic)", "",
          "| Method | Top-1 correct subcategory |", "|---|---|",
          f"| BM25 only | {pct(t['top1_bm25'])} |", f"| Semantic only | {pct(t['top1_semantic'])} |",
          f"| Hybrid | {pct(t['top1_hybrid'])} |"]
    if "baseline_cls" in t:
        L.append(f"| Baseline classifier | {pct(t['baseline_cls'])} |")

    def cls_rows(name: str, v: dict) -> list[str]:
        return [f"| {name} | {v['n']} | {pct(v['category_acc'])} | {pct(v['subcategory_acc'])} | "
                f"{v['subcategory_macro_f1']:.3f} | {pct(v['priority_acc']) if v['priority_acc'] is not None else '-'} | "
                f"{pct(v['p1_recall']) if v['p1_recall'] is not None else '-'} |"]

    L += ["", "## 3. Classification and priority", "",
          "| System | n | Category acc | Subcategory acc | Subcat macro-F1 | Priority acc | P1 recall |",
          "|---|---|---|---|---|---|---|"]
    L += cls_rows("Baseline (TF-IDF + LogReg), full test split", r["pipeline_baseline"])
    if "pipeline_llm" in r:
        L += cls_rows("Baseline, same sample as the LLM", r["pipeline_baseline_same_sample"])
        L += cls_rows(f"LLM pipeline ({r['pipeline_llm']['model']})", r["pipeline_llm"])
        deg = r["pipeline_llm"].get("degraded_share") or 0
        if deg:
            L += ["", f"In the LLM sample, {pct(deg)} of tickets fell back to degraded mode (free-tier rate limit or "
                      "provider busy), so those rows count baseline answers capped at NEEDS_REVIEW."]
    else:
        L += ["", "_LLM rows appear after running with `--llm 30` once API keys are set._"]
    L += ["", "Note: synthetic priority labels come from the same rules as the rule floor, so baseline priority accuracy "
              "is close to 100% by construction; the LLM can only raise urgency above the rule floor.", "",
          "Most common confusions (baseline): " + "; ".join(f"{a} ({n})" for a, n in r["pipeline_baseline"]["confusions"]) or "none"]

    L += ["", "## 4. Confidence gate", "", "| Ticket set | System | ESCALATE | NEEDS_REVIEW | OK |", "|---|---|---|---|---|"]
    for sysname, key in (("baseline", "gate_baseline"), ("LLM", "gate_llm")):
        if key in r:
            for setname, v in r[key].items():
                L.append(f"| {setname} ({v['n']}) | {sysname} | {pct(v['escalate'])} | {pct(v['needs_review'])} | {pct(v['ok'])} |")
    for name, key in (("baseline", "pipeline_baseline"), ("LLM", "pipeline_llm")):
        if key in r:
            s = r[key]["status"]
            L.append(f"| held-out IT tickets ({r[key]['n']}) | {name} | {pct(s.get('ESCALATE', 0))} | "
                     f"{pct(s.get('NEEDS_REVIEW', 0))} | {pct(s.get('OK', 0))} |")
    L += ["", "Degraded (no-LLM) mode is capped at NEEDS_REVIEW by design, so OK only appears with the LLM.", "",
          "**Calibration (accuracy within each confidence bin, baseline pipeline):**", "",
          "| Confidence | n | Subcategory accuracy |", "|---|---|---|"]
    for b, v in r["pipeline_baseline"]["calibration"].items():
        L.append(f"| {b} | {v['n']} | {pct(v['acc'])} |")

    L += ["", "## 5. Traceability and latency", "", "| System | Paths with ≥1 valid citation | Dropped claims / ticket | p50 ms | p95 ms |",
          "|---|---|---|---|---|"]
    for name, key in (("baseline (degraded)", "pipeline_baseline"), ("LLM", "pipeline_llm")):
        if key in r:
            v = r[key]
            pc = pct(v["paths_with_citation"]) if v["paths_with_citation"] is not None else "-"
            L.append(f"| {name} | {pc} | {v['dropped_claims_per_ticket']:.2f} | {v['latency_p50_ms']:.0f} | {v['latency_p95_ms']:.0f} |")
    L += ["", "## 6. Weights used", "", "```json", json.dumps(r["weights"], indent=2), "```", "",
          "## 7. Limitations", "",
          "- The dataset is synthetic and template-based, so retrieval and classification scores are optimistic; "
          "the keyword-trap and out-of-scope sets are the more honest tests.",
          "- Priority labels are rule-derived in the synthetic data.",
          "- LLM results depend on the free-tier model available on the day; every analysis records provider and model.",
          "- LLM latency here includes waiting out the Groq free-tier limit (about 8,000 tokens/min, roughly 3 tickets/min) "
          "during a batch run; a single interactive ticket takes about 2-3 s, and cached tickets are instant.",
          "- The LLM sample is small (30 tickets, stratified by subcategory), so one ticket = 3.3 percentage points.",
          "- Worked examples with full evidence: [docs/WORKED_EXAMPLES.md](docs/WORKED_EXAMPLES.md).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()

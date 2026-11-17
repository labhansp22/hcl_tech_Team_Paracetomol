"""C7 Pipeline orchestrator (owner B): redact -> baseline -> retrieve -> LLM -> validate -> confidence gate.

Fallback layers (Section 9): L1 normal, L2 retry with validator errors, L2b provider failover (llm_provider),
L3 degraded (baseline + KB steps verbatim), L4 nothing retrieved -> escalate.
"""
from __future__ import annotations

import logging
import time
import uuid

from pydantic import ValidationError

from app import config, llm, taxonomy
from app.baseline import Baseline
from app.confidence import decide
from app.llm_provider import LLMUnavailable, available
from app.redact import redact
from app.retriever import Retriever
from app.schema import (AnalysisResult, BaselinePrediction, Classification, EscalationInfo, Evidence, Hit,
                        ResolutionPath, TicketIn)
from app.validator import validate

log = logging.getLogger(__name__)

DEFAULT_QUESTIONS = [
    "Which application or device is affected, and what exact error message do you see?",
    "When did it start, and does it affect only you or other users too?",
    "What have you already tried (restart, another device, another network)?",
]


class Pipeline:
    def __init__(self, retriever: Retriever | None = None, baseline: Baseline | None = None,
                 tax: dict | None = None):
        self.retriever = retriever or Retriever()
        if baseline is None and config.BASELINE_PKL.exists():
            baseline = Baseline.load()
        self.baseline = baseline
        self.tax = tax or taxonomy.load()

    # ------------------------------------------------------------------
    def analyze(self, t: TicketIn, use_llm: bool = True, exclude_ids=(), llm_overrides: dict | None = None,
                use_cache: bool = True) -> AnalysisResult:
        timings: dict[str, int] = {}
        clock = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[name] = int((now - clock) * 1000)
            clock = now

        # 1. redact
        desc, r1 = redact(t.description)
        title, r2 = redact(t.title) if t.title else (None, [])
        text = f"{title}. {desc}" if title else desc
        redactions = sorted(set(r1 + r2))
        rule_prio = taxonomy.rule_priority(text, t.environment)
        lap("redact")

        # 2. baseline
        base: BaselinePrediction | None = self.baseline.predict(text) if self.baseline else None
        lap("baseline")

        # 3. retrieve
        qv = self.retriever.embed_query(text)
        ticket_hits = self.retriever.search_tickets(
            text, t.filters, k=config.K_TICKETS, exclude_ids=exclude_ids, qv=qv,
            query_meta={"application": t.application, "environment": t.environment})
        hints = [s for s, p in (base.top_subcategories if base else [])[:2] if p >= 0.15] or (
            [base.subcategory] if base else [])
        kb_hits = self.retriever.search_kb(text, k=config.K_KB, hint_subcats=hints,
                                           hint_cat=base.category if base else None, qv=qv)
        lap("retrieve")

        # 4-5. LLM + validation (with L2 retry), or L3 degraded
        provider = model = None
        degraded, dropped, insufficient = True, 0, False
        questions: list[str] = []
        classification: Classification | None = None
        paths: list[ResolutionPath] = []
        llm_dups: list[str] = []
        all_hits = ticket_hits + kb_hits
        llm_conf = base.probability if base else 0.3

        if use_llm and available() and all_hits:
            try:
                msgs = llm.build_messages(text, ticket_hits, kb_hits, self.retriever.source_text, self.tax, base,
                                          rule_prio, {"application": t.application, "environment": t.environment})
                res = llm.analyze(msgs, llm_overrides, use_cache)
                lap("llm")
                rep = validate(res.output, all_hits, self.retriever.source_text, self.tax)
                if not rep.paths and res.output.resolution_paths and not res.output.insufficient_evidence:
                    retry = msgs + [{"role": "assistant", "content": res.output.model_dump_json()},
                                    {"role": "user", "content": "Validation failed:\n- " + "\n- ".join(rep.errors[:8])
                                     + "\nFix it: cite only ids from the evidence and copy excerpts exactly. "
                                       "Return the full corrected JSON."}]
                    res = llm.analyze(retry, llm_overrides, use_cache)
                    rep = validate(res.output, all_hits, self.retriever.source_text, self.tax)
                    lap("llm_retry")
                out = res.output
                provider, model = res.provider + (" (cache)" if res.cached else ""), res.model
                degraded, dropped, insufficient = False, rep.dropped, out.insufficient_evidence
                questions = out.clarifying_questions
                paths, llm_dups = rep.paths, rep.duplicate_ids
                llm_conf = out.confidence
                cat, sub = rep.category, rep.subcategory
                src = "llm"
                if not (cat and sub):                       # invalid label -> baseline label
                    cat, sub, src = (base.category, base.subcategory, "baseline") if base else (cat or "Inquiry", sub or "How-To", "llm")
                sec = None
                if out.secondary:
                    sc, ss = taxonomy.canonical(self.tax, out.secondary.category, out.secondary.subcategory)
                    if sc and ss and ss != sub:
                        sec = (sc, ss, round(out.secondary.confidence, 2))
                classification = Classification(
                    category=cat, subcategory=sub, priority=taxonomy.more_urgent(out.priority, rule_prio),
                    confidence=round(out.confidence, 3), secondary=sec, rationale=out.rationale, source=src)
            except (LLMUnavailable, ValidationError, ValueError) as e:
                log.warning("LLM path failed, degraded mode: %s", e)
                lap("llm_failed")

        if degraded:
            classification, paths = self._degraded(base, ticket_hits, kb_hits, rule_prio)
        lap("validate")

        # duplicates: high semantic similarity + same category (keyword overlap alone never counts)
        for h in ticket_hits:
            cos = h.score_breakdown.get("semantic", 0)
            same = h.metadata.get("category") == classification.category
            if same and not h.score_breakdown.get("keyword_only") and (
                    cos >= config.DUP_COS or (h.source_id in llm_dups and cos >= 0.75)):
                h.is_duplicate = True

        # 6. confidence gate
        gate = decide(llm_conf=llm_conf, category=classification.category, priority=classification.priority,
                      baseline=base, kb_hits=kb_hits, ticket_hits=ticket_hits, valid_paths=len(paths),
                      dropped=dropped, insufficient=insufficient, degraded=degraded)
        if not all_hits:
            gate.status = "ESCALATE"
            gate.reasons.append("No matching history or KB article: escalate")

        escalation = None
        if gate.status == "ESCALATE":
            escalation = self._escalation(classification, ticket_hits, kb_hits, questions, gate)
        lap("gate")

        return AnalysisResult(
            analysis_id=f"AN-{uuid.uuid4().hex[:10]}", status=gate.status, confidence=gate.score,
            confidence_reasons=gate.reasons, classification=classification, similar_tickets=ticket_hits,
            kb_matches=kb_hits, resolution_paths=paths, escalation=escalation, redactions=redactions,
            dropped_claims=dropped, degraded_mode=degraded, timings_ms=timings, provider=provider, model=model,
            kb_gap=gate.kb_gap, signals=gate.signals)

    # ------------------------------------------------------------------
    def _degraded(self, base: BaselinePrediction | None, ticket_hits: list[Hit], kb_hits: list[Hit],
                  rule_prio: str) -> tuple[Classification, list[ResolutionPath]]:
        if base:
            cls = Classification(category=base.category, subcategory=base.subcategory, priority=rule_prio,
                                 confidence=round(base.probability, 3),
                                 rationale="Baseline TF-IDF classifier (LLM unavailable). Key terms: "
                                           + ", ".join(base.top_terms or ["-"]),
                                 source="baseline")
        else:
            cat = taxonomy.majority([h.metadata.get("category") for h in ticket_hits[:5]]) or "Inquiry"
            sub = taxonomy.majority([h.metadata.get("subcategory") for h in ticket_hits[:5]]) or "How-To"
            cls = Classification(category=cat, subcategory=sub, priority=rule_prio, confidence=0.3,
                                 rationale="Majority label of the most similar tickets (LLM unavailable).",
                                 source="baseline")
        paths = []
        # articles for the predicted subcategory first, then by score; a 2nd path only if nearly as strong
        ranked = sorted(kb_hits, key=lambda h: (h.metadata.get("subcategory") != cls.subcategory, -h.score))
        top = ranked[0].score if ranked else 0.0
        for h in ranked[:2]:
            a = self.retriever.articles.get(h.source_id)
            if not a or h.score < config.KB_GAP or (paths and h.score < 0.75 * top):
                continue
            paths.append(ResolutionPath(
                rank=len(paths) + 1, title=a.title, steps=a.resolution_steps,
                evidence=[Evidence(source_id=a.kb_id, source_type="kb", excerpt=a.resolution_steps[0],
                                   reason=f"KB article ranked #{len(paths) + 1} for this ticket (score {h.score:.2f})")]))
        return cls, paths

    def _escalation(self, cls: Classification, ticket_hits: list[Hit], kb_hits: list[Hit],
                    questions: list[str], gate) -> EscalationInfo:
        group = None
        if kb_hits and kb_hits[0].score >= config.KB_GAP:
            group = kb_hits[0].metadata.get("assignment_group")
        group = group or taxonomy.majority([h.metadata.get("assignment_group") for h in ticket_hits[:5]
                                            if not h.score_breakdown.get("keyword_only")]) or "Service Desk L1"
        return EscalationInfo(
            assignment_group=group, clarifying_questions=questions or DEFAULT_QUESTIONS,
            diagnostic_bundle={
                "classification": f"{cls.category} / {cls.subcategory} ({cls.priority}, {cls.source})",
                "top_tickets": [f"{h.source_id} ({h.score:.2f})" for h in ticket_hits[:3]],
                "top_kb": [f"{h.source_id} ({h.score:.2f})" for h in kb_hits[:2]],
                "signals": gate.signals, "reasons": gate.reasons[1:],
            })

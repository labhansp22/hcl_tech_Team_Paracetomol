"""C6 Confidence gate (owner B): three independent signals -> OK / NEEDS_REVIEW / ESCALATE (Section 7).

score = C_LLM*llm_conf + C_RET*retrieval + C_AGR*agreement
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app import config
from app.schema import BaselinePrediction, Hit


@dataclass
class Gate:
    status: str
    score: float
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)
    kb_gap: bool = False


def retrieval_signal(kb_hits: list[Hit], ticket_hits: list[Hit]) -> float:
    top_kb = kb_hits[0].score if kb_hits else 0.0
    # keyword-only matches don't count as real support
    tk = [h.score for h in ticket_hits if not h.score_breakdown.get("keyword_only")][:3]
    mean_t = sum(tk) / len(tk) if tk else 0.0
    return 0.5 * top_kb + 0.5 * mean_t


def agreement_signal(category: str, baseline: BaselinePrediction | None, ticket_hits: list[Hit]) -> tuple[float, str]:
    if baseline is None:
        base, note = 0.5, "no baseline available"
    elif category == baseline.category:
        base, note = 1.0, "baseline agrees"
    elif category in [c for c, _ in baseline.top_categories[:3]]:
        base, note = 0.5, f"baseline prefers {baseline.category}"
    else:
        base, note = 0.0, f"baseline disagrees ({baseline.category})"
    top5 = ticket_hits[:5]
    share = sum(h.metadata.get("category") == category for h in top5) / len(top5) if top5 else 0.0
    return base * share, f"{note}; {sum(h.metadata.get('category') == category for h in top5)}/{len(top5)} similar tickets share the category"


def decide(*, llm_conf: float, category: str, priority: str, baseline: BaselinePrediction | None,
           kb_hits: list[Hit], ticket_hits: list[Hit], valid_paths: int, dropped: int,
           insufficient: bool, degraded: bool) -> Gate:
    ret = retrieval_signal(kb_hits, ticket_hits)
    agr, agr_note = agreement_signal(category, baseline, ticket_hits)
    score = config.C_LLM * llm_conf + config.C_RET * ret + config.C_AGR * agr
    top_kb = kb_hits[0].score if kb_hits else 0.0
    g = Gate(status="OK", score=round(score, 3),
             signals={"llm": round(llm_conf, 3), "retrieval": round(ret, 3), "agreement": round(agr, 3),
                      "top_kb": round(top_kb, 3)})
    g.reasons.append(f"Signals: {'baseline' if degraded else 'LLM'} confidence {llm_conf:.2f}, "
                     f"retrieval {ret:.2f}, agreement {agr:.2f} ({agr_note})")

    esc, rev = [], []
    if score < config.T_ESC:
        esc.append(f"Combined confidence {score:.2f} is below {config.T_ESC}")
    if valid_paths == 0:
        esc.append("No resolution path is backed by valid evidence")
    if insufficient:
        esc.append("The model reported that the evidence does not cover this ticket")
    if top_kb < config.KB_GAP:
        g.kb_gap = True
        esc.append(f"Knowledge gap: best KB match is only {top_kb:.2f} (< {config.KB_GAP})")
    if config.T_ESC <= score < config.T_OK:
        rev.append(f"Combined confidence {score:.2f} is between {config.T_ESC} and {config.T_OK}")
    if ret < config.T_RET_OK:
        rev.append(f"Weak evidence: retrieval {ret:.2f} is below {config.T_RET_OK}, so a confident model is not enough")
    if priority == "P1":
        rev.append("P1 tickets are always reviewed by a human")
    if dropped > 0:
        rev.append(f"Validator removed {dropped} unsupported claim(s)")
    if degraded:
        rev.append("LLM unavailable: baseline classification and KB steps shown verbatim")

    if esc:
        g.status, g.reasons = "ESCALATE", g.reasons + esc + rev
    elif rev:
        g.status, g.reasons = "NEEDS_REVIEW", g.reasons + rev
    else:
        g.reasons.append(f"Combined confidence {score:.2f} ≥ {config.T_OK} with valid evidence")
    return g

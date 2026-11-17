"""C5 Evidence validator (owner B): code checks, never trust the LLM.

* every cited source_id must be one of the retrieved hits
* every excerpt must be a verbatim substring of that source (after whitespace/quote/case normalisation;
  "..." may join two verbatim parts that appear in order)
* labels must belong to the taxonomy
* a resolution path with no valid evidence is dropped
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app import taxonomy as taxmod
from app.schema import Evidence, Hit, LLMOutput, ResolutionPath

# LLMs often emit typographic variants (curly quotes, non-breaking hyphens, no-break spaces) for plain ASCII
_Q = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
                    "\u00a0": " ", "\u202f": " ", "\u2009": " "})


def normalise(s: str) -> str:
    s = (s or "").translate(_Q).lower()
    # section labels / list numbers the model adds in front of a real quote ("Resolution steps: 2. ...")
    s = re.sub(r"^\s*((symptoms|diagnostic steps|resolution steps|escalate when|resolution)\s*:\s*)?(\d+\.\s+)?", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .,;:'\"`")


def excerpt_ok(excerpt: str, source: str) -> bool:
    src = re.sub(r"\s+", " ", (source or "").translate(_Q).lower())
    parts = [normalise(p) for p in re.split(r"\.\.\.|…", excerpt or "")]
    parts = [p for p in parts if p]
    if not parts or sum(len(p) for p in parts) < 8:
        return False
    pos = 0
    for p in parts:
        i = src.find(p, pos)
        if i < 0:
            return False
        pos = i + len(p)
    return True


@dataclass
class ValidationReport:
    paths: list[ResolutionPath] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    category: str | None = None
    subcategory: str | None = None
    dropped: int = 0
    errors: list[str] = field(default_factory=list)


def validate(out: LLMOutput, hits: list[Hit], source_text, tax: dict) -> ValidationReport:
    rep = ValidationReport()
    by_id = {h.source_id: h for h in hits}

    cat, sub = taxmod.canonical(tax, out.category, out.subcategory)
    empty = not (out.category or "").strip() and not (out.subcategory or "").strip()
    if (not cat or not sub) and empty and out.insufficient_evidence:
        rep.errors.append("no label: model reported insufficient evidence (not counted as a dropped claim)")
    elif not cat or not sub:
        rep.dropped += 1
        rep.errors.append(f"label '{out.category} / {out.subcategory}' is not in the taxonomy")
    rep.category, rep.subcategory = cat, sub

    for pid, p in enumerate(out.resolution_paths, 1):
        good: list[Evidence] = []
        for ev in p.evidence:
            sid = ev.source_id.strip().strip("[]")
            if sid not in by_id:
                rep.dropped += 1
                rep.errors.append(f"path {pid}: {sid} was not retrieved")
                continue
            if not excerpt_ok(ev.excerpt, source_text(sid) or ""):
                rep.dropped += 1
                rep.errors.append(f"path {pid}: excerpt not found verbatim in {sid}: '{ev.excerpt[:80]}'")
                continue
            good.append(Evidence(source_id=sid, source_type=by_id[sid].source_type,
                                 excerpt=ev.excerpt.strip(), reason=ev.reason or "supports this step"))
        if good and p.steps:
            rep.paths.append(ResolutionPath(rank=len(rep.paths) + 1, title=p.title, steps=p.steps, evidence=good))
        else:
            rep.errors.append(f"path {pid} '{p.title}' dropped: no valid evidence")

    for d in out.duplicate_ids:
        h = by_id.get(d)
        if h and h.source_type == "ticket":
            rep.duplicate_ids.append(d)
        else:
            rep.dropped += 1
            rep.errors.append(f"duplicate id {d} was not a retrieved ticket")
    return rep

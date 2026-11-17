"""C3 Hybrid retriever (owner B): ChromaDB (semantic + metadata filters) + BM25 (keywords).

score = W_SEM*cos + W_BM25*bm25_norm + W_META*meta     (ARCHITECTURE.md Section 8)

* Candidates: top-N from ChromaDB  UNION  top-N from BM25 (same filters applied to both).
* cos for every candidate comes from the stored normalised embeddings, so both lists are scored the same way.
* bm25_norm = raw / max(best raw score, BM25_FLOOR), so weak keyword overlaps stay weak.
* If the query gives no metadata, the meta weight is dropped and the other two are re-normalised.
* RETRIEVER_BACKEND=numpy replaces ChromaDB with an in-memory dot product behind the same interface.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
import re
from datetime import datetime, time
from typing import Iterable

import numpy as np

from app import config
from app.embeddings import Embedder
from app.schema import Hit, KBArticle, SearchFilters

log = logging.getLogger(__name__)

STOP = set("""a an and are as at be been but by can cannot could do does for from has have hi hello i if in into is it
its me my no not of on or our please so than that the their them then there this to too up us was we were what when
which while with you your am after before since team good morning only every all also any just still""".split())
_TOK = re.compile(r"[a-z0-9][a-z0-9_.\-]*[a-z0-9]|[a-z0-9]")
FILTER_FIELDS = ("category", "subcategory", "application", "environment", "status", "assignment_group")


def tokenize(s: str) -> list[str]:
    return [t for t in _TOK.findall((s or "").lower()) if t not in STOP]


def kb_index_text(a: KBArticle) -> str:
    """What we embed / BM25-index for a KB article: title + symptoms."""
    return f"{a.title}. " + " ".join(a.symptoms)


def kb_full_text(a: KBArticle) -> str:
    """What the LLM sees and what excerpts are validated against."""
    lines = [a.title, "Symptoms: " + "; ".join(a.symptoms), "Diagnostic steps:"]
    lines += [f"{i}. {s}" for i, s in enumerate(a.diagnostic_steps, 1)]
    lines += ["Resolution steps:"] + [f"{i}. {s}" for i, s in enumerate(a.resolution_steps, 1)]
    lines += [f"Escalate when: {a.escalation_criteria}", f"Assignment group: {a.assignment_group}"]
    return "\n".join(lines)


def to_where(f: SearchFilters | None) -> dict | None:
    """SearchFilters -> ChromaDB `where` clause."""
    if not f:
        return None
    conds: list[dict] = [{k: getattr(f, k)} for k in FILTER_FIELDS if getattr(f, k)]
    if f.date_from:
        conds.append({"created_ts": {"$gte": int(datetime.combine(f.date_from, time.min).timestamp())}})
    if f.date_to:
        conds.append({"created_ts": {"$lte": int(datetime.combine(f.date_to, time.max).timestamp())}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def matches(meta: dict, f: SearchFilters | None) -> bool:
    """Same filter in Python, for BM25 candidates and the numpy backend."""
    if not f:
        return True
    for k in FILTER_FIELDS:
        v = getattr(f, k)
        if v and meta.get(k) != v:
            return False
    ts = meta.get("created_ts", 0)
    if f.date_from and ts < datetime.combine(f.date_from, time.min).timestamp():
        return False
    if f.date_to and ts > datetime.combine(f.date_to, time.max).timestamp():
        return False
    return True


class _Corpus:
    """One searchable collection: ids, docs, metadata, BM25 and the embedding matrix."""

    def __init__(self, d: dict, vectors: np.ndarray):
        self.ids: list[str] = d["ids"]
        self.docs: list[str] = d["docs"]
        self.metas: list[dict] = d["metas"]
        self.full: list[str] = d["full"]
        self.titles: list[str | None] = d["titles"]
        self.bm25 = d["bm25"]
        self.vec = vectors
        self.pos = {sid: i for i, sid in enumerate(self.ids)}


class Retriever:
    def __init__(self, embedder: Embedder | None = None, backend: str | None = None):
        meta = json.loads(config.INDEX_META.read_text(encoding="utf-8"))
        with open(config.BM25_PKL, "rb") as f:
            data = pickle.load(f)
        vecs = np.load(config.VECTORS_NPZ)
        self.tickets = _Corpus(data["tickets"], vecs["tickets"])
        self.kb = _Corpus(data["kb"], vecs["kb"])
        self.articles: dict[str, KBArticle] = {a["kb_id"]: KBArticle(**a) for a in data["articles"]}
        self.index_meta = meta
        self.embedder = embedder or Embedder(meta["embedder"])
        self.semantic_ok = self.embedder.name == meta["embedder"]
        if not self.semantic_ok:
            log.warning("Index built with %s but %s is loaded: semantic search disabled, BM25 only",
                        meta["embedder"], self.embedder.name)
        self.backend = backend or config.RETRIEVER_BACKEND
        self._cols = {}
        if self.backend == "chroma":
            try:
                import chromadb
                from chromadb.config import Settings
                client = chromadb.PersistentClient(path=str(config.CHROMA_DIR),
                                                   settings=Settings(anonymized_telemetry=False))
                self._cols = {"tickets": client.get_collection(config.TICKETS_COLLECTION),
                              "kb": client.get_collection(config.KB_COLLECTION)}
            except Exception as e:  # pragma: no cover
                log.warning("ChromaDB unavailable (%s); falling back to numpy backend", e)
                self.backend = "numpy"
        self._reranker = None
        if config.RERANK:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(config.RERANK_MODEL, device="cpu")
            except Exception as e:  # pragma: no cover
                log.warning("Reranker failed to load (%s); continuing without it", e)

    # ---- public API ------------------------------------------------------
    def embed_query(self, query: str) -> np.ndarray | None:
        return self.embedder.encode([query])[0] if self.semantic_ok else None

    def search_tickets(self, query: str, filters: SearchFilters | None = None, k: int = config.K_TICKETS,
                       query_meta: dict | None = None, exclude_ids: Iterable[str] = (), mode: str = "hybrid",
                       qv: np.ndarray | None = None) -> list[Hit]:
        c = self.tickets
        exclude = set(exclude_ids)
        mask = np.array([matches(m, filters) and sid not in exclude for sid, m in zip(c.ids, c.metas)])
        qmeta = {k_: v for k_, v in (query_meta or {}).items() if v and k_ in ("application", "environment", "assignment_group")}

        def meta_fn(i: int) -> float:
            if not qmeta:
                return 0.0
            return sum(c.metas[i].get(f) == v for f, v in qmeta.items()) / len(qmeta)

        if qv is None and mode != "bm25":
            qv = self.embed_query(query)
        hits = self._hybrid("tickets", c, query, qv, mask, to_where(filters), len(exclude), meta_fn,
                            use_meta=bool(qmeta), k=k, mode=mode)
        return hits

    def search_kb(self, query: str, k: int = config.K_KB, hint_subcats: Iterable[str] = (),
                  hint_cat: str | None = None, mode: str = "hybrid", qv: np.ndarray | None = None) -> list[Hit]:
        c = self.kb
        hints = [h for h in hint_subcats if h]
        mask = np.ones(len(c.ids), dtype=bool)

        def meta_fn(i: int) -> float:
            m = c.metas[i]
            if m["subcategory"] in hints:
                return 1.0
            return 0.5 if hint_cat and m["category"] == hint_cat else 0.0

        if qv is None and mode != "bm25":
            qv = self.embed_query(query)
        forced = [i for i, m in enumerate(c.metas) if m["subcategory"] in hints]   # mapped articles first
        return self._hybrid("kb", c, query, qv, mask, None, 0, meta_fn, use_meta=bool(hints or hint_cat),
                            k=k, mode=mode, forced=forced)

    def source_text(self, source_id: str) -> str | None:
        for c in (self.tickets, self.kb):
            i = c.pos.get(source_id)
            if i is not None:
                return c.full[i]
        return None

    def ticket_meta(self, source_id: str) -> dict | None:
        i = self.tickets.pos.get(source_id)
        return self.tickets.metas[i] if i is not None else None

    def stats(self) -> dict:
        return {"tickets": len(self.tickets.ids), "kb_articles": len(self.kb.ids), "backend": self.backend,
                "embedder": self.embedder.name, "semantic": self.semantic_ok, "reranker": self._reranker is not None,
                "built_at": self.index_meta.get("built_at")}

    # ---- internals -------------------------------------------------------
    def _semantic_candidates(self, name: str, c: _Corpus, qv: np.ndarray, mask: np.ndarray,
                             where: dict | None, extra: int) -> list[int]:
        n = config.CANDIDATES
        if self.backend == "chroma" and name in self._cols:
            col = self._cols[name]
            try:
                res = col.query(query_embeddings=[qv.tolist()], n_results=min(n + extra, len(c.ids)),
                                where=where, include=["distances"])
                return [c.pos[i] for i in res["ids"][0] if i in c.pos and mask[c.pos[i]]][:n]
            except Exception as e:  # pragma: no cover
                log.warning("Chroma query failed (%s); numpy fallback for this query", e)
        sims = c.vec @ qv
        sims[~mask] = -np.inf
        top = np.argsort(-sims)[:n]
        return [int(i) for i in top if np.isfinite(sims[i])]

    def _hybrid(self, name, c: _Corpus, query, qv, mask, where, extra, meta_fn, use_meta, k, mode,
                forced: list[int] | None = None) -> list[Hit]:
        cands: set[int] = set(forced or [])
        sem_on = qv is not None and mode in ("hybrid", "semantic")
        bm_on = mode in ("hybrid", "bm25")
        if sem_on:
            cands.update(self._semantic_candidates(name, c, qv, mask, where, extra))
        raw = np.zeros(len(c.ids))
        tokens = tokenize(query)
        if bm_on and tokens:
            raw = np.asarray(c.bm25.get_scores(tokens), dtype=float)
            raw[~mask] = 0.0
            top = np.argsort(-raw)[:config.CANDIDATES]
            cands.update(int(i) for i in top if raw[i] > 0)
        if not cands:
            return []
        denom = max(float(raw.max()) if raw.size else 0.0, config.BM25_FLOOR)

        if mode == "semantic":
            w = (1.0, 0.0, 0.0)
        elif mode == "bm25":
            w = (0.0, 1.0, 0.0)
        elif not sem_on:                       # semantic unavailable -> keywords + meta
            w = (0.0, config.W_BM25 + config.W_SEM, config.W_META) if use_meta else (0.0, 1.0, 0.0)
        elif use_meta:
            w = (config.W_SEM, config.W_BM25, config.W_META)
        else:
            s = config.W_SEM + config.W_BM25
            w = (config.W_SEM / s, config.W_BM25 / s, 0.0)

        scored = []
        for i in cands:
            cos = float(max(0.0, c.vec[i] @ qv)) if qv is not None else 0.0
            bm = float(raw[i] / denom) if bm_on else 0.0
            meta = meta_fn(i)
            score = w[0] * cos + w[1] * bm + w[2] * meta
            bd = {"semantic": round(cos, 4), "bm25": round(bm, 4), "meta": round(meta, 4)}
            if bm >= 0.6 and cos < 0.45 and sem_on:
                bd["keyword_only"] = 1.0         # exact-text guard: never a duplicate on keywords alone
            scored.append((score, i, bd))
        scored.sort(key=lambda t: -t[0])

        if self._reranker is not None and mode == "hybrid" and scored:
            top = scored[: max(k * 2, 10)]
            pairs = [(query, c.docs[i]) for _, i, _ in top]
            logits = self._reranker.predict(pairs)
            rer = []
            for (s, i, bd), lg in zip(top, logits):
                r = 1 / (1 + math.exp(-float(lg)))
                bd["rerank"] = round(r, 4)
                rer.append((0.5 * s + 0.5 * r, i, bd))
            scored = sorted(rer, key=lambda t: -t[0]) + scored[len(top):]

        hits = []
        for score, i, bd in scored[:k]:
            m = c.metas[i]
            snippet = c.docs[i] if name == "tickets" else (self.articles[c.ids[i]].symptoms[0] if c.ids[i] in self.articles else c.docs[i])
            hits.append(Hit(
                source_id=c.ids[i], source_type="ticket" if name == "tickets" else "kb",
                score=round(min(1.0, max(0.0, score)), 4), score_breakdown=bd,
                snippet=snippet[:300], title=c.titles[i],
                metadata={k_: v for k_, v in m.items() if k_ != "created_ts" and v not in ("", None)},
            ))
        return hits

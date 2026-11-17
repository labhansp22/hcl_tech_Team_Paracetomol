"""Index build (owner A): dataset -> redact -> split -> baseline -> embeddings -> ChromaDB + BM25.

Run:  python -m app.build_index [--dataset path.csv] [--backend chroma|numpy] [--embedder hashing-384]
Safe to re-run: collections are dropped and rebuilt.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from datetime import datetime

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.model_selection import train_test_split

from app import config, taxonomy
from app.baseline import Baseline
from app.data.loader import load_kb, load_tickets
from app.embeddings import Embedder
from app.redact import redact
from app.retriever import kb_full_text, kb_index_text, tokenize
from app.schema import Ticket


def ticket_doc(t: Ticket) -> str:
    return f"{t.title}. {t.text}" if t.title and not t.text.startswith(t.title) else t.text


def _meta(t: Ticket) -> dict:
    return {
        "category": t.category or "", "subcategory": t.subcategory or "", "status": t.status or "",
        "application": t.application or "", "environment": t.environment or "",
        "assignment_group": t.assignment_group or "", "priority": t.priority or "",
        "created_ts": int(t.created_at.timestamp()) if t.created_at else 0,
        "created_at": t.created_at.isoformat(timespec="minutes") if t.created_at else "",
    }


def build(dataset: str | None = None, backend: str | None = None, embed_model: str | None = None,
          test_frac: float = 0.2) -> dict:
    t0 = time.time()
    backend = backend or config.RETRIEVER_BACKEND
    tickets = load_tickets(dataset)
    for t in tickets:                                # redact BEFORE anything is stored or embedded
        t.text = redact(t.text)[0]
        t.title = redact(t.title)[0] if t.title else t.title
        t.resolution = redact(t.resolution)[0] if t.resolution else t.resolution
    kb = load_kb()

    labelled = [t for t in tickets if t.subcategory]
    tax = (taxonomy.build_from_labels([(t.category or t.subcategory, t.subcategory) for t in labelled])
           if labelled else taxonomy.DEFAULT_TAXONOMY)
    taxonomy.save(tax)
    sub2cat = taxonomy.sub_to_cat(tax)

    # stratified split: test tickets are "new incoming" tickets, never indexed
    counts = Counter(t.subcategory for t in labelled)
    strat_ok = labelled and min(counts.values()) >= 2
    train, test = train_test_split(tickets, test_size=test_frac, random_state=config.SEED,
                                   stratify=[t.subcategory for t in tickets] if strat_ok and len(labelled) == len(tickets) else None)
    config.SPLITS_JSON.write_text(json.dumps({
        "train_ids": [t.id for t in train],
        "test": [json.loads(t.model_dump_json()) for t in test],
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    if labelled:
        tr = [t for t in train if t.subcategory]
        Baseline.train([ticket_doc(t) for t in tr], [t.subcategory for t in tr], sub2cat).save()

    emb = Embedder(embed_model or config.EMBED_MODEL)
    docs = [ticket_doc(t) for t in train]
    full = [d + (f"\nResolution: {t.resolution}" if t.resolution else "") for d, t in zip(docs, train)]
    metas = [_meta(t) for t in train]
    kb_docs = [kb_index_text(a) for a in kb]
    kb_metas = [{"category": a.category, "subcategory": a.subcategory, "assignment_group": a.assignment_group} for a in kb]
    t_vec = emb.encode(docs)
    k_vec = emb.encode(kb_docs)

    with open(config.BM25_PKL, "wb") as f:
        pickle.dump({
            "tickets": {"ids": [t.id for t in train], "docs": docs, "full": full, "metas": metas,
                        "titles": [t.title for t in train], "bm25": BM25Okapi([tokenize(d) for d in docs])},
            "kb": {"ids": [a.kb_id for a in kb], "docs": kb_docs, "full": [kb_full_text(a) for a in kb],
                   "metas": kb_metas, "titles": [a.title for a in kb],
                   "bm25": BM25Okapi([tokenize(d + " " + " ".join(a.resolution_steps)) for d, a in zip(kb_docs, kb)])},
            "articles": [a.model_dump() for a in kb],
        }, f)
    np.savez(config.VECTORS_NPZ, tickets=t_vec, kb=k_vec)

    if backend == "chroma":
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
        for name, ids, vecs, ds, ms in [
            (config.TICKETS_COLLECTION, [t.id for t in train], t_vec, docs, metas),
            (config.KB_COLLECTION, [a.kb_id for a in kb], k_vec, kb_docs, kb_metas),
        ]:
            try:
                client.delete_collection(name)
            except Exception:
                pass
            col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
            for s in range(0, len(ids), 1000):
                col.add(ids=ids[s:s + 1000], embeddings=vecs[s:s + 1000].tolist(),
                        documents=ds[s:s + 1000], metadatas=ms[s:s + 1000])

    info = {"embedder": emb.name, "backend": backend, "tickets_indexed": len(train), "test_tickets": len(test),
            "kb_articles": len(kb), "dataset": dataset or config.DATASET_PATH or "synthetic",
            "built_at": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0, 1)}
    config.INDEX_META.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--backend", default=None, choices=["chroma", "numpy"])
    ap.add_argument("--embedder", default=None, help="model name, or hashing-384 for the offline fallback")
    a = ap.parse_args()
    print(json.dumps(build(a.dataset, a.backend, a.embedder), indent=2))

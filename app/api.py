"""C8 API (owner C): FastAPI over the pipeline.

Run:  uvicorn app.api:app --port 8000      Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from app import llm_provider
from app.db import AuditDB
from app.redact import redact
from app.schema import AnalysisResult, FeedbackIn, SearchFilters, TicketIn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.pipeline import Pipeline
    STATE["pipeline"] = Pipeline()          # loads embeddings, ChromaDB, BM25, baseline once
    STATE["db"] = AuditDB()
    yield


app = FastAPI(title="IT Ticket Analysis & Resolution Assistant", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    p = STATE["pipeline"]
    return {"status": "ok", "index": p.retriever.stats(), "llm_chain": llm_provider.available(),
            "llm_pool": llm_provider.pool_status(), "baseline": p.baseline is not None}


@app.get("/taxonomy")
def get_taxonomy() -> dict:
    return STATE["pipeline"].tax


@app.post("/analyze", response_model=AnalysisResult)
def analyze(ticket: TicketIn, use_llm: bool = True) -> AnalysisResult:
    result = STATE["pipeline"].analyze(ticket, use_llm=use_llm)
    STATE["db"].save_analysis(ticket, result)
    return result


@app.get("/search")
def search(q: str = Query(min_length=2), k: int = Query(8, ge=1, le=50),
           category: Optional[str] = None, subcategory: Optional[str] = None, application: Optional[str] = None,
           environment: Optional[str] = None, status: Optional[str] = None, assignment_group: Optional[str] = None,
           date_from: Optional[date] = None, date_to: Optional[date] = None,
           mode: Literal["hybrid", "semantic", "bm25"] = "hybrid") -> dict:
    r = STATE["pipeline"].retriever
    clean, _ = redact(q)
    f = SearchFilters(category=category, subcategory=subcategory, application=application, environment=environment,
                      status=status, assignment_group=assignment_group, date_from=date_from, date_to=date_to)
    return {"query": clean, "tickets": r.search_tickets(clean, f, k=k, mode=mode),
            "kb": r.search_kb(clean, k=min(k, 5), mode=mode)}


@app.post("/feedback")
def feedback(f: FeedbackIn) -> dict:
    db: AuditDB = STATE["db"]
    if not db.exists(f.analysis_id):
        raise HTTPException(404, f"unknown analysis_id {f.analysis_id}")
    return {"id": db.save_feedback(f), "status": "recorded"}


@app.get("/analyses")
def analyses(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
    return STATE["db"].recent(limit)


@app.get("/stats")
def stats() -> dict:
    return STATE["db"].stats()


@app.get("/kb/{kb_id}")
def kb_article(kb_id: str) -> dict:
    a = STATE["pipeline"].retriever.articles.get(kb_id)
    if not a:
        raise HTTPException(404, "unknown KB id")
    return a.model_dump()


@app.get("/audit/export")
def export(format: Literal["csv", "json"] = "csv"):
    db: AuditDB = STATE["db"]
    if format == "json":
        return JSONResponse(db.export_rows())
    return PlainTextResponse(db.export_csv(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=audit_export.csv"})

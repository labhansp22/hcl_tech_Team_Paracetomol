"""SHARED (frozen contract, ARCHITECTURE.md Section 5)."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Priority = Literal["P1", "P2", "P3", "P4"]
Status = Literal["OK", "NEEDS_REVIEW", "ESCALATE"]
SourceType = Literal["ticket", "kb"]


# ---- 5.1 core entities ---------------------------------------------------
class Ticket(BaseModel):
    id: str
    title: Optional[str] = None
    text: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    priority: Optional[Priority] = None
    status: Optional[Literal["open", "resolved", "closed"]] = None
    application: Optional[str] = None
    environment: Optional[Literal["prod", "uat", "dev"]] = None
    assignment_group: Optional[str] = None
    created_at: Optional[datetime] = None
    resolution: Optional[str] = None
    synthetic_fields: list[str] = []


class KBArticle(BaseModel):
    kb_id: str
    title: str
    category: str
    subcategory: str
    symptoms: list[str]
    diagnostic_steps: list[str]
    resolution_steps: list[str]
    escalation_criteria: str
    assignment_group: str


# ---- 5.2 API I/O ---------------------------------------------------------
class SearchFilters(BaseModel):
    category: Optional[str] = None
    subcategory: Optional[str] = None
    application: Optional[str] = None
    environment: Optional[str] = None
    status: Optional[str] = None
    assignment_group: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None


class TicketIn(BaseModel):
    title: Optional[str] = None
    description: str = Field(min_length=3, max_length=8000)
    application: Optional[str] = None
    environment: Optional[str] = None
    filters: SearchFilters = SearchFilters()


class Hit(BaseModel):
    source_id: str
    source_type: SourceType
    score: float
    score_breakdown: dict[str, float]
    snippet: str
    is_duplicate: bool = False
    title: Optional[str] = None
    metadata: dict[str, Any] = {}


class Evidence(BaseModel):
    source_id: str
    source_type: SourceType
    excerpt: str
    reason: str


class ResolutionPath(BaseModel):
    rank: int
    title: str
    steps: list[str]
    evidence: list[Evidence]


class Classification(BaseModel):
    category: str
    subcategory: str
    priority: Priority
    confidence: float
    secondary: Optional[tuple[str, str, float]] = None
    rationale: str
    source: Literal["llm", "baseline"]


class EscalationInfo(BaseModel):
    assignment_group: str
    clarifying_questions: list[str]
    diagnostic_bundle: dict[str, Any]


class AnalysisResult(BaseModel):
    analysis_id: str
    status: Status
    confidence: float
    confidence_reasons: list[str]
    classification: Classification
    similar_tickets: list[Hit]
    kb_matches: list[Hit]
    resolution_paths: list[ResolutionPath]
    escalation: Optional[EscalationInfo] = None
    redactions: list[str]
    dropped_claims: int
    degraded_mode: bool
    timings_ms: dict[str, int]
    # additions (do not break the frozen fields above)
    provider: Optional[str] = None
    model: Optional[str] = None
    kb_gap: bool = False
    signals: dict[str, float] = {}


class FeedbackIn(BaseModel):
    analysis_id: str
    item_type: Literal["classification", "similar_ticket", "resolution_path"]
    item_id: str
    verdict: Literal["useful", "incorrect", "unresolved"]
    corrected_label: Optional[str] = None
    note: Optional[str] = Field(default=None, max_length=2000)


# ---- baseline ------------------------------------------------------------
class BaselinePrediction(BaseModel):
    category: str
    subcategory: str
    probability: float
    top_subcategories: list[tuple[str, float]]   # best first
    top_categories: list[tuple[str, float]]
    top_terms: list[str]


# ---- 5.3 LLM output (enforced by Pydantic, not by the provider) ----------
class LLMEvidence(BaseModel):
    source_id: str
    excerpt: str
    reason: str = ""


class LLMPath(BaseModel):
    title: str
    steps: list[str]
    evidence: list[LLMEvidence] = []


class LLMSecondary(BaseModel):
    category: str
    subcategory: str
    confidence: float = 0.0


class LLMOutput(BaseModel):
    category: str
    subcategory: str
    priority: Priority
    confidence: float
    secondary: Optional[LLMSecondary] = None
    rationale: str = ""
    duplicate_ids: list[str] = []
    resolution_paths: list[LLMPath] = []
    insufficient_evidence: bool = False
    clarifying_questions: list[str] = []

    @field_validator("priority", mode="before")
    @classmethod
    def _prio(cls, v: Any) -> Any:
        return str(v).strip().upper() if v is not None else v

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        v = float(v)
        if v > 1:          # some models answer in percent
            v = v / 100
        return max(0.0, min(1.0, v))

    @field_validator("secondary", mode="before")
    @classmethod
    def _sec(cls, v: Any) -> Any:
        if not v or (isinstance(v, dict) and not v.get("category")):
            return None
        return v

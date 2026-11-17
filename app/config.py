"""SHARED: paths, weights, thresholds and model ids. Every value can be overridden in .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ---- paths ---------------------------------------------------------------
DATA_DIR = ROOT / "data"
TICKETS_JSON = DATA_DIR / "tickets.json"          # synthetic dataset (generated)
KB_JSON = DATA_DIR / "kb" / "articles.json"       # curated KB
CHROMA_DIR = DATA_DIR / "chroma"
BM25_PKL = DATA_DIR / "bm25.pkl"                  # docs, metadata, BM25 indexes
VECTORS_NPZ = DATA_DIR / "vectors.npz"            # normalised embeddings (numpy backend + scoring)
INDEX_META = DATA_DIR / "index_meta.json"
BASELINE_PKL = DATA_DIR / "baseline.pkl"
TAXONOMY_JSON = DATA_DIR / "taxonomy.json"
SPLITS_JSON = DATA_DIR / "splits.json"
CACHE_DIR = DATA_DIR / "cache" / "llm"
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "audit.db"))
DATASET_PATH = os.getenv("DATASET_PATH", "")      # CSV/XLSX/JSON; empty = synthetic

# ---- retrieval -----------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
RETRIEVER_BACKEND = os.getenv("RETRIEVER_BACKEND", "chroma")   # chroma | numpy
TICKETS_COLLECTION = "tickets"
KB_COLLECTION = "kb_articles"
CANDIDATES = int(os.getenv("CANDIDATES", 30))     # top-N from each of semantic and BM25
W_SEM = _f("W_SEM", 0.6)
W_BM25 = _f("W_BM25", 0.3)
W_META = _f("W_META", 0.1)
BM25_FLOOR = _f("BM25_FLOOR", 8.0)                # normalise by max(best score, floor)
DUP_COS = _f("DUP_COS", 0.85)
RERANK = _b("RERANK", False)
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
K_TICKETS = int(os.getenv("K_TICKETS", 8))
K_KB = int(os.getenv("K_KB", 4))

# ---- confidence gate (Section 7) ----------------------------------------
C_LLM = _f("C_LLM", 0.35)
C_RET = _f("C_RET", 0.40)
C_AGR = _f("C_AGR", 0.25)
T_OK = _f("T_OK", 0.70)
T_ESC = _f("T_ESC", 0.45)
KB_GAP = _f("KB_GAP", 0.30)
T_RET_OK = _f("T_RET_OK", 0.60)                 # OK needs at least this much retrieval evidence

# ---- LLM -----------------------------------------------------------------
LLM_CHAIN = [p.strip() for p in os.getenv("LLM_CHAIN", "groq,gemini").split(",") if p.strip()]
LLM_TIMEOUT = _f("LLM_TIMEOUT", 25)
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1000))
LLM_MAX_WAIT = _f("LLM_MAX_WAIT", 15)               # seconds we will wait once on a 429/503 before failing over
LLM_TICKETS = int(os.getenv("LLM_TICKETS", 5))      # how many similar tickets go into the prompt
LLM_KB = int(os.getenv("LLM_KB", 3))                # how many KB articles go into the prompt
PROMPT_VERSION = "v1"

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
SEED = 42

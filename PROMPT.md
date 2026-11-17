# Context for a new Claude (or other AI assistant) session

> **How to use:** open a Claude Code chat in the project folder on your own laptop and send:
> *"Read PROMPT.md and docs/PROJECT_CONTEXT.md, then confirm you understand the project before doing anything. I am Role N."*
> Everything below was decided or built in earlier chats (2026-11-10 to 2026-11-17).

## Who we are and how to help

- A **4-person team** at the **HCLTech "Future Ready AI Engineer" hackathon** (6 hours: 1 h architecture + 5 h build/test).
- The brief on the day is **UC1, Intelligent IT Ticket Analysis & Resolution Assistant** (`problem.docx`), and the app for it is built.
- Each person owns one role (see the table below) and works on their own laptop and their own git branch.
- Some of us are **new to RAG**: explain simply, step by step, with everyday analogies, and name the file and line.
- The full onboarding is in **`docs/PROJECT_CONTEXT.md`**: functionality, then architecture, then code level, then setup.

| Role | Area | Owns |
|---|---|---|
| 1 | Data, classifier, evaluation | `data/`, `app/data/loader.py`, `app/taxonomy.py`, `app/baseline.py`, `app/build_index.py`, `eval/` |
| 2 | Retrieval and privacy | `app/redact.py`, `app/embeddings.py`, `app/retriever.py` |
| 3 | LLM and trust layer | `app/llm_provider.py`, `app/llm.py`, `app/validator.py`, `app/confidence.py`, `app/pipeline.py` |
| 4 | API, UI, audit, docs | `app/api.py`, `app/db.py`, `ui/app.py`, `scripts/`, `README.md` |
| shared | contract and settings | `app/schema.py`, `app/config.py`, `tests/`: change only after telling the team |

## Hard constraints (don't propose otherwise)

- **No paid API keys.** LLM = free keys only: **Groq → Gemini**, via the `openai` Python client against their OpenAI-compatible endpoints.
- **No local/offline LLM** (no Ollama): the laptops have about 8 GB RAM.
- Stack: **Streamlit UI + FastAPI**, LLM-centric **RAG**, **ChromaDB + BM25** hybrid search, SQLite audit log, synthetic data.
- Python 3.10 to 3.12, one virtual environment per laptop. Code must stay Python 3.10 compatible (e.g. no nested same-quote f-strings).

## What is built (working, tested)

- Data: 1,800 synthetic tickets, 26 subcategories in 8 categories (`data/generate.py`, seed 42); 36 hand-written KB articles (`data/kb/articles.json`).
- Pipeline (`app/pipeline.py`): redact → baseline classifier → hybrid search → LLM (JSON) → validator → confidence gate → result.
- API + UI + SQLite audit with feedback and CSV/JSON export; 6 demo tickets in `app/demo.py`.
- 33 tests pass with no network (`python -m pytest -q`).
- Submission docs: `README.md`, `EVALUATION.md`, `docs/WORKED_EXAMPLES.md`.

Key design (details in `docs/PROJECT_CONTEXT.md` Part 2 and `docs/ARCHITECTURE.md`):
- Search score `0.6·cosine + 0.3·bm25_norm + 0.1·meta`; BM25 normalised by `max(best, 8)`; duplicate = cosine ≥ 0.85 + same category.
- Embeddings `paraphrase-multilingual-MiniLM-L12-v2` (384-d), loaded offline-first. Cross-encoder reranker built in but off (measured: no gain, 10× slower).
- LLM: JSON mode, temperature 0, Pydantic `LLMOutput`, one repair retry, disk cache `data/cache/llm/`; prompt sends top 3 KB + top 5 tickets.
- Validator: cited IDs must be retrieved; quotes must be verbatim (ignoring case, spacing, curly quotes, Unicode hyphens); labels from the taxonomy.
- Confidence `0.35·LLM + 0.40·retrieval + 0.25·agreement`; ESCALATE < 0.45, no valid evidence, insufficient evidence, or top KB < 0.30;
  NEEDS_REVIEW 0.45–0.70, retrieval < 0.60 (weak evidence), P1, dropped claims, or degraded mode; else OK.
- Degraded mode (no LLM): baseline label + top KB steps verbatim, never OK.

## LLM setup

- `.env` (git-ignored) holds each person's own `GROQ_API_KEY`, `GEMINI_API_KEY`, `GROQ_MODEL=openai/gpt-oss-120b`, `GEMINI_MODEL=gemini-3.5-flash`.
- Groq free tier ≈ 8,000 tokens/min and 200,000 tokens/day per model per account. Several keys (comma-separated, one per teammate's account) are
  balanced round robin with failover and cooldown (`app/llm_provider.py`); several models per provider are tried in order.
- If a model returns 404 it was retired: set a current model name in `.env`.

## Still to do (ask before starting any of these)

1. **Git:** create the GitHub repo and branches `feat/data-eval`, `feat/retrieval`, `feat/llm-trust`, `feat/api-ui`; each person commits their own files under their own identity.
2. After any data change: `python -m app.build_index`, `python -m eval.run_eval --llm 30`, `python scripts/worked_examples.py`, `python scripts/prewarm_cache.py`.
3. Practise the demo (`learning/DEMO_GUIDE.md`) and the judge questions (`learning/pages/1-rag-study-guide-170-questions.html`).

## Reading material (all offline, in this folder)

`docs/PROJECT_CONTEXT.md` (start here) · `learning/README.md` · `learning/pages/*.html` (RAG guide with 170 Q&A, code walkthrough,
architecture overview, HLD prep with 71 Q&A) · `learning/DEMO_GUIDE.md` · `docs/ARCHITECTURE.md` · `docs/PLAYBOOK.md` (other briefs) ·
`docs/SETUP.md` · `EVALUATION.md` · `README.md`.

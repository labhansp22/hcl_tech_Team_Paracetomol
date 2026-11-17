"""Run every demo ticket through the LLM once so the demo works instantly (and offline) from the cache.

Run:  python scripts/prewarm_cache.py     (needs GROQ_API_KEY or GEMINI_API_KEY in .env)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.demo import DEMO_TICKETS  # noqa: E402
from app.llm_provider import available  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.schema import TicketIn  # noqa: E402

if not available():
    sys.exit("No LLM key in .env: add GROQ_API_KEY or GEMINI_API_KEY first.")
p = Pipeline()
for d in DEMO_TICKETS:
    t = TicketIn(**{k: v for k, v in d.items() if k != "label"})
    r = p.analyze(t)
    print(f"{d['label']:<34} {r.status:<13} {r.confidence:.2f}  {r.classification.subcategory:<20} "
          f"{r.provider or 'DEGRADED'}  dropped={r.dropped_claims}")
    time.sleep(1.5)

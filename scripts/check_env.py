"""Pre-hackathon environment check. Run: python scripts/check_env.py

Verifies packages import, embedding models are cached, and the Groq and
Gemini API keys in .env work. Prints a PASS/FAIL table.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so "import app" works

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        RESULTS.append((name, True, f"{detail} ({time.perf_counter() - t0:.1f}s)"))
    except Exception as exc:  # noqa: BLE001 - we want every failure reported
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"[:160]))


def python_version() -> str:
    v = sys.version_info
    if not (v.major == 3 and v.minor in (10, 11, 12)):
        raise RuntimeError(f"Python {v.major}.{v.minor} found; use 3.10-3.12")
    return f"{v.major}.{v.minor}.{v.micro}"


PACKAGES = [
    "fastapi", "uvicorn", "streamlit", "plotly", "pydantic", "dotenv", "httpx",
    "openai", "chromadb", "sentence_transformers", "rank_bm25", "pypdf", "bs4",
    "markdown", "openpyxl", "docx", "pandas", "numpy", "sklearn", "lightgbm",
    "shap", "statsmodels", "faker", "pytest",
]

MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def import_pkg(mod: str):
    return lambda: getattr(importlib.import_module(mod), "__version__", "ok")


def embed_model(name: str):
    def _run() -> str:
        from sentence_transformers import SentenceTransformer

        vec = SentenceTransformer(name).encode(["VPN keeps disconnecting"])
        return f"dim={vec.shape[1]}"

    return _run


def cross_encoder() -> str:
    from sentence_transformers import CrossEncoder

    score = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict([("vpn", "vpn drops")])
    return f"score={float(score[0]):.2f}"


def chroma_roundtrip() -> str:
    import tempfile

    import chromadb

    # ignore_cleanup_errors: on Windows Chroma keeps chroma.sqlite3 open until exit
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        client = chromadb.PersistentClient(path=d)
        col = client.create_collection("smoke_test",metadata={"hnsw:space": "cosine"})
        col.add(ids=["a"], embeddings=[[0.1, 0.2, 0.3]], metadatas=[{"k": "v"}])
        res = col.query(query_embeddings=[[0.1, 0.2, 0.3]], n_results=1, where={"k": "v"})
        return f"hit={res['ids'][0][0]}"


def llm(base_url: str, key: str, model_env: str, default_model: str):
    def _run() -> str:
        from openai import OpenAI

        if not key:
            raise RuntimeError("API key not set in .env")
        model = os.getenv(model_env, default_model).split(",")[0].strip()
        client = OpenAI(base_url=base_url, api_key=key, timeout=30)
        out = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 'Reply with JSON {"ok": true}'}],
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        return f"{model}: {(out.choices[0].message.content or '').strip()[:30]}"

    return _run


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    check("python", python_version)
    for p in PACKAGES:
        check(f"import {p}", import_pkg(p))
    check("chromadb roundtrip", chroma_roundtrip)
    for m in MODELS:
        check(f"model {m.split('/')[-1]}", embed_model(m))
    check("model ms-marco cross-encoder", cross_encoder)
    # every key separately (a pool of keys is balanced round robin by app/llm_provider.py)
    from app.llm_provider import PROVIDERS, keys_for
    for prov in PROVIDERS:
        keys = keys_for(prov) or [""]
        for i, key in enumerate(keys):
            name = f"LLM {prov}" + (f" key {i + 1}/{len(keys)}" if len(keys) > 1 else "")
            check(name, llm(PROVIDERS[prov]["base_url"], key, PROVIDERS[prov]["model_env"], PROVIDERS[prov]["default"]))
    width = max(len(n) for n, _, _ in RESULTS)
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

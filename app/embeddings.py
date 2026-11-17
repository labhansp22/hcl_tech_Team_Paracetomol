"""Embedding model wrapper (owner B).

Primary: sentence-transformers (multilingual MiniLM, 384 dims, cosine).
Fallback: a stateless hashing vectoriser (also 384 dims) if the model cannot be loaded, e.g. no network
on first run. The index records which embedder built it, so queries always use the same one.
"""
from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger(__name__)
HASHING = "hashing-384"


def _normalise(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


class Embedder:
    def __init__(self, model_name: str):
        self.name = model_name
        self._st = None
        self._hv = None
        if model_name == HASHING:
            self._init_hashing()
            return
        try:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from sentence_transformers import SentenceTransformer
            try:      # cached copy first: fast and works without network
                self._st = SentenceTransformer(model_name, device="cpu", local_files_only=True)
            except Exception:
                self._st = SentenceTransformer(model_name, device="cpu")
        except Exception as e:  # pragma: no cover - depends on environment
            log.warning("Embedding model %s failed to load (%s); using hashing fallback", model_name, e)
            self.name = HASHING
            self._init_hashing()

    def _init_hashing(self) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer
        self._hv = HashingVectorizer(n_features=384, alternate_sign=False, ngram_range=(1, 2), norm=None)

    @property
    def is_fallback(self) -> bool:
        return self._st is None

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if self._st is not None:
            v = self._st.encode(texts, batch_size=batch_size, show_progress_bar=len(texts) > 200,
                                convert_to_numpy=True, normalize_embeddings=True)
            return v.astype(np.float32)
        return _normalise(self._hv.transform(texts).toarray())

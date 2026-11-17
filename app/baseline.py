"""C2 Baseline classifier (owner A): TF-IDF (word + char n-grams) + LogisticRegression.

Used as the fallback classification, the agreement signal for confidence, and the comparison in the evaluation.
"""
from __future__ import annotations

import pickle
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from app import config
from app.schema import BaselinePrediction


class Baseline:
    def __init__(self, model: Pipeline, sub2cat: dict[str, str]):
        self.model = model
        self.sub2cat = sub2cat

    @classmethod
    def train(cls, texts: list[str], subcategories: list[str], sub2cat: dict[str, str]) -> "Baseline":
        feats = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
        ])
        model = Pipeline([("tfidf", feats), ("clf", LogisticRegression(max_iter=3000, C=5.0))])
        model.fit(texts, subcategories)
        return cls(model, sub2cat)

    def predict(self, text: str) -> BaselinePrediction:
        probs = self.model.predict_proba([text])[0]
        classes = self.model.classes_
        order = np.argsort(probs)[::-1]
        top_subs = [(str(classes[i]), float(probs[i])) for i in order[:3]]
        cat_p: dict[str, float] = defaultdict(float)
        for c, p in zip(classes, probs):
            cat_p[self.sub2cat.get(str(c), str(c))] += float(p)
        top_cats = sorted(cat_p.items(), key=lambda x: -x[1])[:3]
        sub, p = top_subs[0]
        return BaselinePrediction(
            category=self.sub2cat.get(sub, sub), subcategory=sub, probability=p,
            top_subcategories=top_subs, top_categories=top_cats, top_terms=self._top_terms(text, order[0]),
        )

    def _top_terms(self, text: str, cls_idx: int, n: int = 5) -> list[str]:
        """Words in this ticket that pushed most towards the predicted class (explainability)."""
        word_vec: TfidfVectorizer = self.model.named_steps["tfidf"].transformer_list[0][1]
        clf: LogisticRegression = self.model.named_steps["clf"]
        x = word_vec.transform([text])
        n_word = len(word_vec.vocabulary_)
        coefs = clf.coef_[cls_idx][:n_word] if clf.coef_.shape[0] > 1 else clf.coef_[0][:n_word]
        contrib = x.multiply(coefs).tocoo()
        names = word_vec.get_feature_names_out()
        pairs = sorted(zip(contrib.col, contrib.data), key=lambda t: -t[1])
        return [str(names[c]) for c, v in pairs[:n] if v > 0]

    def save(self, path=config.BASELINE_PKL) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "sub2cat": self.sub2cat}, f)

    @classmethod
    def load(cls, path=config.BASELINE_PKL) -> "Baseline":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["model"], d["sub2cat"])

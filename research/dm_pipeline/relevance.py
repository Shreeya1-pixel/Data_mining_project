"""Title-relevance filter (heuristic + optional supervised head)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

_GEO_KW = re.compile(
    r"\b(war|conflict|sanction|tariff|missile|troop|invasion|ceasefire|"
    r"oil|opec|embargo|blockade|cyber.?attack|election|coup|protest|"
    r"central bank|interest rate|trade dispute|geopolitic|nato|ukraine|"
    r"gaza|israel|iran|china|taiwan|pakistan|border)\b",
    re.I,
)
_OFFTOPIC = re.compile(r"\b(box office|cricket|bollywood|recipe|horoscope|ipl)\b", re.I)


def heuristic_relevant(title: str) -> int:
    t = title or ""
    if _OFFTOPIC.search(t):
        return 0
    return 1 if _GEO_KW.search(t) else 0


def soft_label(df: pd.DataFrame, title_col: str = "title") -> pd.Series:
    return df[title_col].map(heuristic_relevant)


def build_relevance_model(texts: list[str], labels: list[int]) -> Pipeline:
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe.fit(texts, labels)
    return pipe


def evaluate_relevance(texts: list[str], labels: list[int], seed: int = 0) -> dict:
    y = np.asarray(labels)
    if len(set(y.tolist())) < 2 or len(y) < 20:
        return {"ok": False, "reason": "need ≥20 labels with both classes"}
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = build_relevance_model(X_train, y_train.tolist())
    pred = model.predict(X_test)
    return {
        "ok": True,
        "f1": float(f1_score(y_test, pred)),
        "report": classification_report(y_test, pred, output_dict=True),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def apply_relevance(df: pd.DataFrame, model: Pipeline | None = None, title_col: str = "title") -> pd.DataFrame:
    out = df.copy()
    if model is None:
        out["relevant"] = soft_label(out, title_col)
    else:
        out["relevant"] = model.predict(out[title_col].astype(str).tolist())
    return out.loc[out["relevant"] == 1].copy()

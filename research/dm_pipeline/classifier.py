"""GeoRule CBA-style classifier vs DT / kNN / NB / SVM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

from research.dm_pipeline.mining import Rule


@dataclass
class PredResult:
    name: str
    accuracy: float
    macro_f1: float
    precision: float
    recall: float
    roc_auc: float | None
    n_test: int


def _rule_rank_key(r: Rule) -> tuple:
    return (-r.confidence, -r.lift, -r.support)


class GeoRuleClassifier:
    """First matching lagged rule wins (CBA-style); else training majority class."""

    def __init__(self, rules: list[Rule], target_sector: str, default_label: str = "FLAT"):
        self.target_sector = target_sector
        self.default_label = default_label
        # keep rules whose consequent is this sector's UP/DOWN
        self.rules = sorted(
            [
                r
                for r in rules
                if any(c in (f"{target_sector}_UP", f"{target_sector}_DOWN") for c in r.consequents)
            ],
            key=_rule_rank_key,
        )
        self.majority_ = default_label

    def fit_majority(self, y: list[str]) -> None:
        if y:
            vals, counts = np.unique(y, return_counts=True)
            self.majority_ = str(vals[np.argmax(counts)])

    def predict_one(self, items: set[str]) -> str:
        for r in self.rules:
            if r.antecedents.issubset(items):
                for c in r.consequents:
                    if c == f"{self.target_sector}_UP":
                        return "UP"
                    if c == f"{self.target_sector}_DOWN":
                        return "DOWN"
        return self.majority_

    def predict(self, baskets: list[set[str]]) -> list[str]:
        return [self.predict_one(b) for b in baskets]

    def predict_proba_scores(self, baskets: list[set[str]], classes: list[str]) -> np.ndarray:
        """Confidence of top matching rule per class; prior if none."""
        prior = 1.0 / max(len(classes), 1)
        scores = np.full((len(baskets), len(classes)), prior)
        class_idx = {c: i for i, c in enumerate(classes)}
        for row, items in enumerate(baskets):
            matched = {c: 0.0 for c in classes}
            for r in self.rules:
                if not r.antecedents.issubset(items):
                    continue
                for c in r.consequents:
                    lab = None
                    if c == f"{self.target_sector}_UP":
                        lab = "UP"
                    elif c == f"{self.target_sector}_DOWN":
                        lab = "DOWN"
                    if lab in matched:
                        matched[lab] = max(matched[lab], r.confidence)
            if any(v > 0 for v in matched.values()):
                for lab, v in matched.items():
                    scores[row, class_idx[lab]] = v if v > 0 else prior * 0.5
                # FLAT gets residual mass if present
                if "FLAT" in class_idx and matched.get("UP", 0) == 0 and matched.get("DOWN", 0) == 0:
                    scores[row, class_idx["FLAT"]] = 1.0
        # row-normalize
        scores = scores / scores.sum(axis=1, keepdims=True).clip(min=1e-9)
        return scores


def lagged_feature_matrix(baskets: list[list[str]]) -> tuple[pd.DataFrame, list[str]]:
    cols = sorted({i for b in baskets for i in b if i.endswith("_L1")})
    rows = []
    for b in baskets:
        s = set(b)
        rows.append([1 if c in s else 0 for c in cols])
    return pd.DataFrame(rows, columns=cols), cols


def target_labels(baskets: list[list[str]], sector: str) -> list[str]:
    labs = []
    for b in baskets:
        s = set(b)
        if f"{sector}_UP" in s:
            labs.append("UP")
        elif f"{sector}_DOWN" in s:
            labs.append("DOWN")
        else:
            labs.append("FLAT")
    return labs


def _metrics(y_true: list[str], y_pred: list[str], y_score: np.ndarray | None, classes: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    auc = None
    if y_score is not None and len(set(y_true)) > 1:
        try:
            le = LabelEncoder()
            le.fit(classes)
            y_bin = le.transform(y_true)
            # one-vs-rest macro AUC
            from sklearn.preprocessing import label_binarize

            Y = label_binarize(y_true, classes=classes)
            if Y.shape[1] == 1:
                auc = float(roc_auc_score(y_true, y_score[:, 1] if y_score.shape[1] > 1 else y_score.ravel()))
            else:
                auc = float(roc_auc_score(Y, y_score, average="macro", multi_class="ovr"))
        except Exception:
            auc = None
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(p),
        "recall": float(r),
        "roc_auc": auc,
    }


def compare_classifiers(
    train_baskets: list[list[str]],
    test_baskets: list[list[str]],
    rules: list[Rule],
    sector: str,
) -> pd.DataFrame:
    y_tr = target_labels(train_baskets, sector)
    y_te = target_labels(test_baskets, sector)
    classes = sorted(set(y_tr) | set(y_te))

    geo = GeoRuleClassifier(rules, sector)
    geo.fit_majority(y_tr)
    pred_geo = geo.predict([set(b) for b in test_baskets])
    scores_geo = geo.predict_proba_scores([set(b) for b in test_baskets], classes)

    X_tr, cols = lagged_feature_matrix(train_baskets)
    X_te, _ = lagged_feature_matrix(test_baskets)
    # align columns
    for c in cols:
        if c not in X_te.columns:
            X_te[c] = 0
    X_te = X_te[cols]

    models: dict[str, object] = {
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=0),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": BernoulliNB(),
        "svm": CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=2),
    }

    # Skip SVM calibration when a class has <2 train examples
    from collections import Counter

    if min(Counter(y_tr).values(), default=0) < 2:
        models.pop("svm", None)

    rows = []
    m = _metrics(y_te, pred_geo, scores_geo, classes)
    rows.append({"name": "georule", **m, "n_test": len(y_te)})

    # price-only rules: antecedents that are only *_UP_L1 / *_DOWN_L1
    price_rules = [
        r
        for r in rules
        if all(a.endswith("_UP_L1") or a.endswith("_DOWN_L1") for a in r.antecedents)
    ]
    geo_p = GeoRuleClassifier(price_rules, sector)
    geo_p.fit_majority(y_tr)
    pred_p = geo_p.predict([set(b) for b in test_baskets])
    m = _metrics(y_te, pred_p, None, classes)
    rows.append({"name": "georule_price_only", **m, "n_test": len(y_te)})

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        score = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_te)
            # align to classes
            if hasattr(model, "classes_"):
                score = np.zeros((len(X_te), len(classes)))
                for j, c in enumerate(model.classes_):
                    if c in classes:
                        score[:, classes.index(c)] = proba[:, j]
        m = _metrics(y_te, list(pred), score, classes)
        rows.append({"name": name, **m, "n_test": len(y_te)})

    return pd.DataFrame(rows)

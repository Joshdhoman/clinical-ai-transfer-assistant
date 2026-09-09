"""Fit the decline model and choose an operating threshold — without touching test.

Sequence:
  1. Fit logistic regression on the training split.
  2. Choose the probability threshold that minimises expected weighted error
     cost on the *validation* split.
  3. Only then score the held-out test split, once, at that fixed threshold.

Step 2 on validation rather than test is the correction to the earlier notebook,
which picked and evaluated the threshold on the same partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from . import SEED
from .features import TARGET, build_feature_frame, make_encoder

# Default asymmetric error costs. A missed decline (false negative) means the
# transfer centre gets no early warning; a false alarm (false positive) means a
# coordinator double-checks a request that would have been accepted. The miss is
# the more expensive mistake, so the cost-minimising threshold sits below 0.5.
# These are illustrative planning weights, not measured dollar costs.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0

THRESHOLD_GRID = np.round(np.arange(0.02, 0.985, 0.01), 3)


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("encode", make_encoder()),
            (
                # No class_weight: keep the predicted probabilities close to
                # calibrated so the threshold argument in notebook 05 is the
                # whole story, not stacked on top of a reweighting.
                "logit",
                LogisticRegression(max_iter=2000, random_state=SEED),
            ),
        ]
    )


def _weighted_cost(y_true, y_prob, threshold: float) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE


def cost_curve(y_true, y_prob) -> pd.DataFrame:
    rows = []
    for t in THRESHOLD_GRID:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(t),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "weighted_cost": fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(y_validation, p_validation) -> dict:
    """Lowest expected weighted cost on validation; ties break toward higher recall."""
    curve = cost_curve(y_validation, p_validation)
    best_cost = curve["weighted_cost"].min()
    best = curve[curve["weighted_cost"] == best_cost].sort_values(
        "recall", ascending=False
    ).iloc[0]
    return {
        "threshold": float(best["threshold"]),
        "validation_weighted_cost": float(best_cost),
        "validation_precision": float(best["precision"]),
        "validation_recall": float(best["recall"]),
        "cost_false_negative": COST_FALSE_NEGATIVE,
        "cost_false_positive": COST_FALSE_POSITIVE,
        "default_threshold_cost": float(_weighted_cost(y_validation, p_validation, 0.5)),
    }


def evaluate_at(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "weighted_cost": float(fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE),
    }


@dataclass
class TrainedModel:
    pipeline: Pipeline
    threshold: float
    selection: dict
    metrics: dict = field(default_factory=dict)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(build_feature_frame(df))[:, 1]

    def flag(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self.threshold).astype(int)


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """Signed logistic-regression coefficients as odds ratios, largest effect first."""
    encoder = pipeline.named_steps["encode"]
    names = encoder.get_feature_names_out()
    coefs = pipeline.named_steps["logit"].coef_.ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    table["odds_ratio"] = np.exp(table["coefficient"])
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).drop(
        columns="abs_coefficient"
    ).reset_index(drop=True)


def train(df: pd.DataFrame) -> TrainedModel:
    """Fit on train, pick threshold on validation, score test once."""
    parts = {name: df[df["split"] == name] for name in ("train", "validation", "test")}
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f"split '{name}' is empty")

    pipeline = make_model()
    pipeline.fit(build_feature_frame(parts["train"]), parts["train"][TARGET])

    p_val = pipeline.predict_proba(build_feature_frame(parts["validation"]))[:, 1]
    selection = select_threshold(parts["validation"][TARGET].to_numpy(), p_val)
    threshold = selection["threshold"]

    metrics = {}
    for name in ("train", "validation", "test"):
        proba = pipeline.predict_proba(build_feature_frame(parts[name]))[:, 1]
        metrics[name] = evaluate_at(parts[name][TARGET].to_numpy(), proba, threshold)
    # Also record test at the naive 0.5 line, to show what the shift bought.
    p_test = pipeline.predict_proba(build_feature_frame(parts["test"]))[:, 1]
    metrics["test_default_threshold"] = evaluate_at(
        parts["test"][TARGET].to_numpy(), p_test, 0.5
    )

    return TrainedModel(pipeline, threshold, selection, metrics)

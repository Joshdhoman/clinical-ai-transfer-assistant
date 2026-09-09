"""Interpretable tree comparison using only note-extracted features."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .schema import Extraction, ROUTES
from .synthetic import SEED

# Requested LOC is intentionally excluded to avoid simply echoing a referral.
# Facility, age, outcome and capacity are excluded from acuity prediction.
CATEGORICAL = ["oxygen_support", "vasopressor_use", "monitoring_required", "specialty_need", "requested_service"]
NUMERIC = ["systolic_bp", "heart_rate", "spo2"]
FEATURES = CATEGORICAL + NUMERIC


def feature_frame(results: list[Extraction]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append({k: (str(result.values[k]) if result.values[k] is not None else "unknown")
                     if k in CATEGORICAL else result.values[k] for k in FEATURES})
    frame = pd.DataFrame(rows, columns=FEATURES)
    frame[NUMERIC] = frame[NUMERIC].astype(float)
    return frame


def make_model(depth: int) -> Pipeline:
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), NUMERIC),
    ])
    return Pipeline([("preprocess", preprocessing),
                     ("tree", DecisionTreeClassifier(max_depth=depth, min_samples_leaf=12,
                                                     class_weight="balanced", random_state=SEED))])


def train_model(x_train, y_train, x_validation, y_validation):
    """Pick depth using validation macro-F1. Test records are never accepted here."""
    candidates = []
    best = None
    for depth in [3, 4, 5, 6]:
        model = make_model(depth).fit(x_train, y_train)
        score = f1_score(y_validation, model.predict(x_validation), labels=ROUTES,
                         average="macro", zero_division=0)
        candidates.append({"max_depth": depth, "validation_macro_f1": float(score)})
        if best is None or score > best[0]:
            best = (score, model, depth)
    # Keep the train-only fitted model; no test or validation refitting.
    return best[1], {"selected_depth": best[2], "candidates": candidates}


def explain_tree(model: Pipeline, result: Extraction) -> dict:
    x = feature_frame([result])
    transformed = model.named_steps["preprocess"].transform(x)
    names = model.named_steps["preprocess"].get_feature_names_out()
    tree = model.named_steps["tree"]
    path = tree.decision_path(transformed).indices
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    steps = []
    for node in path:
        feature = tree.tree_.feature[node]
        if feature < 0:
            continue
        observed = float(dense[0, feature])
        threshold = float(tree.tree_.threshold[node])
        steps.append(f"{names[feature]} = {observed:.2f} {'<=' if observed <= threshold else '>'} {threshold:.2f}")
    return {"route": str(model.predict(x)[0]), "decision_path": steps,
            "leaf_class_weights": dict(zip(tree.classes_, map(float, model.predict_proba(x)[0]))),
            "caution": "Class-weighted leaf proportions are uncalibrated and are not clinical confidence."}

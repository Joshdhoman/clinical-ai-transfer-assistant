"""App-level smoke checks that do not need a running Streamlit server."""

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_committed_artifacts_exist():
    for rel in [
        "artifacts/decline_model.joblib",
        "artifacts/model_selection.json",
        "artifacts/run_manifest.json",
        "reports/metrics.json",
        "reports/coefficients.csv",
        "reports/equity_chi_square.json",
        "reports/equity_payer_logit.json",
        "reports/figures/confusion_test.png",
    ]:
        assert (ROOT / rel).exists(), rel


def test_committed_model_scores_a_request():
    bundle = joblib.load(ROOT / "artifacts" / "decline_model.joblib")
    pipeline, threshold = bundle["pipeline"], bundle["threshold"]
    assert 0 < threshold < 1
    from transfer_decline.data import prepared
    from transfer_decline.features import build_feature_frame

    sample = build_feature_frame(prepared(ROOT).head(5))
    proba = pipeline.predict_proba(sample)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_metrics_json_matches_committed_threshold():
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    bundle = joblib.load(ROOT / "artifacts" / "decline_model.joblib")
    assert metrics["operating_threshold"] == bundle["threshold"]


def test_app_module_imports():
    import ast

    ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))

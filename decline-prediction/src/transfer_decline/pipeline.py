"""End-to-end reproducible run: data -> model -> audit -> honest artifacts.

`python scripts/build_project.py` calls `run(root)`. Every number in the
notebooks and the Streamlit app is read back from what this writes, never typed
in by hand.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import SEED, __version__
from .data import prepared, validate
from .equity import (
    chi_square_tests,
    decline_rate_by_payer,
    payer_logit_summary,
)
from .modeling import cost_curve, coefficient_table, train

_TRACKED_PACKAGES = ["pandas", "numpy", "scikit-learn", "statsmodels", "scipy"]


def _manifest(df: pd.DataFrame, threshold: float) -> dict:
    return {
        "project_version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name) for name in _TRACKED_PACKAGES
        },
        "rows_after_clean": int(len(df)),
        "split_counts": df["split"].value_counts().to_dict(),
        "operating_threshold": threshold,
        "data": "synthetic — no real patient, facility, or payer data",
    }


def _plot_confusion(cm: dict, title: str, path: Path) -> None:
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "#16324a", fontsize=13)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Pred. accept", "Pred. decline"],
        yticklabels=["Actual accept", "Actual decline"],
        title=title,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cost_curve(curve: pd.DataFrame, threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(curve["threshold"], curve["weighted_cost"], color="#2f6f4e")
    ax.axvline(threshold, color="#c0392b", linestyle="--",
               label=f"selected = {threshold:.2f}")
    ax.axvline(0.5, color="#888", linestyle=":", label="default = 0.50")
    ax.set(xlabel="Decision threshold", ylabel="Weighted error cost (validation)",
           title="Threshold selection on the validation split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_calibration(y_true, y_prob, path: Path) -> None:
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#888", linestyle=":")
    ax.plot(mean_pred, frac_pos, marker="o", color="#2f6f4e")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed decline rate",
           title="Test-set calibration (quantile bins)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(root: Path) -> dict:
    root = Path(root)
    artifacts = root / "artifacts"
    reports = root / "reports"
    figures = reports / "figures"
    for folder in (artifacts, reports, figures):
        folder.mkdir(parents=True, exist_ok=True)

    df = prepared(root)
    checks = validate(df)
    (reports / "data_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    model = train(df)

    # --- model artifacts -----------------------------------------------------
    joblib.dump(
        {"pipeline": model.pipeline, "threshold": model.threshold,
         "feature_columns": list(model.pipeline.named_steps["encode"].feature_names_in_)},
        artifacts / "decline_model.joblib",
    )
    (artifacts / "model_selection.json").write_text(
        json.dumps(model.selection, indent=2), encoding="utf-8"
    )
    (artifacts / "run_manifest.json").write_text(
        json.dumps(_manifest(df, model.threshold), indent=2), encoding="utf-8"
    )

    metrics = {"operating_threshold": model.threshold, "splits": model.metrics}
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    coefficient_table(model.pipeline).to_csv(reports / "coefficients.csv", index=False)

    validation_df = df[df["split"] == "validation"]
    p_val = model.pipeline.predict_proba(
        validation_df[model.pipeline.named_steps["encode"].feature_names_in_]
    )[:, 1]
    curve = cost_curve(validation_df["declined"].to_numpy(), p_val)
    curve.to_csv(reports / "threshold_cost_curve.csv", index=False)

    test_df = df[df["split"] == "test"].copy()
    p_test = model.predict_proba(test_df)
    test_df_out = test_df[["transfer_request_id", "declined", "payer",
                           "requested_service_line", "requested_level_of_care"]].copy()
    test_df_out["decline_probability"] = p_test
    test_df_out["flagged"] = (p_test >= model.threshold).astype(int)
    test_df_out.to_csv(reports / "test_predictions.csv", index=False)

    # --- equity audit ------------------------------------------------------
    chi = chi_square_tests(df)
    payer_summary = payer_logit_summary(df)
    (reports / "equity_chi_square.json").write_text(json.dumps(chi, indent=2), encoding="utf-8")
    (reports / "equity_payer_logit.json").write_text(json.dumps(payer_summary, indent=2), encoding="utf-8")
    decline_rate_by_payer(df).to_csv(reports / "decline_rate_by_payer.csv", index=False)

    # --- figures ---------------------------------------------------------
    _plot_confusion(model.metrics["test"]["confusion_matrix"],
                    f"Test set · threshold {model.threshold:.2f}",
                    figures / "confusion_test.png")
    _plot_confusion(model.metrics["test_default_threshold"]["confusion_matrix"],
                    "Test set · default threshold 0.50",
                    figures / "confusion_test_default.png")
    _plot_cost_curve(curve, model.threshold, figures / "threshold_cost_curve.png")
    _plot_calibration(test_df["declined"].to_numpy(), p_test, figures / "calibration_test.png")

    return {
        "threshold": model.threshold,
        "test": model.metrics["test"],
        "test_default_threshold": model.metrics["test_default_threshold"],
        "payer_audit": payer_summary["reading"],
    }

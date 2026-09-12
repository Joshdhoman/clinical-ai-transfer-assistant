"""Reproducible held-out metrics, missingness, failures, and subgroup diagnostics."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.tree import export_text

from .extraction import RuleBasedExtractor
from .modeling import FEATURES, feature_frame, train_model
from .routing import recommend
from .schema import ABSTAIN, CRITICAL, FIELDS, ROUTES
from .synthetic import SEED, write_dataset


def classification_metrics(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=ROUTES, average="macro", zero_division=0)
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    covered = pred != ABSTAIN
    # Accuracy includes abstentions as failures; selective accuracy is secondary.
    return {"n": len(truth), "accuracy": float(accuracy_score(truth, pred)),
            "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1),
            "coverage": float(covered.mean()),
            "selective_accuracy": float(accuracy_score(truth[covered], pred[covered])) if covered.any() else None,
            "icu_false_positives": int(((pred == "ICU") & (truth != "ICU")).sum()),
            "icu_false_negatives": int(((truth == "ICU") & (pred != "ICU")).sum()),
            "icu_fn_abstentions": int(((truth == "ICU") & (pred == ABSTAIN)).sum()),
            "per_class": classification_report(truth, pred, labels=ROUTES, output_dict=True, zero_division=0),
            "confusion_labels": ROUTES + [ABSTAIN],
            "confusion_matrix": confusion_matrix(truth, pred, labels=ROUTES + [ABSTAIN]).tolist()}


def extraction_metrics(data, results):
    rows, errors = [], []
    truth_missing, pred_missing = [], []
    for field in FIELDS:
        tp = fp = fn = exact = 0
        for row, result in zip(data.itertuples(), results):
            expected = json.loads(row.visible_annotations)[field]
            actual = result.values[field]
            exact += expected == actual
            tp += expected is not None and expected == actual
            fp += actual is not None and expected != actual
            fn += expected is not None and expected != actual
            if field in CRITICAL:
                truth_missing.append(expected is None)
                pred_missing.append(actual is None)
            if expected != actual:
                errors.append({"request_id": row.request_id, "field": field, "expected": expected,
                               "actual": actual, "note": row.free_text_transfer_note})
        p, r = tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0
        rows.append({"field": field, "n": len(data), "exact_match_accuracy": exact / len(data),
                     "precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0,
                     "tp": tp, "fp": fp, "fn": fn})
    p, r, f, _ = precision_recall_fscore_support(truth_missing, pred_missing, average="binary", zero_division=0)
    missing = {"unit": "routing field per test note", "positive": "unavailable (absent or conflicting)",
               "precision": float(p), "recall": float(r), "f1": float(f),
               "accuracy": float(accuracy_score(truth_missing, pred_missing)),
               "confusion_matrix": confusion_matrix(truth_missing, pred_missing, labels=[False, True]).tolist()}
    return pd.DataFrame(rows), pd.DataFrame(errors, columns=["request_id", "field", "expected", "actual", "note"]), missing


def plot_confusions(metrics, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    labels = ["ICU", "Telemetry", "Med/surg", "Specialty", "Withheld"]
    for ax, name in zip(axes, ["rules", "decision_tree"]):
        matrix = np.asarray(metrics[name]["confusion_matrix"])[:4]
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max()/2 else "#19364a")
        ax.set(xticks=range(5), xticklabels=labels, yticks=range(4), yticklabels=labels[:4],
               xlabel="Predicted route", ylabel="Synthetic scenario target",
               title=f"{name.replace('_', ' ').title()} | accuracy {metrics[name]['accuracy']:.1%}")
    fig.suptitle("Held-out template families · synthetic data only", fontsize=15)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_pipeline(root: Path) -> dict:
    for folder in ["artifacts", "reports", "reports/figures"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    data = write_dataset(root)
    extractor = RuleBasedExtractor()
    results = [extractor.extract(n) for n in data.free_text_transfer_note]
    x = feature_frame(results)
    train, val, test = [data.split.eq(s) for s in ["train", "validation", "test"]]
    model, selection = train_model(x[train], data.loc[train, "routing_target"], x[val], data.loc[val, "routing_target"])
    test_data = data[test].reset_index(drop=True)
    test_results = [r for r, use in zip(results, test) if use]
    recommendations = [recommend(r) for r in test_results]
    predictions = test_data[["request_id", "routing_target", "patient_age", "referring_facility", "template_family", "free_text_transfer_note"]].copy()
    predictions["rules"] = [r.route for r in recommendations]
    predictions["decision_tree"] = model.predict(x[test])
    predictions["critical_missing_count"] = [sum(r.values[f] is None for f in CRITICAL) for r in test_results]
    predictions["rule_explanation"] = [" ".join(r.reasons) for r in recommendations]
    predictions["conflicts"] = [json.dumps(r.conflicts) for r in test_results]
    metrics = {name: classification_metrics(predictions.routing_target, predictions[name]) for name in ["rules", "decision_tree"]}
    # A majority-class baseline makes the comparison interpretable.
    majority = data.loc[train, "routing_target"].mode()[0]
    metrics["majority_baseline"] = classification_metrics(predictions.routing_target, [majority]*len(predictions))
    extraction, extraction_errors, missing = extraction_metrics(test_data, test_results)
    metrics["missing_information"] = missing
    joblib.dump(model, root / "artifacts/routing_tree.joblib")
    (root / "artifacts/model_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    names = model.named_steps["preprocess"].get_feature_names_out()
    (root / "reports/decision_tree.txt").write_text(export_text(model.named_steps["tree"], feature_names=list(names)), encoding="utf-8")
    pd.DataFrame({"feature": names, "importance": model.named_steps["tree"].feature_importances_}).sort_values("importance", ascending=False).to_csv(root / "reports/feature_importance.csv", index=False)
    predictions.to_csv(root / "reports/test_predictions.csv", index=False)
    extraction.to_csv(root / "reports/extraction_metrics.csv", index=False)
    extraction_errors.to_csv(root / "reports/extraction_errors.csv", index=False)
    (root / "reports/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusions(metrics, root / "reports/figures/confusion_matrices.png")
    failures = []
    for row in predictions.itertuples():
        for approach in ["rules", "decision_tree"]:
            predicted = getattr(row, approach)
            if predicted == row.routing_target:
                continue
            kind = "abstention" if predicted == ABSTAIN else "ICU false negative" if row.routing_target == "ICU" else "ICU false positive" if predicted == "ICU" else "route mismatch"
            failures.append({"request_id": row.request_id, "approach": approach, "expected": row.routing_target,
                             "predicted": predicted, "failure_type": kind,
                             "critical_missing_count": row.critical_missing_count,
                             "note": row.free_text_transfer_note, "rule_explanation": row.rule_explanation})
    failure_frame = pd.DataFrame(failures)
    failure_frame.to_csv(root / "reports/failures.csv", index=False)
    subgroups = []
    predictions["age_band"] = pd.cut(predictions.patient_age, [17, 39, 64, 120], labels=["18-39", "40-64", "65+"])
    predictions["completeness_group"] = np.where(predictions.critical_missing_count == 0, "complete routing fields", "incomplete routing fields")
    for dimension in ["age_band", "referring_facility", "completeness_group", "template_family"]:
        for group, frame in predictions.groupby(dimension, observed=True):
            for approach in ["rules", "decision_tree"]:
                m = classification_metrics(frame.routing_target, frame[approach])
                subgroups.append({"dimension": dimension, "group": str(group), "approach": approach,
                                  **{k: m[k] for k in ["n", "accuracy", "macro_f1", "coverage", "icu_false_negatives"]}})
    pd.DataFrame(subgroups).to_csv(root / "reports/subgroups.csv", index=False)
    manifest = {"seed": SEED, "python": platform.python_version(), "features": FEATURES,
                "packages": {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "scikit-learn", "streamlit", "plotly", "matplotlib"]},
                "split_counts": data.split.value_counts().to_dict(), "selection": selection,
                "dataset_sha256": hashlib.sha256((root / "data/synthetic/transfer_requests.csv").read_bytes()).hexdigest(),
                "test_use": "Excluded from fitting and hyperparameter selection; development test diagnostics informed parser bug fixes. Not independent external validation.",
                "simulation_caveat": "Labels and vocabulary derive from one authored scenario generator; metrics are not clinical validation."}
    (root / "artifacts/run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = ["# Executed evaluation", "", "Synthetic-only results. 625 training / 125 validation / 250 test records; template families do not overlap.", "",
              "| Approach | Accuracy | Macro precision | Macro recall | Macro F1 | Coverage | ICU FP | ICU FN* |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["majority_baseline", "rules", "decision_tree"]:
        m = metrics[name]
        report.append(f"| {name} | {m['accuracy']:.3f} | {m['macro_precision']:.3f} | {m['macro_recall']:.3f} | {m['macro_f1']:.3f} | {m['coverage']:.3f} | {m['icu_false_positives']} | {m['icu_false_negatives']} |")
    report.extend(["", "*ICU false negatives include withheld suggestions; inspect `icu_fn_abstentions` in metrics.json to distinguish escalation from lower-acuity errors.",
                   "", f"Rules selective accuracy among covered requests: {metrics['rules']['selective_accuracy']:.3f}. This excludes withheld requests and must not replace overall accuracy.",
                   f"Selected tree depth: {selection['selected_depth']} (validation macro-F1). Critical missing-information precision/recall/F1: {missing['precision']:.3f}/{missing['recall']:.3f}/{missing['f1']:.3f}.",
                   "", "## Per-route performance", "", "| Approach | Route | Precision | Recall | F1 | Support |", "|---|---|---:|---:|---:|---:|"])
    for name in ["rules", "decision_tree"]:
        for route in ROUTES:
            r = metrics[name]["per_class"][route]
            report.append(f"| {name} | {route} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {r['support']:.0f} |")
    report.extend(["", "## Interpretation", "", "The rules deliberately withhold lower-acuity suggestions when required referral facts are absent. The tree can infer a label from correlated service and vital-sign patterns, increasing coverage while risking unsupported confidence. Synthetic measurements are strongly scenario-correlated, making the ML task easier than real referrals. Neither system has independent clinician labels, external data, prospective evaluation, or calibrated clinical probabilities. Test diagnostics informed parser bug fixes; this is a development benchmark, not untouched external validation.",
                   "", "Missing-field scoring treats both absence and contradiction as unavailable. Extraction precision counts a wrong value as a false positive; recall also counts it as a false negative. Exact match includes unknowns, so it can look better than value recovery. Subgroup results are descriptive and do not establish fairness.",
                   "", "## Actual failure examples", ""])
    # Include examples from both models and different error types, without cherry-picking success.
    if len(failure_frame):
        examples = failure_frame.groupby(["approach", "failure_type"], sort=True).head(1)
        for row in examples.itertuples():
            report.extend([f"### {row.request_id} — {row.approach}: {row.failure_type}", "",
                           f"Scenario target: **{row.expected}**. Prediction: **{row.predicted}**. Missing routing fields: {row.critical_missing_count}.",
                           "", row.note, "", f"Rules rationale: {row.rule_explanation}", ""])
    report.extend(["## Files", "", "See `metrics.json`, `test_predictions.csv`, `failures.csv`, `extraction_metrics.csv`, `extraction_errors.csv`, `subgroups.csv`, and `decision_tree.txt` for complete results."])
    (root / "reports/RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics

"""Transfer-center decline prediction — synthetic demonstration.

Not a clinical tool. Synthetic data, illustrative error costs, decision support
only. Run `python scripts/build_project.py` to regenerate everything this reads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

# This project lives in a subdirectory of the repository, so the hosted demo is
# not pip-installed. Put its own src/ on the path before importing the package.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from transfer_decline.data import prepared
from transfer_decline.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)
from transfer_decline.modeling import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE

st.set_page_config(page_title="Transfer-center decline prediction", page_icon="🏥", layout="wide")


@st.cache_resource
def load_model():
    bundle = joblib.load(ROOT / "artifacts" / "decline_model.joblib")
    return bundle["pipeline"], bundle["threshold"]


@st.cache_data
def load_reports():
    return {
        "metrics": json.loads((ROOT / "reports" / "metrics.json").read_text()),
        "selection": json.loads((ROOT / "artifacts" / "model_selection.json").read_text()),
        "manifest": json.loads((ROOT / "artifacts" / "run_manifest.json").read_text()),
        "coefficients": pd.read_csv(ROOT / "reports" / "coefficients.csv"),
        "chi": json.loads((ROOT / "reports" / "equity_chi_square.json").read_text()),
        "payer": json.loads((ROOT / "reports" / "equity_payer_logit.json").read_text()),
        "cost_curve": pd.read_csv(ROOT / "reports" / "threshold_cost_curve.csv"),
    }


@st.cache_data
def load_frame():
    return prepared(ROOT)


pipeline, threshold = load_model()
reports = load_reports()
data = load_frame()

st.title("Transfer-center decline prediction & payer-equity audit")
st.warning(
    "**Synthetic data. Decision support only. Not validated for clinical use.** "
    "Error costs are illustrative planning weights, not measured. Nothing here "
    "describes any real hospital, transfer center, or payer.",
    icon="⚠️",
)

scorer, evaluation, equity, about = st.tabs(
    ["Score a request", "Model evaluation", "Payer-equity audit", "About"]
)

# --------------------------------------------------------------------------- #
with scorer:
    st.subheader("Estimate decline probability at request time")
    st.caption(
        f"Flagged when the estimated probability reaches the operating threshold "
        f"of **{threshold:.2f}**, chosen on the validation split by minimising "
        f"{COST_FALSE_NEGATIVE:.0f}× (missed decline) + {COST_FALSE_POSITIVE:.0f}× (false alarm)."
    )
    defaults = data[FEATURE_COLUMNS].iloc[0]
    cols = st.columns(3)
    entry = {}
    numeric_bounds = {
        "patient_age": (18, 99, 1),
        "distance_miles": (0.0, 200.0, 1.0),
        "acuity_score": (1.0, 5.0, 1.0),
        "icu_occupancy_pct": (0.5, 1.0, 0.01),
        "medsurg_occupancy_pct": (0.5, 1.0, 0.01),
        "ed_boarding_count": (0, 30, 1),
        "request_hour": (0, 23, 1),
        "request_dow": (0, 6, 1),
        "request_month": (1, 12, 1),
    }
    for i, feature in enumerate(NUMERIC_FEATURES):
        low, high, step = numeric_bounds[feature]
        with cols[i % 3]:
            entry[feature] = st.slider(
                feature, float(low), float(high), float(defaults[feature]), float(step)
            )
    for i, feature in enumerate(["is_weekend", "is_overnight"]):
        with cols[i % 3]:
            entry[feature] = int(st.checkbox(feature, value=bool(defaults[feature])))
    for i, feature in enumerate(CATEGORICAL_FEATURES):
        options = sorted(data[feature].dropna().unique())
        with cols[i % 3]:
            entry[feature] = st.selectbox(
                feature, options, index=options.index(defaults[feature]) if defaults[feature] in options else 0
            )

    request = pd.DataFrame([entry])[FEATURE_COLUMNS]
    probability = float(pipeline.predict_proba(request)[:, 1][0])
    flagged = probability >= threshold

    left, right = st.columns([1, 2])
    left.metric("Estimated decline probability", f"{probability:.1%}")
    (left.error if flagged else left.success)(
        "FLAG for early review" if flagged else "Not flagged"
    )
    right.caption(
        "The flag is an early-warning prompt for a coordinator, not a decision. "
        "A person still evaluates capacity, acuity, service acceptance, and placement."
    )

# --------------------------------------------------------------------------- #
with evaluation:
    st.subheader("Held-out test performance")
    splits = reports["metrics"]["splits"]
    table = pd.DataFrame(
        {
            s: {k: splits[s][k] for k in ["n", "roc_auc", "pr_auc", "brier", "precision", "recall", "f1", "specificity", "weighted_cost"]}
            for s in ["train", "validation", "test"]
        }
    ).T
    st.dataframe(table.style.format(precision=3), width="stretch")

    c1, c2 = st.columns(2)
    c1.markdown("**At the operating threshold**")
    c1.image(str(ROOT / "reports" / "figures" / "confusion_test.png"))
    c2.markdown("**At the default 0.50 threshold**")
    c2.image(str(ROOT / "reports" / "figures" / "confusion_test_default.png"))
    st.caption(
        f"Default 0.50 recall on declines: {splits['test_default_threshold']['recall']:.0%}. "
        f"At the selected threshold: {splits['test']['recall']:.0%}. "
        "Lowering the threshold is what makes the model operationally useful when a "
        "missed decline is the costlier error."
    )

    c3, c4 = st.columns(2)
    c3.image(str(ROOT / "reports" / "figures" / "threshold_cost_curve.png"))
    c4.image(str(ROOT / "reports" / "figures" / "calibration_test.png"))

    st.markdown("**Largest logistic-regression effects (odds ratios)**")
    st.dataframe(reports["coefficients"].head(15).style.format(precision=3), width="stretch")

# --------------------------------------------------------------------------- #
with equity:
    st.subheader("Is the decline decision associated with payer?")
    payer_chi = {k: v for k, v in reports["chi"]["payer_vs_decision"].items() if k != "table"}
    occ_chi = {k: v for k, v in reports["chi"]["medsurg_occupancy_vs_decision"].items() if k != "table"}
    c1, c2 = st.columns(2)
    c1.markdown("**Chi-square: payer × decision**")
    c1.json(payer_chi)
    c2.markdown("**Chi-square: occupancy band × decision** (mechanism check)")
    c2.json(occ_chi)

    st.markdown("**Multivariable logistic regression — adjusted odds ratios for payer**")
    st.caption(
        "Adjusted for acuity, ICU and med-surg occupancy, ED boarding, service line, "
        f"and level of care. Reference payer: {reports['payer']['reference_payer']}. "
        f"Complete-case n = {reports['payer']['n_complete_case']:,}."
    )
    st.dataframe(
        pd.DataFrame(reports["payer"]["odds_ratios"]).style.format(precision=3),
        width="stretch",
    )
    st.info(reports["payer"]["reading"])
    st.caption(
        "Payer is **excluded from the predictive model** regardless of this result — "
        "a clean audit is not a permanent guarantee, and a model that never sees payer "
        "cannot learn to use it."
    )

# --------------------------------------------------------------------------- #
with about:
    st.markdown(
        """
This is a portfolio project by an RN Clinical Placement Coordinator with an MS in
Data Science. It works a synthetic transfer-center extract from raw data through
a reproducible pipeline: clean and split chronologically, engineer request-time
features (no post-decision leakage), audit payer equity with a chi-square test
and a covariate-adjusted logistic regression, train a logistic-regression decline
model, and choose an operating threshold on the validation split by expected
weighted error cost.

- **Repo & notebooks:** github.com/Joshdhoman/transfer-center-decline-prediction
- **Companion project:** Clinical AI Transfer Assistant (NLP + interpretable routing)

**What it can show:** a reproducible workflow, a leakage-aware split, a defensible
threshold argument, and an equity audit method. **What it cannot show:** anything
about real transfer-center performance, real payer equity, time savings, or
clinical safety.
"""
    )
    st.json(reports["manifest"])

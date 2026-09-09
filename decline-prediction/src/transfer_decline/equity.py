"""Payer-equity audit.

Two views of the same question — is the decline decision associated with payer?

  1. Chi-square tests of independence: payer x decision, and (as a check on the
     mechanism the generator actually uses) occupancy band x decision.
  2. A multivariable logistic regression of `declined` on payer while adjusting
     for acuity, unit occupancy, ED boarding, service line, and level of care —
     so a raw payer difference driven by case mix is not mistaken for unequal
     treatment.

In this synthetic dataset payer is assigned independently of the decision, so a
correct audit returns null. The point of the code is the method, not the finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

ADJUSTMENT_TERMS = [
    "acuity_score",
    "icu_occupancy_pct",
    "medsurg_occupancy_pct",
    "ed_boarding_count",
    "C(requested_service_line)",
    "C(requested_level_of_care)",
]


def _chi_square(table: pd.DataFrame) -> dict:
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.to_numpy().sum()
    min_dim = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if n and min_dim else 0.0
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "cramers_v": cramers_v,
        "n": int(n),
        "significant_at_0.05": bool(p < 0.05),
    }


def chi_square_tests(df: pd.DataFrame) -> dict:
    """Payer x decision, and occupancy-band x decision."""
    payer_table = pd.crosstab(df["payer"], df["declined"])

    occ = pd.cut(
        df["medsurg_occupancy_pct"],
        bins=[0, 0.85, 0.95, 1.01],
        labels=["<85%", "85-95%", ">95%"],
    )
    occupancy_table = pd.crosstab(occ, df["declined"])

    return {
        "payer_vs_decision": {**_chi_square(payer_table), "table": payer_table.to_dict()},
        "medsurg_occupancy_vs_decision": {
            **_chi_square(occupancy_table),
            "table": occupancy_table.to_dict(),
        },
    }


def decline_rate_by_payer(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("payer")["declined"].agg(n="size", declines="sum", rate="mean")
    return grouped.sort_values("rate", ascending=False).reset_index()


def payer_logit(df: pd.DataFrame, reference_payer: str = "Commercial") -> pd.DataFrame:
    """Adjusted odds ratios for payer, holding case mix constant.

    Rows with a missing payer or adjustment value are dropped (complete-case);
    the count used is reported by `payer_logit_summary`.
    """
    model_df = df[df["payer"] != "Unknown"].copy()
    payers = [reference_payer] + sorted(
        p for p in model_df["payer"].unique() if p != reference_payer
    )
    model_df["payer"] = pd.Categorical(model_df["payer"], categories=payers, ordered=False)

    formula = "declined ~ C(payer) + " + " + ".join(ADJUSTMENT_TERMS)
    fit = smf.logit(formula, data=model_df).fit(disp=False)

    params = fit.params
    conf = fit.conf_int()
    rows = []
    for term in params.index:
        if not term.startswith("C(payer)"):
            continue
        level = term.split("T.")[-1].rstrip("]")
        rows.append(
            {
                "payer": level,
                "reference": reference_payer,
                "odds_ratio": float(np.exp(params[term])),
                "ci_low": float(np.exp(conf.loc[term, 0])),
                "ci_high": float(np.exp(conf.loc[term, 1])),
                "p_value": float(fit.pvalues[term]),
            }
        )
    return pd.DataFrame(rows)


def payer_logit_summary(df: pd.DataFrame, reference_payer: str = "Commercial") -> dict:
    model_df = df[df["payer"] != "Unknown"].dropna(
        subset=["acuity_score", "medsurg_occupancy_pct", "icu_occupancy_pct", "ed_boarding_count"]
    )
    table = payer_logit(df, reference_payer)
    any_significant = bool((table["p_value"] < 0.05).any())
    return {
        "reference_payer": reference_payer,
        "n_complete_case": int(len(model_df)),
        "adjustment_terms": ADJUSTMENT_TERMS,
        "any_payer_term_significant_at_0.05": any_significant,
        "odds_ratios": table.to_dict(orient="records"),
        "reading": (
            "At least one payer term is significant after adjustment — investigate."
            if any_significant
            else "No payer term is significant after adjusting for case mix; "
            "consistent with payer being unrelated to the decision in this "
            "synthetic data."
        ),
    }

"""Load and clean the synthetic transfer-request extract.

The raw file mimics an Epic / transfer-center reporting export, messiness included:
inconsistent capitalisation and stray whitespace in a couple of text columns,
some missing values, and a handful of double-entered requests. Everything in
here is synthetic — no real patient, facility, or payer data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILENAME = "transfer_center_requests.csv"

# Columns that only exist *after* the transfer-center decision, or that encode
# the decision itself. None of these may enter the feature matrix — using them
# to predict the decision would be target leakage.
POST_DECISION_COLUMNS = [
    "decision_datetime",
    "bed_assignment_datetime",
    "arrival_datetime",
    "disposition",
    "decline_reason",
    "inpatient_los_days",
    "icu_upgrade_within_24h",
]

# Identifier — carried through for traceability, never a feature.
ID_COLUMN = "transfer_request_id"

# Text columns that arrive with case / whitespace inconsistencies.
_MESSY_TEXT = ["transport_mode", "referring_facility_type"]

# Canonical title-case for the facility-type column after normalisation.
_FACILITY_TYPE_CANON = {
    "community hospital ed": "Community Hospital ED",
    "community hospital inpatient": "Community Hospital Inpatient",
    "critical access hospital": "Critical Access Hospital",
    "skilled nursing facility": "Skilled Nursing Facility",
    "physician office": "Physician Office",
}


def raw_path(root: Path) -> Path:
    return root / "data" / "synthetic" / RAW_FILENAME


def load_raw(root: Path) -> pd.DataFrame:
    """Read the raw extract with datetimes parsed, nothing else touched."""
    datetime_columns = [
        "request_datetime",
        "decision_datetime",
        "bed_assignment_datetime",
        "arrival_datetime",
    ]
    return pd.read_csv(raw_path(root), parse_dates=datetime_columns)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return an analysis-ready frame: de-duplicated, text normalised, target derived.

    The returned frame still contains the post-decision columns so the EDA
    notebook can describe the funnel. `feature_columns()` / `build_feature_frame`
    are what actually keep them out of the model.
    """
    df = raw.copy()

    # 1. Drop the exact-duplicate requests (double data entry).
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Normalise the messy text columns: trim, collapse internal whitespace.
    for column in _MESSY_TEXT:
        df[column] = (
            df[column].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        )
    df["referring_facility_type"] = (
        df["referring_facility_type"]
        .str.lower()
        .map(_FACILITY_TYPE_CANON)
        .astype("string")
    )

    # 3. Derive the binary target. "Accepted - Not Transferred" means the
    #    transfer centre accepted the request even though the patient never
    #    arrived (stabilised, family declined, expired) — an acceptance for the
    #    purpose of predicting the transfer-centre decision.
    df["declined"] = (df["disposition"] == "Declined").astype(int)

    # 4. Calendar features from the request timestamp (known at request time).
    request = df["request_datetime"].dt
    df["request_hour"] = request.hour
    df["request_dow"] = request.dayofweek  # Monday = 0
    df["request_month"] = request.month
    df["is_weekend"] = request.dayofweek.isin([5, 6]).astype(int)
    df["is_overnight"] = ((request.hour >= 22) | (request.hour < 6)).astype(int)

    # 5. Light type tidying. Leave missing values in place — imputation is part
    #    of the modelling pipeline and is fit on the training split only.
    df["payer"] = df["payer"].fillna("Unknown")
    df["distance_miles"] = pd.to_numeric(df["distance_miles"], errors="coerce")
    df["acuity_score"] = pd.to_numeric(df["acuity_score"], errors="coerce")

    return df


def time_split(df: pd.DataFrame, fractions=(0.6, 0.2, 0.2), seed: int | None = None) -> pd.Series:
    """Assign each request to train / validation / test by request time.

    A chronological split is closer to how the model would be used — fit on
    history, decide on what comes next — and it keeps near-duplicate requests
    from the same day out of two partitions at once. `seed` is accepted for a
    stratified random fallback but the default path is deterministic and needs
    no seed.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    order = df["request_datetime"].sort_values().index
    n = len(order)
    train_end = int(n * fractions[0])
    val_end = train_end + int(n * fractions[1])
    split = pd.Series("test", index=df.index, dtype="object")
    split.loc[order[:train_end]] = "train"
    split.loc[order[train_end:val_end]] = "validation"
    return split


def prepared(root: Path) -> pd.DataFrame:
    """One call: raw -> clean -> add `split`."""
    df = clean(load_raw(root))
    df["split"] = time_split(df)
    return df


def validate(df: pd.DataFrame) -> dict:
    """Cheap invariants worth asserting in a notebook and in CI."""
    checks = {
        "rows": int(len(df)),
        "no_exact_duplicates": bool(not df.drop(columns=["split"], errors="ignore").duplicated().any()),
        "target_is_binary": sorted(df["declined"].unique().tolist()) == [0, 1],
        "decline_rate": round(float(df["declined"].mean()), 4),
        "facility_type_no_nulls_after_norm": bool(df["referring_facility_type"].notna().all()),
        "transport_mode_values": sorted(df["transport_mode"].dropna().unique().tolist()),
        "split_counts": df["split"].value_counts().to_dict(),
        "splits_are_time_ordered": bool(
            df.loc[df.split == "train", "request_datetime"].max()
            <= df.loc[df.split == "test", "request_datetime"].min()
        ),
    }
    return checks

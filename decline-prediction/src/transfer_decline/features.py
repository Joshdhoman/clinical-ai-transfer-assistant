"""Feature matrix construction.

Only information available at the moment the request is logged is allowed in.
The encoder is a scikit-learn ColumnTransformer so it can be fit on the training
split and reused unchanged on validation and test.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Payer is deliberately excluded from the predictive model (see notebook 04 and
# the model card). It is audited separately; a model that never sees it cannot
# learn to lean on it.
NUMERIC_FEATURES = [
    "patient_age",
    "distance_miles",
    "acuity_score",
    "icu_occupancy_pct",
    "medsurg_occupancy_pct",
    "ed_boarding_count",
    "request_hour",
    "request_dow",
    "request_month",
]

BINARY_FEATURES = ["is_weekend", "is_overnight"]

CATEGORICAL_FEATURES = [
    "referring_facility",
    "referring_facility_type",
    "transport_mode",
    "requested_service_line",
    "requested_level_of_care",
    "sex",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

TARGET = "declined"

# Columns that must never appear in FEATURE_COLUMNS. Imported by the tests.
FORBIDDEN = {
    "transfer_request_id",
    "disposition",
    "decline_reason",
    "decision_datetime",
    "bed_assignment_datetime",
    "arrival_datetime",
    "inpatient_los_days",
    "icu_upgrade_within_24h",
    "payer",
    "declined",
}


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model input columns. No fitting happens here."""
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"feature columns absent from frame: {sorted(missing)}")
    return df.loc[:, FEATURE_COLUMNS].copy()


def make_encoder() -> ColumnTransformer:
    """Median-impute + scale numerics, most-frequent-impute + one-hot categoricals."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )

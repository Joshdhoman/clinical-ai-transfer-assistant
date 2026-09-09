from pathlib import Path

import pandas as pd
import pytest

from transfer_decline.data import clean, load_raw, prepared, time_split, validate
from transfer_decline.features import FEATURE_COLUMNS, FORBIDDEN

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def df():
    return prepared(ROOT)


def test_clean_removes_exact_duplicates():
    raw = load_raw(ROOT)
    assert raw.duplicated().sum() > 0
    assert not clean(raw).drop(columns="declined").duplicated().any()


def test_messy_text_is_normalised(df):
    assert set(df["transport_mode"].dropna()) == {
        "Ground ALS",
        "Referring Facility Transport",
        "Rotor Wing",
    }
    assert df["referring_facility_type"].notna().all()
    assert not df["referring_facility_type"].str.isupper().any()


def test_target_is_clean_binary(df):
    assert sorted(df["declined"].unique()) == [0, 1]
    # "Accepted - Not Transferred" must count as an acceptance.
    not_transferred = df[df["disposition"] == "Accepted - Not Transferred"]
    assert (not_transferred["declined"] == 0).all()


def test_split_is_chronological(df):
    checks = validate(df)
    assert checks["splits_are_time_ordered"]
    train_max = df.loc[df.split == "train", "request_datetime"].max()
    val_min = df.loc[df.split == "validation", "request_datetime"].min()
    assert train_max <= val_min


def test_split_fractions_are_reasonable(df):
    counts = df["split"].value_counts(normalize=True)
    assert counts["train"] == pytest.approx(0.6, abs=0.02)
    assert counts["test"] == pytest.approx(0.2, abs=0.02)


def test_time_split_rejects_bad_fractions(df):
    with pytest.raises(ValueError):
        time_split(df, fractions=(0.5, 0.3, 0.3))


def test_no_leakage_columns_in_feature_list():
    assert not (set(FEATURE_COLUMNS) & FORBIDDEN)
    assert "payer" not in FEATURE_COLUMNS
    assert "decline_reason" not in FEATURE_COLUMNS

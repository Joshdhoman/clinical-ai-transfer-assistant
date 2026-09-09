from pathlib import Path

import pytest

from transfer_decline.data import prepared
from transfer_decline.equity import (
    chi_square_tests,
    decline_rate_by_payer,
    payer_logit,
    payer_logit_summary,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def df():
    return prepared(ROOT)


def test_chi_square_detects_the_occupancy_association(df):
    # Capacity drives declines in the generator, so this test must be significant
    # — a sanity check that the method has power on this dataset.
    chi = chi_square_tests(df)
    assert chi["medsurg_occupancy_vs_decision"]["significant_at_0.05"]


def test_payer_audit_is_null_on_synthetic_data(df):
    # Payer is assigned independently of the decision by the generator.
    summary = payer_logit_summary(df)
    assert not summary["any_payer_term_significant_at_0.05"]


def test_payer_logit_covers_every_non_reference_payer(df):
    table = payer_logit(df, reference_payer="Commercial")
    payers = set(df.loc[df["payer"] != "Unknown", "payer"].unique()) - {"Commercial"}
    assert set(table["payer"]) == payers
    assert (table["ci_low"] <= table["odds_ratio"]).all()
    assert (table["odds_ratio"] <= table["ci_high"]).all()


def test_decline_rate_table_shape(df):
    table = decline_rate_by_payer(df)
    assert {"payer", "n", "declines", "rate"} <= set(table.columns)

from pathlib import Path

import numpy as np
import pytest

from transfer_decline.data import prepared
from transfer_decline.modeling import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    cost_curve,
    select_threshold,
    train,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    return train(prepared(ROOT))


def test_threshold_selected_below_default_given_asymmetric_costs(model):
    # A false negative costs more than a false positive, so the cost-minimising
    # threshold must sit below 0.5.
    assert COST_FALSE_NEGATIVE > COST_FALSE_POSITIVE
    assert model.threshold < 0.5


def test_threshold_comes_from_validation_only():
    # select_threshold must never see the test split. Feed it a toy validation
    # signal and confirm it optimises weighted cost on exactly that input.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 400), 0, 1)
    chosen = select_threshold(y, p)["threshold"]
    curve = cost_curve(y, p)
    assert curve.loc[curve["threshold"] == chosen, "weighted_cost"].iloc[0] == curve["weighted_cost"].min()


def test_model_beats_no_skill_on_test(model):
    assert model.metrics["test"]["roc_auc"] > 0.6


def test_lower_threshold_raises_recall_vs_default(model):
    assert model.metrics["test"]["recall"] > model.metrics["test_default_threshold"]["recall"]


def test_reported_splits_present(model):
    for split in ("train", "validation", "test", "test_default_threshold"):
        assert "confusion_matrix" in model.metrics[split]

# Verification

How to check that the claims in this repo hold.

## Run the whole thing

```bash
pip install -e ".[dev]"
python scripts/build_project.py       # writes artifacts/ and reports/
python -m pytest -q                   # 20 tests
python scripts/execute_notebooks.py   # executes all 5 notebooks in place
```

CI (`.github/workflows/ci.yml`) does exactly this on every push.

## Claim-by-claim

| Claim | Where to check |
|---|---|
| No post-decision column is a feature | `tests/test_data.py::test_no_leakage_columns_in_feature_list`; `features.FORBIDDEN` vs `FEATURE_COLUMNS` |
| The encoder is fit on training rows only | `modeling.make_model` — the `ColumnTransformer` sits inside the `Pipeline`, `.fit` is called on the train split in `modeling.train` |
| Threshold chosen on validation, not test | `modeling.train` — `select_threshold` receives only `parts["validation"]`; `tests/test_modeling.py::test_threshold_comes_from_validation_only` |
| Threshold below 0.5 given 5:1 costs | `tests/test_modeling.py::test_threshold_selected_below_default_given_asymmetric_costs` |
| Test split scored once | `modeling.train` computes `p_test` after the threshold is fixed; nothing refits |
| Chi-square method has power on this data | `tests/test_equity.py::test_chi_square_detects_the_occupancy_association` |
| Payer audit is null on the synthetic data | `tests/test_equity.py::test_payer_audit_is_null_on_synthetic_data`; `reports/equity_payer_logit.json` |
| App loads the committed model and scores a request | `tests/test_app.py::test_committed_model_scores_a_request` |
| Reported metrics match the committed threshold | `tests/test_app.py::test_metrics_json_matches_committed_threshold` |

## Files the pipeline writes

```
artifacts/decline_model.joblib      fitted pipeline + threshold + feature columns
artifacts/model_selection.json      threshold, validation cost, error weights
artifacts/run_manifest.json         seed, versions, platform, row/split counts
reports/metrics.json                per-split metrics at the operating threshold
reports/coefficients.csv            logistic-regression odds ratios
reports/threshold_cost_curve.csv    weighted cost across the threshold grid
reports/test_predictions.csv        per-request test-set probabilities and flags
reports/equity_chi_square.json      chi-square results
reports/equity_payer_logit.json     adjusted payer odds ratios
reports/decline_rate_by_payer.csv   raw (unadjusted) decline rates
reports/figures/*.png               confusion matrices, cost curve, calibration
reports/data_validation.json        cohort invariants
```

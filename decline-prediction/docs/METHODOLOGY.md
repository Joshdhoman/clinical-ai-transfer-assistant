# Methodology

## Pipeline

```
load_raw ─► clean ─► time_split ─► build_feature_frame ─► make_encoder (fit on train)
                                          │
                    ┌─────────────────────┼─────────────────────┐
              train logit            select_threshold       evaluate_at
             (train split)          (validation split)     (test split, once)
                                          │
                                    equity audit (whole cohort)
                                          │
                              artifacts/ + reports/ + figures/
```

Everything runs from `python scripts/build_project.py`. The notebooks and the
Streamlit app read the written artifacts; they do not recompute headline numbers.

## Design decisions

### Target

`declined = 1` for disposition `Declined`. `Accepted - Not Transferred` is an
acceptance — the transfer center said yes; the patient not arriving is a
downstream event. Modelling that as a decline would blur the decision being
predicted.

### Chronological split

Earliest 60% of requests train, next 20% validation, last 20% test. Rationale:
the model's job is to score an incoming request from history, so evaluation
should mimic that ordering; and two near-identical requests from the same shift
should not land in different partitions. Cost: the validation and test windows
are genuinely different periods, so point estimates shift between them (see the
model card). A stratified random split is available in `time_split` as a
fallback but is not the default.

### Leakage control

`features.FORBIDDEN` lists every column that encodes the decision or postdates
it. `FEATURE_COLUMNS` is built only from request-time fields. A test asserts the
two sets are disjoint. The encoder is fit inside the sklearn pipeline on training
rows only, so imputation statistics and one-hot vocabularies never see
validation or test.

### Threshold selection

The default 0.5 cutoff is arbitrary for an operational flag. We define an error
cost — `5·FN + 1·FP` — where a missed decline (no early warning) is five times a
false alarm (a coordinator double-checks a request that was fine). We grid the
threshold from 0.02 to 0.98 and take the minimum-cost value **on validation**,
breaking ties toward higher recall. That value (0.17) is frozen before the test
split is touched. This is the correction to an earlier version that chose and
evaluated the threshold on the same data.

### No class weighting

`class_weight="balanced"` plus a lowered threshold would apply the
imbalance correction twice and push probabilities away from calibration. Plain
logistic regression keeps the Brier score low and makes the threshold move the
single, legible intervention.

### Equity audit

Two methods, because they answer slightly different questions:

- **Chi-square test of independence** — is payer associated with the decision at
  all, marginally? Reported with Cramér's V for effect size. The occupancy-band
  test is included as a positive control: it *should* be significant, and is.
- **Multivariable logistic regression** — does a payer difference survive
  adjustment for acuity, ICU and med-surg occupancy, ED boarding, service line,
  and level of care? This separates "declined more" from "sicker, or arriving
  when the hospital is full". Adjusted odds ratios with 95% CIs, reference payer
  Commercial, complete-case.

Both come back null here because the generator assigns payer at random with
respect to the decision. Payer is excluded from the predictive model regardless
of the audit result.

## Reproducing the reported run

`artifacts/run_manifest.json` records the seed, Python version, platform, and the
version of every analysis package. CI re-runs the whole pipeline, the tests, and
the notebooks on every push.

# Model card — transfer-center decline prediction

## Model and status

Logistic-regression classifier, version 0.1.0. Local portfolio prototype on
synthetic data. Predicts the probability that a transfer request will be declined,
using information available when the request is logged. Trained on the earliest
60% of requests; operating threshold selected on the next 20% (validation);
reported metrics computed once on the final 20% (test). See
[`reports/metrics.json`](../reports/metrics.json),
[`artifacts/model_selection.json`](../artifacts/model_selection.json), and
[`artifacts/run_manifest.json`](../artifacts/run_manifest.json).

## Intended use

Demonstrate a reproducible operations-analytics workflow: how a request-time
signal could give a transfer-center coordinator an early warning, and how the
decision threshold should follow the relative cost of the two errors. Intended
users are portfolio reviewers and analysts. The flag is a prompt for human
review, never a decision.

## Inappropriate uses

Do not use for accepting or declining transfers, triage, level-of-care
assignment, bed allocation, staffing, or any real patient. Do not use with real
data without a fresh equity audit, calibration check, and prospective evaluation.
No metric here authorizes operational use.

## Data

Synthetic extract of 9,000 transfer requests (after removing 60 exact-duplicate
rows) over roughly six months, imitating an Epic / transfer-center report. Fields:
request timestamp, referring facility and type, distance, transport mode,
requested service line and level of care, patient age and sex, payer, acuity
score, ICU and med-surg occupancy, ED boarding count, and the disposition.
Missing values occur in distance, payer, and acuity. See
[`DATA_DICTIONARY.md`](DATA_DICTIONARY.md).

**Target:** `declined = 1` when disposition is `Declined`. `Accepted - Not
Transferred` (accepted, patient never arrived) counts as an acceptance.

**Excluded from features (target leakage):** decision/bed-assignment/arrival
timestamps, decline reason, inpatient length of stay, ICU upgrade within 24h,
the request ID, and payer (audited separately, then withheld by design).

## Methodology

- **Split:** chronological, 60 / 20 / 20 train / validation / test. Chosen over a
  random split because the model would be used to score future requests from
  past history, and because near-duplicate requests from one shift should not
  span partitions.
- **Encoding:** median imputation + standardisation for numeric features (with a
  missing-indicator), most-frequent imputation + one-hot for categoricals, rare
  levels (< 20) folded together. Fit on training rows only, inside the pipeline.
- **Model:** `LogisticRegression`, no class weighting, so predicted
  probabilities stay close to calibrated and the threshold move is the whole
  intervention.
- **Threshold:** grid over 0.02–0.98; pick the value minimising
  `5·(false negatives) + 1·(false positives)` on validation; ties break toward
  higher recall. Selected value: **0.17**.

## Evaluation and limitations

Test ROC-AUC ≈ 0.76, PR-AUC ≈ 0.42 (prevalence 0.19). At threshold 0.17: recall
on declines ≈ 0.65, precision ≈ 0.37; at 0.50, recall ≈ 0.07. Brier ≈ 0.13.

- **Synthetic data with designed relationships.** Occupancy, ED boarding, acuity,
  and level of care drive declines because the generator was built that way.
  Performance reflects that design, not a discovered operational fact.
- **Split instability.** Validation recall at the selected threshold (≈ 0.41) is
  lower than test recall (≈ 0.65); the two 20% windows are different time
  periods. The weighted-cost improvement over the default threshold holds on
  both, but point estimates move between periods.
- **Illustrative costs.** The 5:1 ratio is a planning assumption. Real weights
  would come from operational and financial estimates and would change the
  threshold.
- **Single model.** No comparison to trees or gradient boosting; a first-pass
  interpretable baseline only.
- **No calibration layer, no subgroup performance guarantees, no drift
  monitoring.** All of these would be required before real use.

## Equity

Payer-equity audit: chi-square test of payer × decision, and a logistic
regression of `declined` on payer adjusting for acuity, ICU and med-surg
occupancy, ED boarding, service line, and level of care (reference payer:
Commercial; complete-case n ≈ 8,600). On this synthetic data the audit is null —
no payer term significant, all adjusted odds-ratio CIs cross 1. On real data the
same audit could surface a disparity; that would be the start of an
investigation, not a verdict. Payer is excluded from the predictive model
regardless.

## Human oversight

The model output is an early-warning flag. A coordinator still evaluates
capacity, acuity, service acceptance, transport, and placement, and makes the
decision. Recording or acting on a flag does not accept, schedule, or transfer
anyone.

# Executed evaluation

Synthetic-only results. 625 training / 125 validation / 250 test records; template families do not overlap.

| Approach | Accuracy | Macro precision | Macro recall | Macro F1 | Coverage | ICU FP | ICU FN* |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_baseline | 0.328 | 0.082 | 0.250 | 0.123 | 1.000 | 168 | 0 |
| rules | 0.732 | 1.000 | 0.707 | 0.818 | 0.732 | 0 | 2 |
| decision_tree | 0.976 | 0.975 | 0.974 | 0.974 | 1.000 | 1 | 0 |

*ICU false negatives include withheld suggestions; inspect `icu_fn_abstentions` in metrics.json to distinguish escalation from lower-acuity errors.

Rules selective accuracy among covered requests: 1.000. This excludes withheld requests and must not replace overall accuracy.
Selected tree depth: 6 (validation macro-F1). Critical missing-information precision/recall/F1: 1.000/1.000/1.000.

## Per-route performance

| Approach | Route | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| rules | ICU | 1.000 | 0.976 | 0.988 | 82 |
| rules | telemetry | 1.000 | 0.606 | 0.754 | 71 |
| rules | medical/surgical | 1.000 | 0.558 | 0.716 | 52 |
| rules | specialty review | 1.000 | 0.689 | 0.816 | 45 |
| decision_tree | ICU | 0.988 | 1.000 | 0.994 | 82 |
| decision_tree | telemetry | 0.986 | 0.958 | 0.971 | 71 |
| decision_tree | medical/surgical | 0.926 | 0.962 | 0.943 | 52 |
| decision_tree | specialty review | 1.000 | 0.978 | 0.989 | 45 |

## Interpretation

The rules deliberately withhold lower-acuity suggestions when required referral facts are absent. The tree can infer a label from correlated service and vital-sign patterns, increasing coverage while risking unsupported confidence. Synthetic measurements are strongly scenario-correlated, making the ML task easier than real referrals. Neither system has independent clinician labels, external data, prospective evaluation, or calibrated clinical probabilities. Test diagnostics informed parser bug fixes; this is a development benchmark, not untouched external validation.

Missing-field scoring treats both absence and contradiction as unavailable. Extraction precision counts a wrong value as a false positive; recall also counts it as a false negative. Exact match includes unknowns, so it can look better than value recovery. Subgroup results are descriptive and do not establish fairness.

## Actual failure examples

### SYN-0007 — rules: abstention

Scenario target: **telemetry**. Prediction: **insufficient information**. Missing routing fields: 1.

SYNTHETIC TRAINING CASE. Referrer asks for telemetry bed. Timing: urgent. BP 113/70. Service needed: cardiology. Resp status: nasal cannula. Capacity status: available. Pt is 30 y/o. Needs tele. Currently on norepinephrine. No tertiary specialty assessment needed. Required isolation: none. Pulse 97. Referrer: Synthetic Prairie Hospital. Working diagnosis: heart failure. O2 sat 96%. No vasoactive infusion.

Rules rationale: Essential referral or support facts are unknown; a lower-acuity route cannot be supported.

### SYN-0071 — decision_tree: route mismatch

Scenario target: **medical/surgical**. Prediction: **telemetry**. Missing routing fields: 1.

SYNTHETIC TRAINING CASE. Referrer: Synthetic West Community. Pulse 61. No vasoactive infusion. Service needed: internal medicine. Pt is 25 y/o. O2 sat 99%. Timing: routine. Working diagnosis: pneumonia. Resp status: nasal cannula. BP 116/70. Capacity status: limited. No tertiary specialty assessment needed. Required isolation: contact.

Rules rationale: Specialty or monitoring requirements are unknown; clarify before suggesting a routine route.

### SYN-0975 — decision_tree: ICU false positive

Scenario target: **medical/surgical**. Prediction: **ICU**. Missing routing fields: 3.

SYNTHETIC TRAINING CASE. Timing: urgent. No vasoactive infusion. Working diagnosis: urinary tract infection. Pulse 104. Capacity status: available. Referrer asks for medical/surgical bed. O2 sat 100%. Pt is 22 y/o. Required isolation: contact. Service needed: internal medicine. No tertiary specialty assessment needed. Specialists contacted: internal medicine. If deterioration occurs, consider norepinephrine.

Rules rationale: Essential referral or support facts are unknown; a lower-acuity route cannot be supported.

## Files

See `metrics.json`, `test_predictions.csv`, `failures.csv`, `extraction_metrics.csv`, `extraction_errors.csv`, `subgroups.csv`, and `decision_tree.txt` for complete results.

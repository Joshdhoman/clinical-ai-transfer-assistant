# Transfer Center Decline Prediction & Payer-Equity Audit

A reproducible healthcare-operations analysis: work a synthetic transfer-center
extract from a raw export through data cleaning, a leakage-aware chronological
split, a payer-equity audit, a logistic-regression decline model, and a
cost-based operating-threshold decision.

Built by an RN Clinical Placement Coordinator with 12 years of clinical
experience and an MS in Data Science. This is the second project in this
repository — the first is the [Clinical AI Transfer Assistant](../README.md) at
the repository root. The two are independent: separate packages, separate
dependency files, separate CI jobs, separate deployed apps.

**Synthetic data only. Decision support only. Not validated for clinical use.
Error costs are illustrative planning weights, not measured. Nothing here
describes any real hospital, transfer center, or payer.**

## The question

A transfer center receives requests from other facilities to move patients in.
Each request is accepted or declined. Two operational questions:

1. **Can a decline be flagged at request time**, from information available when
   the call comes in, so a coordinator gets an early warning instead of finding
   out when the bed is needed?
2. **Is the decline decision associated with payer** — and does any raw
   difference survive adjustment for how sick the patients are and how full the
   hospital is?

## What's here

```
src/transfer_decline/
  data.py        load + clean the raw export, derive the target, chronological split
  features.py    request-time feature matrix + train-only encoder (no leakage)
  modeling.py    logistic regression, threshold selection on validation by weighted cost
  equity.py      chi-square tests + covariate-adjusted payer logistic regression
  pipeline.py    end-to-end run -> artifacts/ + reports/
notebooks/       01 data prep · 02 EDA + funnel · 03 features · 04 equity audit · 05 model
app.py           Streamlit demo: score a request, model evaluation, equity audit
tests/           leakage checks, threshold logic, equity method, committed-artifact checks
docs/            model card, data dictionary, methodology, verification
```

## Method, and the two things it gets right on purpose

- **No target leakage.** Every column recorded after the transfer-center
  decision — decision/arrival timestamps, decline reason, length of stay, ICU
  upgrade — is excluded from the feature matrix. `tests/test_data.py` enforces it.
- **Threshold tuned on validation, evaluated on test.** The probability
  threshold that minimises expected weighted error cost (a missed decline costs
  5×, a false alarm 1×) is chosen on the **validation** split and then frozen.
  The **test** split is scored once, at that fixed threshold. An earlier version
  of this analysis picked and evaluated the threshold on the same partition;
  this one does not.
- **Payer is audited, then excluded from the model.** A chi-square test and a
  logistic regression adjusting for acuity, occupancy, ED boarding, service
  line, and level of care. Payer never enters the predictive model regardless of
  the result.

## Results (executed pipeline, synthetic data)

| Split | ROC-AUC | PR-AUC | Precision | Recall | Weighted cost |
|---|---:|---:|---:|---:|---:|
| Test @ operating threshold 0.17 | 0.76 | 0.42 | 0.37 | **0.65** | 956 |
| Test @ default threshold 0.50 | 0.76 | 0.42 | 0.60 | **0.07** | 1577 |

At 0.50 the model catches 7% of declines — useless as an early warning. Moving
the threshold to the validation-selected 0.17 trades precision for recall and
cuts weighted error cost by ~40%. That threshold argument is the point of the
project, not the AUC.

**Payer-equity audit:** chi-square payer × decision p = 0.25; every adjusted
payer odds ratio's 95% CI crosses 1. The audit is null — as it should be, since
the generator assigns payer independently of the decision. The occupancy × decision
check is strongly significant, confirming the method has power on this data.

## Run it

Everything runs from **inside this directory**, not the repository root.

```bash
cd decline-prediction
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -e ".[dev]"
python scripts/build_project.py       # regenerate artifacts/ and reports/
python -m pytest -q
python scripts/execute_notebooks.py   # run all five notebooks in place
streamlit run app.py
```

`app.py` puts its own `src/` on `sys.path`, so the hosted Streamlit demo works
without an install step. `requirements.txt` here lists dependencies explicitly
rather than `.`, because pip would resolve `.` against the repository root and
install the transfer-assistant package instead.

## What this supports, and what it doesn't

**Supports:** a reproducible pipeline, a leakage-aware and time-ordered split, a
defensible operating-threshold argument, and a covariate-adjusted equity-audit
method.

**Does not support:** any claim about real transfer-center performance, real
payer equity, time saved, cost saved, or clinical safety. The relationships in
the data are designed into the synthetic generator; strong synthetic performance
can reflect that design rather than a discovered fact.

---
Synthetic demonstration · not for clinical use

# Verification record

Local verification performed September 9, 2026 on Windows with Python 3.12.4. Exact library versions and dataset hash are in `artifacts/run_manifest.json`; the complete environment is recorded in `requirements-lock.txt`.

- Generated 1,000 synthetic records; all 15 dataset validation checks passed.
- Trained candidates on 625 records and selected depth on 125 validation records. Reported all 250 test records, including abstentions, failures, and subgroup denominators.
- 21 pytest checks passed, including capacity invariance, ICU priority, pediatric abstention, unknown versus explicit negative, conflicting evidence, history/conditional text, reproducibility, feature exclusions, unseen ML input, decision-record validation, duplicate-decision rejection, final coordinator decision, stale-result invalidation, and evaluation/governance rendering.
- Executed all five notebooks with saved outputs; exploratory and model plots are embedded using the inline backend.
- Opened the local Streamlit server in an isolated Chrome session using agent-browser. Confirmed meaningful page content, synthetic acknowledgment, sample analysis, ICU rationale/capacity warning, and evaluation navigation. Browser page-error and console logs were empty during this check.
- Captured and visually inspected actual request-review and evaluation screenshots in `docs/screenshots/`.
- The independent-of-template language diagnostic suite passed 17/20 field assertions across 18 authored cases. The three failures are retained in `reports/challenge_results.csv`; they are not hidden by the passing regression suite.

This verification covers local prototype behavior. GitHub Actions is configured but has not run remotely. No clinical, external-dataset, prospective, production deployment, performance/load, security, accessibility-conformance, or authenticated audit certification was performed. Screenshots show only synthetic examples.

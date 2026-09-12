# Model card

## Model and status

**Clinical AI Workflow Assistant for Patient Transfer Operations, version 0.1.0.** Local portfolio prototype; synthetic-only adult English-language referrals. No Rush affiliation or endorsement. Components: rule-based extractor, deterministic routing rules, and a class-weighted decision tree selected by validation macro-F1. See [executed results](../reports/RESULTS.md), [model selection](../artifacts/model_selection.json), and [run manifest](../artifacts/run_manifest.json).

## Intended use

Demonstrate how a clinical placement coordinator can translate narrative referral intake into structured operational information, clarification questions, a provisional routing pathway, and documented human review. Intended users are portfolio reviewers, analysts, and clinicians exploring workflow design with fictional cases.

## Inappropriate uses

Do not use for patient triage, acceptance or denial of transfers, final level-of-care assignment, treatment selection, transport decisions, bed allocation, or time-critical clinical decisions. Do not use with real referrals, pediatric patients, non-English notes, or as a substitute for direct clinician communication. No accuracy claim authorizes clinical use.

## Data, targets, and methodology

1,000 seeded fictional adult scenarios; 16 scenario types; eight prose families; five fictional facilities. Scenario labels are assigned before rendering notes, independently of the routing code, but encode similar simplified support assumptions. Complete source facts are separate from note-visible labels, omissions, and conflicts. Train/validation/test counts are 625/125/250 with disjoint template families. The held-out families still share vocabulary and a generator with development data. Parser bug fixes were informed by development test diagnostics; this benchmark is not a pristine external test.

The tree uses extracted oxygen, pressor, monitoring, specialty, service, SBP, heart rate, and SpO2 fields. It excludes identifiers, source truth, scenario ID, requested LOC, outcome, facility, capacity, and age. Numerical imputation and one-hot encoding fit on training data only. Candidate depths 3–6 are selected on validation macro-F1; minimum leaf size is 12. The selected estimator remains fitted on training data only. No paid API or LLM is involved.

## Evaluation and limitations

Actual accuracy, macro and class-specific precision/recall/F1, confusion matrices, ICU errors, missing-information results, extraction errors, subgroup diagnostics, and examples are saved in `reports/`. Abstentions count as misses in overall accuracy. Selective accuracy is reported with coverage. ICU false negatives include abstentions and are disaggregated in JSON; a withheld suggestion is operationally different from a lower-acuity recommendation.

The simulation is intentionally small and simplified. Template wording, scenario-associated vitals, and service categories make prediction easier than authentic referrals. Rules and generated labels share assumptions, so high agreement can be circular. A perfect generated-field score would indicate compatibility with supported templates, not broad NLP understanding. The challenge set exposes parser limits and is a regression/development suite, not a validation cohort.

Unsupported abbreviations, temporality, negation scope, multi-device transitions, complex conditions, decimal measurements, copied-forward text, unfamiliar services, conflicting vitals, and clinical deterioration may be mishandled or left unknown. The system cannot determine whether the referral describes current stability. The tracked checklist omits many clinical and operational requirements, including mental status, trends, medications/doses, interventions, transport capability, goals of care, and accepting physician identity.

## Explainability and confidence

The extractor exposes matched text, normalized values, and contradictions. The router returns rule IDs and plain-language rationale. Tree output includes its actual feature decision path. Confidence is a qualitative documentation-support label; the percentage shown is the fraction of routing fields available. It is **not calibrated probability, clinical certainty, or likelihood of a successful transfer**. Tree leaf proportions are class-weighted and uncalibrated. There is no clinically validated confidence threshold.

Prototype support rules send pressors or advanced respiratory support to ICU review, otherwise check specialty and monitoring needs before a routine suggestion. This is not a universal clinical standard: actual HFNC/NIV placement, telemetry criteria, and service acceptance depend on local policies, trends, staffing, and clinician assessment. Specialty review is an acceptance/escalation pathway, not a level of care. The system surfaces low vital readings for human review without claiming a complete acuity assessment.

## Human oversight

All suggestions require source verification by a qualified clinician. Unknowns remain unknown; essential missing facts can trigger abstention. Capacity never downgrades clinical routing. The interface highlights disagreement with the requested unit and the tree. A coordinator makes the final decision by accepting the suggested route or choosing a different route, with a rationale and explicit review acknowledgment. Editing the referral invalidates the previous analysis; duplicate decision submissions are rejected within the local process. Recording a final decision does not accept, schedule, dispatch, or transfer anyone.

## Potential bias

Scenario prevalence, adult-only sampling, English-only text, age distribution, and uniform physiology within scenario families do not represent real populations. One fictional facility has increased documentation omissions to demonstrate referral-source bias. Age and facility are excluded from prediction, but correlated service and physiology can still produce disparities. `subgroups.csv` reports age-band, source, template, and completeness slices with denominators; small synthetic groups cannot establish fairness. No race, ethnicity, sex, disability, socioeconomic, or language fairness validation has been performed. Excluding these attributes is not evidence of equitable performance.

## PHI, privacy, and security

All committed records and examples are generated or hand-authored fiction, with no real patient source. The app is bound to localhost by default and has no clinical integration or external inference call. Users must confirm synthetic input. That acknowledgment is not a PHI detector, de-identification process, or security control. Raw notes remain in server session memory while the session is active. Browser extensions, hosting, crash dumps, or machine access could still expose entered text. Do not enter real patient information.

The local decision record omits raw notes and extracted values but stores a note hash, routing, fictional reviewer name, and free-text rationale. Hashing is not anonymization; a rationale can contain identifiers if misused. Audit files are gitignored. Model artifacts use joblib: load only locally generated trusted artifacts because pickle-based formats can execute code. Public hosting would require security design beyond this project.

## Audit logging

Append-only-by-convention local JSONL decision records contain a UUID, analysis ID, UTC time, note SHA-256, extractor/router versions, suggested and final routes, rationale, fictional reviewer name, route-change flag, completeness, and rule IDs. A process-local lock avoids concurrent thread writes and duplicate analysis submissions. There is no authenticated identity, encryption, immutable retention, cross-process transaction guarantee, legal audit trail, or tamper evidence. Production would need access controls, retention governance, secure storage, and monitored record integrity.

## Model drift and monitoring proposal

Monitor input field availability, unrecognized vocabulary, conflict rates, route mix, abstention, rule/model disagreement, clinician override reasons, and performance on adjudicated samples. Compare by facility, demographic group, service, and shift with adequate sample sizes. Separately track documentation change versus clinical case-mix change. Drift detection must trigger investigation; it must not automatically retrain or promote models. No production monitoring is implemented here.

## Validation before real clinical use

1. Map the institution's intake workflow, acceptance authority, supported services, capacity rules, and escalation paths with clinicians and operations leaders.
2. Obtain privacy, security, legal/regulatory, clinical governance, and institutional review as applicable; establish an approved data environment and intended-use boundary.
3. Assemble representative, appropriately authorized referrals with independent clinician adjudication, disagreements resolved by a documented process. Lock external/time-based test cohorts and prespecify acceptable failure rates.
4. Evaluate extraction, negation, missingness, calibration, ICU under-routing, abstention, subgroup performance, temporal changes, and operational burden with confidence intervals and adequate sample sizes.
5. Run silent prospective testing, usability studies, and failure simulations before any controlled decision-support pilot. Establish clear fallback procedures, monitoring ownership, version control, and stop criteria.
6. Measure clarification burden, time to complete referral, avoidable callbacks, user comprehension, override behavior, and placement safety. This project makes no measured efficiency, safety, or cost-saving claim.

## Governance references

[FDA Clinical Decision Support Software guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software) informs the emphasis on independently reviewable recommendations; it does not establish this prototype's device status or clinical validity. [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) informs documenting risks, measurement, oversight, and monitoring. Neither source supplies the simulation's routing criteria.

# Data dictionary and synthetic generation

`data/synthetic/transfer_requests.csv` contains exactly 1,000 fictional records generated with seed 20260909. One row is one referral; no longitudinal patient identity exists. All source facts are complete; note omissions and conflicts are preserved separately. The source columns must not be passed to inference in place of extracted values.

| Field | Type / values | Meaning |
|---|---|---|
| request_id | SYN-0001 … SYN-1000 | Fictional referral identifier |
| synthetic | boolean, always true | Provenance marker |
| scenario_id | integer, 0–15 | Generation scenario; excluded from model |
| template_family | integer, 0–7 | Prose family for leakage-aware split |
| split | train / validation / test | Families 0–4 / 5 / 6–7 |
| patient_age | integer, 18–95 | Age in completed fictional scenario |
| presenting_problem | string | One of 16 scenario presentations |
| requested_service | categorical | critical care, pulmonology, cardiology, internal medicine, orthopedics, transplant, neurosurgery |
| requested_level_of_care | categorical | ICU, telemetry, medical/surgical, specialty review; may disagree with target |
| oxygen_support | categorical | room air, nasal cannula, high-flow nasal cannula, noninvasive ventilation, mechanical ventilation |
| vasopressor_use | boolean | Current fictional support; no dose modeled |
| isolation_requirements | categorical | none, contact, droplet, airborne |
| relevant_consultants | string | Scenario-associated consultant service |
| referring_facility | string | One of five explicitly fictional facilities |
| transfer_urgency | categorical | emergent, urgent, routine |
| bed_availability | categorical | available, limited, unavailable; static fictional snapshot |
| systolic_bp | integer, mmHg | Simulated single measurement; not a trend |
| heart_rate | integer, bpm | Simulated single measurement |
| spo2 | integer, percent | Simulated saturation |
| monitoring_required | boolean | Explicit simulated cardiac-monitoring requirement |
| specialty_need | boolean | Explicit need for specialty acceptance/review |
| transfer_outcome | categorical | accepted, pending clarification, capacity delay, redirected; simulated operational result, not prediction target |
| routing_target | categorical | Authored scenario target; not independent clinical adjudication |
| free_text_transfer_note | string | Shuffled prose clauses with varied framing, omissions, occasional conflicts, history, and conditional statements |
| visible_annotations | JSON object | Field values available in the note; null if omitted or contradictory |
| omitted_fields | JSON array | Fields intentionally absent; distinct from explicit negatives |
| conflict_fields | JSON array | Deliberately contradictory fields requiring clarification |

Each field is omitted with probability 0.065, or 0.12 at Synthetic West Community. Explicit pressor contradictions occur with probability 0.035 when the field is present. Historical ventilation and conditional pressor phrases occur with probabilities 0.15 and 0.10. A requested-route alternative is sampled with probability 0.16 (it can match by chance). These are simulation parameters, not observed healthcare statistics. Outcomes are correlated with capacity by design. Vitals and support correlate strongly with scenario labels; this intentionally simple design inflates predictability.

Validation checks row count, uniqueness, complete source truth, ranges, route vocabulary, fictional source markers, annotation lineage, all classes in every split, and absence of template-family overlap. It does not prove clinical realism or privacy of arbitrary user-entered text.

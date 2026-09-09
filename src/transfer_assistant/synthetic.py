"""Seeded fictional adult transfers. No patient records or external data are used.

Scenario targets are authored independently of the extraction/router implementation.
They are simulation assumptions, not clinical ground truth or hospital policy.
"""

import json
import random
from pathlib import Path

import pandas as pd

from .schema import FIELDS, ROUTES

SEED = 20260909
# problem, service, oxygen, pressors, monitoring, specialty, target
SCENARIOS = [
    ("septic shock", "critical care", "nasal cannula", True, True, False, "ICU"),
    ("respiratory failure", "pulmonology", "mechanical ventilation", False, True, False, "ICU"),
    ("hypoxemic pneumonia", "pulmonology", "high-flow nasal cannula", False, True, False, "ICU"),
    ("COPD exacerbation", "pulmonology", "noninvasive ventilation", False, True, False, "ICU"),
    ("cardiogenic shock", "cardiology", "nasal cannula", True, True, False, "ICU"),
    ("atrial fibrillation", "cardiology", "room air", False, True, False, "telemetry"),
    ("chest pain", "cardiology", "room air", False, True, False, "telemetry"),
    ("syncope", "internal medicine", "room air", False, True, False, "telemetry"),
    ("heart failure", "cardiology", "nasal cannula", False, True, False, "telemetry"),
    ("cellulitis", "internal medicine", "room air", False, False, False, "medical/surgical"),
    ("urinary tract infection", "internal medicine", "room air", False, False, False, "medical/surgical"),
    ("dehydration", "internal medicine", "room air", False, False, False, "medical/surgical"),
    ("pneumonia", "internal medicine", "nasal cannula", False, False, False, "medical/surgical"),
    ("complex fracture", "orthopedics", "room air", False, False, True, "specialty review"),
    ("transplant complication", "transplant", "room air", False, False, True, "specialty review"),
    ("intracranial lesion", "neurosurgery", "room air", False, True, True, "specialty review"),
]
FACILITIES = ["Synthetic North Hospital", "Synthetic Valley ED", "Synthetic Prairie Hospital",
              "Synthetic Lakeside Center", "Synthetic West Community"]


def render_field(name: str, value, family: int) -> str:
    """Eight fixed prose families; test families use different framing/abbreviations."""
    variants = {
        "patient_age": [f"Age: {value}", f"{value}-year-old adult", f"Patient age {value}", f"Adult aged {value}", f"{value} yo adult", f"Age is {value} years", f"Pt is {value} y/o", f"{value} years old"],
        "presenting_problem": [f"Presenting problem: {value}", f"Presentation: {value}", f"Reason for transfer: {value}", f"Primary concern: {value}", f"Problem: {value}", f"Transfer diagnosis: {value}", f"Working diagnosis: {value}", f"Chief concern: {value}"],
        "requested_service": [f"Requested service: {value}", f"Service requested: {value}", f"Seeking {value} service", f"Service: {value}", f"Requesting {value}", f"Receiving service: {value}", f"Service needed: {value}", f"Referral to {value}"],
        "requested_level_of_care": [f"Requested level of care: {value}", f"Requested bed: {value}", f"Sending team requests {value} bed", f"LOC requested: {value}", f"Requested unit: {value}", f"Bed request: {value}", f"Referrer asks for {value} bed", f"Requested placement: {value}"],
        "oxygen_support": [f"Oxygen support: {value}", f"Currently on {value}", f"Respiratory support: {value}", f"O2: {value}", f"Current oxygen: {value}", f"Support is {value}", f"Resp status: {value}", f"Oxygen device: {value}"],
        "vasopressor_use": (["Vasopressor use: yes", "Currently on norepinephrine", "Pressors running", "Receiving levophed", "On vasopressors", "Vasopressor infusion active", "Norepinephrine infusing", "Pressor support required"] if value else ["Vasopressor use: no", "No vasopressors", "Not on pressors", "Off norepinephrine", "No pressor support", "Vasopressors not required", "No vasoactive infusion", "Pressor-free currently"]),
        "isolation_requirements": [f"Isolation: {value}", f"Precautions: {value}", f"Isolation requirements: {value}", f"Isolation status: {value}", f"Precaution type: {value}", f"Infection precautions: {value}", f"Required isolation: {value}", f"Isolation category: {value}"],
        "relevant_consultants": [f"Consultants: {value}", f"Consults: {value}", f"Relevant consultants: {value}", f"Consultant: {value}", f"Consult requested: {value}", f"Consulting team: {value}", f"Specialists contacted: {value}", f"Consult team: {value}"],
        "referring_facility": [f"Referring facility: {value}", f"From {value}", f"Sending facility: {value}", f"Referral source: {value}", f"Origin: {value}", f"Outside hospital: {value}", f"Referrer: {value}", f"Facility of origin: {value}"],
        "transfer_urgency": [f"Urgency: {value}", f"Transfer urgency: {value}", f"Priority: {value}", f"Transfer priority: {value}", f"Request urgency: {value}", f"Urgency level: {value}", f"Timing: {value}", f"Priority category: {value}"],
        "bed_availability": [f"Bed availability: {value}", f"Capacity: {value}", f"Bed status: {value}", f"Receiving capacity: {value}", f"Bed supply: {value}", f"Placement availability: {value}", f"Capacity status: {value}", f"Available beds: {value}"],
        "systolic_bp": [f"SBP {value} mmHg"] * 6 + [f"BP {value}/70", f"Systolic BP: {value}"],
        "heart_rate": [f"HR {value} bpm"] * 6 + [f"Pulse {value}", f"Heart rate: {value}"],
        "spo2": [f"SpO2 {value}%"] * 6 + [f"O2 sat {value}%", f"Saturation: {value}%"],
        "monitoring_required": (["Continuous cardiac monitoring required", "Telemetry required", "Needs cardiac monitoring", "Monitoring required: yes", "Continuous monitoring needed", "Cardiac monitoring indicated", "Needs tele", "Rhythm monitoring needed"] if value else ["No continuous monitoring required", "Telemetry not required", "No cardiac monitoring needed", "Monitoring required: no", "Continuous monitoring not needed", "Cardiac monitoring not indicated", "No tele needed", "Rhythm monitoring not needed"]),
        "specialty_need": (["Specialty evaluation required", "Needs specialty review", "Specialty need: yes", "Specialty review required", "Specialist acceptance needed", "Specialty evaluation pending", "Tertiary specialty assessment needed", "Specialty review is required"] if value else ["No specialty evaluation required", "Specialty review not needed", "Specialty need: no", "No specialty review required", "Specialist acceptance not needed", "Specialty evaluation not indicated", "No tertiary specialty assessment needed", "Specialty review is not required"]),
    }
    return variants[name][family]


def generate_dataset(n: int = 1000, seed: int = SEED) -> pd.DataFrame:
    if n < 80:
        raise ValueError("Use at least 80 records to represent all template families.")
    rng = random.Random(seed)
    records = []
    for i in range(n):
        problem, service, oxygen, pressors, monitor, specialty, target = rng.choice(SCENARIOS)
        family = i % 8
        requested = target if rng.random() > .16 else rng.choice(ROUTES)
        facts = dict(zip(FIELDS, [
            rng.randint(18, 95), problem, service, requested, oxygen, pressors,
            rng.choices(["none", "contact", "droplet", "airborne"], [65, 18, 12, 5])[0],
            service, rng.choice(FACILITIES),
            "emergent" if target == "ICU" else rng.choice(["urgent", "routine"]),
            rng.choices(["available", "limited", "unavailable"], [45, 35, 20])[0],
            rng.randint(78, 106) if pressors else rng.randint(108, 159),
            rng.randint(95, 135) if target == "ICU" else rng.randint(60, 110),
            rng.randint(88, 96) if target == "ICU" else rng.randint(94, 100), monitor, specialty,
        ]))
        visible = {}
        clauses = []
        omissions = []
        for name, value in facts.items():
            # One source has more incomplete referrals: simulated documentation bias.
            probability = .12 if facts["referring_facility"] == FACILITIES[-1] else .065
            if rng.random() < probability:
                visible[name] = None
                omissions.append(name)
            else:
                visible[name] = value
                clauses.append(render_field(name, value, family))
        # Adversarial ambiguity is annotated separately from genuinely absent fields.
        conflict_fields = []
        if rng.random() < .035 and visible["vasopressor_use"] is not None:
            clauses.append("No vasopressors" if pressors else "Currently on norepinephrine")
            visible["vasopressor_use"] = None
            conflict_fields.append("vasopressor_use")
        if rng.random() < .15:
            clauses.append("History of prior mechanical ventilation")
        if rng.random() < .10:
            clauses.append("If deterioration occurs, consider norepinephrine")
        rng.shuffle(clauses)
        note = "SYNTHETIC TRAINING CASE. " + ". ".join(clauses) + "."
        outcome = rng.choices(["accepted", "pending clarification", "capacity delay", "redirected"],
                              [20, 20, 55, 5] if facts["bed_availability"] == "unavailable" else [65, 20, 10, 5])[0]
        records.append({"request_id": f"SYN-{i+1:04d}", "synthetic": True,
                        "scenario_id": SCENARIOS.index((problem, service, oxygen, pressors, monitor, specialty, target)),
                        "template_family": family,
                        "split": "train" if family < 5 else "validation" if family == 5 else "test",
                        **facts, "transfer_outcome": outcome, "routing_target": target,
                        "free_text_transfer_note": note, "visible_annotations": json.dumps(visible, sort_keys=True),
                        "omitted_fields": json.dumps(omissions), "conflict_fields": json.dumps(conflict_fields)})
    return pd.DataFrame(records)


def validate_dataset(df: pd.DataFrame, expected_rows: int = 1000) -> dict:
    """Fail fast on schema, ranges, lineage, duplicate notes, or split overlap."""
    checks = {
        "row_count": len(df) == expected_rows,
        "unique_ids": df.request_id.is_unique,
        "unique_notes": df.free_text_transfer_note.is_unique,
        "all_synthetic": bool(df.synthetic.eq(True).all()),
        "age_range": bool(df.patient_age.between(18, 95).all()),
        "vital_ranges": bool((df.systolic_bp.between(70, 200) & df.heart_rate.between(40, 180) & df.spo2.between(70, 100)).all()),
        "required_columns": set(FIELDS).issubset(df.columns),
        "all_routes": set(df.routing_target) == set(ROUTES),
        "fictional_facilities": bool(df.referring_facility.isin(FACILITIES).all()),
        "synthetic_note_marker": bool(df.free_text_transfer_note.str.startswith("SYNTHETIC TRAINING CASE.").all()),
        "no_template_overlap": int(df.groupby("template_family").split.nunique().max()) == 1,
        "complete_source_truth": not bool(df[FIELDS].isna().any().any()),
        "annotation_keys": all(set(json.loads(s)) == set(FIELDS) for s in df.visible_annotations),
        "annotation_lineage": all(
            all((v is None if k in json.loads(row.omitted_fields) + json.loads(row.conflict_fields)
                 else v == getattr(row, k)) for k, v in json.loads(row.visible_annotations).items())
            for row in df.itertuples()),
        "routes_in_each_split": all(set(g.routing_target) == set(ROUTES) for _, g in df.groupby("split")),
    }
    report = {"passed": all(checks.values()), "checks": checks, "rows": len(df),
              "seed": SEED, "class_counts": df.routing_target.value_counts().to_dict(),
              "split_counts": df.split.value_counts().to_dict(),
              "records_with_omissions": int(df.omitted_fields.ne("[]").sum()),
              "records_with_conflicts": int(df.conflict_fields.ne("[]").sum())}
    if not report["passed"]:
        raise ValueError(f"Validation failed: {report}")
    return report


def write_dataset(root: Path, n: int = 1000) -> pd.DataFrame:
    output = root / "data" / "synthetic"
    output.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(n)
    report = validate_dataset(df, n)
    df.to_csv(output / "transfer_requests.csv", index=False, lineterminator="\n")
    (output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return df

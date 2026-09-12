"""Transparent simulation rules, not validated level-of-care criteria."""

from .schema import ABSTAIN, CRITICAL, Extraction, Recommendation, missing_fields

VERSION = "routing-rules-1.0"
ADVANCED_SUPPORT = {"mechanical ventilation", "noninvasive ventilation", "high-flow nasal cannula"}


def recommend(result: Extraction) -> Recommendation:
    v = result.values
    missing = missing_fields(result)
    completeness = (len(CRITICAL) - len(missing["critical"])) / len(CRITICAL)
    warnings = ["Prototype assumptions only; a qualified clinician must verify acuity and destination."]
    if missing["critical"]:
        warnings.append("Clarify routing fields: " + ", ".join(missing["critical"]) + ".")
    if result.conflicts:
        warnings.append("Conflicting statements require source verification: " + ", ".join(result.conflicts) + ".")
    if v["bed_availability"] in {"limited", "unavailable"}:
        warnings.append("Capacity constraint: coordinate an appropriate bed; capacity does not lower the suggested level of care.")
    if v["isolation_requirements"] in {"contact", "droplet", "airborne"}:
        warnings.append("Confirm isolation-compatible placement and receiving-facility precautions.")
    # Vitals are surfaced for clinician review, not used as validated triage thresholds.
    if v["systolic_bp"] is not None and v["systolic_bp"] < 90:
        warnings.append("SBP below 90 in this note: flag for immediate human acuity review; the prototype cannot assess stability.")
    if v["spo2"] is not None and v["spo2"] < 90:
        warnings.append("SpO2 below 90 in this note: flag for immediate human acuity review; verify context and current support.")
    reasons, rule_ids = [], []
    if v["patient_age"] is not None and v["patient_age"] < 18:
        route = ABSTAIN
        reasons.append("Pediatric requests are outside the adult simulation scope.")
        rule_ids.append("R00")
    elif v["vasopressor_use"] is True or v["oxygen_support"] in ADVANCED_SUPPORT:
        route = "ICU"
        if v["vasopressor_use"] is True:
            reasons.append("Active vasopressor support triggers the prototype ICU review pathway.")
            rule_ids.append("R01")
        if v["oxygen_support"] in ADVANCED_SUPPORT:
            reasons.append(f"{v['oxygen_support'].capitalize()} triggers the prototype ICU review pathway.")
            rule_ids.append("R02")
    elif any(v[x] is None for x in ["patient_age", "presenting_problem", "requested_service", "oxygen_support", "vasopressor_use"]):
        route = ABSTAIN
        reasons.append("Essential referral or support facts are unknown; a lower-acuity route cannot be supported.")
        rule_ids.append("R03")
    elif v["specialty_need"] is True:
        route = "specialty review"
        reasons.append("An explicit specialty evaluation requirement needs service acceptance and acuity review.")
        rule_ids.append("R04")
    elif v["specialty_need"] is None or v["monitoring_required"] is None:
        route = ABSTAIN
        reasons.append("Specialty or monitoring requirements are unknown; clarify before suggesting a routine route.")
        rule_ids.append("R05")
    elif v["monitoring_required"] is True:
        route = "telemetry"
        reasons.append("Explicit continuous monitoring need with no documented pressors or advanced respiratory support.")
        rule_ids.append("R06")
    else:
        route = "medical/surgical"
        reasons.append("Explicit negatives for pressors, monitoring, and specialty need; documented support is room air or nasal cannula.")
        rule_ids.append("R07")
    if v["specialty_need"] is True and route == "ICU":
        warnings.append("Specialty review is also required; the ICU pathway does not replace service consultation.")
    if v["requested_level_of_care"] and route not in {ABSTAIN, v["requested_level_of_care"]}:
        warnings.append("Suggestion differs from the referring team's requested level of care; reconcile with a clinician.")
    confidence = "Withheld" if route == ABSTAIN else "Limited" if missing["critical"] or result.conflicts else "Supported by documented fields"
    return Recommendation(route, reasons, rule_ids, warnings, completeness, confidence)

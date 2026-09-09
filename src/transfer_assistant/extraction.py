"""Deterministic clause-level NLP with explicit negatives and evidence spans.

Deliberately bounded English vocabulary. Unrecognized, historical, conditional,
and contradictory statements are never silently converted into normal findings.
"""

import re

from .schema import Extraction

# Fields with a label followed by a bounded value. Unknown labels stay unparsed.
LABELS = {
    "presenting_problem": r"presenting problem|presentation|reason for transfer|primary concern|problem|transfer diagnosis|working diagnosis|chief concern",
    "requested_service": r"requested service|service requested|service|receiving service|service needed|referral to|requesting|seeking",
    "requested_level_of_care": r"requested level of care|requested bed|loc requested|requested unit|bed request|requested placement|sending team requests|referrer asks for",
    "isolation_requirements": r"isolation requirements|isolation status|isolation category|required isolation|isolation|precautions|precaution type|infection precautions",
    "relevant_consultants": r"relevant consultants|consultants|consults|consultant|consult requested|consulting team|specialists contacted|consult team",
    "referring_facility": r"referring facility|sending facility|referral source|outside hospital|facility of origin|referrer|origin|from",
    "transfer_urgency": r"transfer urgency|transfer priority|request urgency|urgency level|urgency|priority category|priority|timing",
    "bed_availability": r"bed availability|bed status|receiving capacity|bed supply|placement availability|capacity status|available beds|capacity",
}
ENUMS = {
    "requested_service": {"critical care", "pulmonology", "cardiology", "internal medicine", "orthopedics", "transplant", "neurosurgery"},
    "requested_level_of_care": {"icu", "telemetry", "medical/surgical", "specialty review"},
    "isolation_requirements": {"none", "contact", "droplet", "airborne"},
    "transfer_urgency": {"emergent", "urgent", "routine"},
    "bed_availability": {"available", "limited", "unavailable"},
}
OXYGEN = {
    "mechanical ventilation": r"mechanical ventilation|mechanically ventilated|intubated|on ventilator",
    "noninvasive ventilation": r"noninvasive ventilation|non-invasive ventilation|bipap|cpap",
    "high-flow nasal cannula": r"high-flow nasal cannula|high flow nasal cannula|hfnc",
    "nasal cannula": r"nasal cannula|\d+\s*(?:l|liters?)(?:/min)?\s*(?:nc|oxygen)|\bnc\b",
    "room air": r"room air|\bra\b",
}
NEGATIVE_PRESSOR = r"\b(?:no|not on|off)\s+(?:vasopressors?|pressors?|pressor support|norepinephrine|levophed|vasoactive infusion)\b|vasopressor use:\s*no|vasopressors not required|pressor-free currently"
POSITIVE_PRESSOR = r"vasopressor use:\s*yes|(?:currently on|receiving|on)\s+(?:norepinephrine|levophed|vasopressors?)|pressors running|vasopressor infusion active|norepinephrine infusing|pressor support required"
NEGATIVE_MONITOR = r"no (?:continuous monitoring required|cardiac monitoring needed|tele needed)|(?:telemetry|continuous monitoring|cardiac monitoring|rhythm monitoring) (?:not required|not needed|not indicated)|monitoring required:\s*no"
POSITIVE_MONITOR = r"continuous cardiac monitoring required|telemetry required|needs cardiac monitoring|monitoring required:\s*yes|continuous monitoring needed|cardiac monitoring indicated|needs tele\b|rhythm monitoring needed"
NEGATIVE_SPECIALTY = r"no (?:specialty evaluation required|specialty review required|tertiary specialty assessment needed)|specialty (?:review|evaluation) (?:not needed|not indicated|is not required)|specialty need:\s*no|specialist acceptance not needed"
POSITIVE_SPECIALTY = r"specialty evaluation required|needs specialty review|specialty need:\s*yes|specialty review required|specialist acceptance needed|specialty evaluation pending|tertiary specialty assessment needed|specialty review is required"


class RuleBasedExtractor:
    version = "rules-nlp-1.0"

    def extract(self, note: str) -> Extraction:
        if not isinstance(note, str) or not note.strip():
            raise ValueError("Paste a nonempty synthetic transfer note.")
        if len(note) > 12000:
            raise ValueError("This demo accepts notes up to 12,000 characters.")
        result = Extraction(version=self.version)
        candidates: dict[str, list] = {}

        def add(name, value, clause):
            candidates.setdefault(name, []).append(value)
            result.evidence.setdefault(name, []).append(clause.strip())

        # Preserve decimal numbers; segment on punctuation and contrast/conjunction.
        clauses = re.split(r"\.(?!\d)|[;\n]|\s+but\s+|,\s*|\s+and\s+", note, flags=re.I)
        for raw in clauses:
            clause = raw.strip()
            lower = clause.lower()
            if not lower or re.search(r"\b(history of|previously|prior|if |consider|may require|might need|yesterday)\b", lower):
                continue
            age = re.search(r"\b(?:age(?: is)?[: ]+|patient age\s+|adult aged\s+)(\d{1,3})\b|\b(\d{1,3})(?:-year-old|\s+yo\b|\s+y/o\b|\s+years old)", lower)
            if age:
                value = int(age.group(1) or age.group(2))
                if 0 <= value <= 120:
                    add("patient_age", value, clause)
            for name, labels in LABELS.items():
                # Prefer longest labels ("service needed" before "service").
                labels = "|".join(sorted(labels.split("|"), key=len, reverse=True))
                match = re.fullmatch(rf"(?:{labels})\s*:?\s+(.+?)", clause, flags=re.I)
                if not match:
                    continue
                value = match.group(1).strip().lower()
                if name == "requested_service":
                    value = re.sub(r"\s+service$", "", value)
                if name == "requested_level_of_care":
                    value = re.sub(r"\s+bed$", "", value)
                if value in {"unknown", "not reported", "pending", "unspecified", "n/a", "tbd"}:
                    continue
                if name in ENUMS and value not in ENUMS[name]:
                    continue
                if name == "requested_level_of_care" and value == "icu":
                    value = "ICU"
                if name in {"referring_facility", "presenting_problem"}:
                    value = match.group(1).strip()
                add(name, value, clause)
                break  # A labeled clause describes one field; avoid prefix collisions.
            oxygen_clause = lower
            # Avoid equating negated oxygen device mentions with active support.
            for category, pattern in OXYGEN.items():
                for match in re.finditer(pattern, oxygen_clause):
                    prefix = oxygen_clause[max(0, match.start()-30):match.start()]
                    if re.search(r"\b(?:no|not|off|without|denies)\b", prefix):
                        continue
                    # High-flow contains 'nasal cannula', but is one device.
                    if category == "nasal cannula" and re.search(r"high[- ]flow", oxygen_clause):
                        continue
                    add("oxygen_support", category, clause)
            for name, neg, pos in [
                ("vasopressor_use", NEGATIVE_PRESSOR, POSITIVE_PRESSOR),
                ("monitoring_required", NEGATIVE_MONITOR, POSITIVE_MONITOR),
                ("specialty_need", NEGATIVE_SPECIALTY, POSITIVE_SPECIALTY),
            ]:
                negative = list(re.finditer(neg, lower))
                if negative:
                    add(name, False, clause)
                scrubbed = re.sub(neg, "", lower)
                if re.search(pos, scrubbed):
                    add(name, True, clause)
            for name, pattern, limits in [
                ("systolic_bp", r"\b(?:sbp|systolic bp)\s*:?\s*(\d{2,3})\b|\bbp\s+(\d{2,3})/\d{2,3}", (40, 260)),
                ("heart_rate", r"\b(?:hr|pulse|heart rate)\s*:?\s*(\d{2,3})\b", (20, 250)),
                ("spo2", r"\b(?:spo2|o2 sat|saturation)\s*:?\s*(\d{2,3})\s*%", (40, 100)),
            ]:
                for match in re.finditer(pattern, lower):
                    value = int(next(x for x in match.groups() if x is not None))
                    if limits[0] <= value <= limits[1]:
                        add(name, value, clause)
        for name, values in candidates.items():
            unique = set(values)
            if len(unique) == 1:
                result.values[name] = values[0]
            else:
                result.conflicts.append(name)
        return result

"""Hand-authored boundary cases; failures are reported, not hidden or relabeled."""
import json
from pathlib import Path

import pandas as pd

from transfer_assistant.extraction import RuleBasedExtractor

CASES = [
    ("explicit negative", "No vasopressors.", {"vasopressor_use": False}),
    ("absent support", "Age: 61.", {"vasopressor_use": None, "oxygen_support": None}),
    ("negated ventilation", "Not intubated. Currently on room air.", {"oxygen_support": "room air"}),
    ("historical ventilation", "History of prior mechanical ventilation.", {"oxygen_support": None}),
    ("conditional support", "If deterioration occurs, consider norepinephrine.", {"vasopressor_use": None}),
    ("conflicting pressors", "No vasopressors. Currently on norepinephrine.", {"vasopressor_use": None}),
    ("HFNC abbreviation", "Currently on HFNC.", {"oxygen_support": "high-flow nasal cannula"}),
    ("NIV abbreviation", "On BiPAP.", {"oxygen_support": "noninvasive ventilation"}),
    ("numeric boundaries", "Age: 54. Service: cardiology. HR 95. Isolation: none.", {"requested_service": "cardiology", "isolation_requirements": "none"}),
    ("unknown isolation", "Isolation: unknown.", {"isolation_requirements": None}),
    ("device contradiction", "Currently on room air. Intubated.", {"oxygen_support": None}),
    ("vital contradiction", "SBP 80 mmHg. SBP 120 mmHg.", {"systolic_bp": None}),
    # These expectations are linguistically authored; current narrow regexes may fail.
    ("abbreviation outside lexicon", "Levo gtt at 0.05 mcg/kg/min.", {"vasopressor_use": True}),
    ("resolved device transition", "Extubated this morning, now on 2L NC.", {"oxygen_support": "nasal cannula"}),
    ("negation across conjunction", "No norepinephrine or vasopressin running.", {"vasopressor_use": False}),
    ("hypothetical device", "May need BiPAP overnight.", {"oxygen_support": None}),
    ("temporal oxygen transition", "Previously on BiPAP, currently on room air.", {"oxygen_support": "room air"}),
    ("unsupported isolation synonym", "Enteric precautions required.", {"isolation_requirements": "contact"}),
]


def run(root: Path):
    extractor = RuleBasedExtractor()
    rows = []
    for name, note, expected in CASES:
        actual = extractor.extract(note)
        for field, value in expected.items():
            rows.append({"case": name, "note": note, "field": field, "expected": value,
                         "actual": actual.values[field], "passed": actual.values[field] == value})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "reports/challenge_results.csv", index=False)
    summary = {"cases": len(CASES), "field_assertions": len(frame), "passed": int(frame.passed.sum()),
               "failed": int((~frame.passed).sum()), "purpose": "Authored diagnostic regression cases, not clinical validation."}
    (root / "reports/challenge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])

"""Shared extraction contract; unknown values remain None, never implied negatives."""

from dataclasses import dataclass, field
from typing import Any, Protocol

ROUTES = ["ICU", "telemetry", "medical/surgical", "specialty review"]
ABSTAIN = "insufficient information"
FIELDS = [
    "patient_age", "presenting_problem", "requested_service", "requested_level_of_care",
    "oxygen_support", "vasopressor_use", "isolation_requirements", "relevant_consultants",
    "referring_facility", "transfer_urgency", "bed_availability", "systolic_bp",
    "heart_rate", "spo2", "monitoring_required", "specialty_need",
]
CRITICAL = ["patient_age", "presenting_problem", "requested_service", "oxygen_support",
            "vasopressor_use", "isolation_requirements", "systolic_bp", "heart_rate",
            "spo2", "monitoring_required", "specialty_need"]


@dataclass
class Extraction:
    values: dict[str, Any] = field(default_factory=lambda: dict.fromkeys(FIELDS))
    evidence: dict[str, list[str]] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    version: str = "rules-nlp-1.0"


class Extractor(Protocol):
    """Future LLM adapters must return the same note-grounded, nullable schema."""

    def extract(self, note: str) -> Extraction: ...


@dataclass
class Recommendation:
    route: str
    reasons: list[str]
    rule_ids: list[str]
    warnings: list[str]
    completeness: float
    confidence: str
    requires_review: bool = True


def missing_fields(result: Extraction) -> dict[str, list[str]]:
    return {
        "critical": [f for f in CRITICAL if result.values.get(f) is None],
        "operational": [f for f in FIELDS if f not in CRITICAL and result.values.get(f) is None],
    }

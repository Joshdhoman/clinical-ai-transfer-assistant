"""Minimal local demo audit. No raw note or extracted clinical values are persisted."""

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .routing import VERSION
from .schema import ABSTAIN, ROUTES, Extraction, Recommendation

_LOCK = threading.Lock()


def decision_record(note: str, extraction: Extraction, recommendation: Recommendation,
                    selected_route: str, reason: str, reviewer: str, verified: bool,
                    analysis_id: str) -> dict:
    if not verified:
        raise ValueError("Acknowledge the coordinator review before recording the final decision.")
    if selected_route not in ROUTES + [ABSTAIN]:
        raise ValueError("Invalid review route.")
    if len(reason.strip()) < 10:
        raise ValueError("Provide a review reason of at least 10 characters.")
    if not reviewer.strip() or len(reviewer) > 80:
        raise ValueError("Use a fictional reviewer name of 1–80 characters.")
    if len(reason) > 1000:
        raise ValueError("Keep the synthetic review reason within 1,000 characters.")
    return {"record_id": str(uuid.uuid4()), "analysis_id": analysis_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "synthetic_only": True,
            "note_sha256": hashlib.sha256(note.encode()).hexdigest(),
            "extractor_version": extraction.version, "router_version": VERSION,
            "suggested_route": recommendation.route, "final_route": selected_route,
            "route_changed": selected_route != recommendation.route,
            "decision_rationale": reason.strip(), "reviewer_name": reviewer.strip(),
            "review_acknowledged": True,
            "critical_completeness": recommendation.completeness,
            "rule_ids": recommendation.rule_ids, "conflict_fields": extraction.conflicts}


def append_record(record: dict, path: Path) -> None:
    """Append once per analysis in one process; not a multiprocess audit database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        if path.exists():
            prior = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            if any(item["analysis_id"] == record["analysis_id"] for item in prior):
                raise ValueError("This analysis already has a final decision. Analyze again to record a new decision.")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

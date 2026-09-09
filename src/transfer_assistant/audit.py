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


def review_event(note: str, extraction: Extraction, recommendation: Recommendation,
                 selected_route: str, reason: str, reviewer: str, verified: bool,
                 analysis_id: str) -> dict:
    if not verified:
        raise ValueError("Confirm human review before recording a decision.")
    if selected_route not in ROUTES + [ABSTAIN]:
        raise ValueError("Invalid review route.")
    if len(reason.strip()) < 10:
        raise ValueError("Provide a review reason of at least 10 characters.")
    if not reviewer.strip() or len(reviewer) > 80:
        raise ValueError("Use a fictional reviewer alias of 1–80 characters.")
    if len(reason) > 1000:
        raise ValueError("Keep the synthetic review reason within 1,000 characters.")
    return {"event_id": str(uuid.uuid4()), "analysis_id": analysis_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "synthetic_only": True,
            "note_sha256": hashlib.sha256(note.encode()).hexdigest(),
            "extractor_version": extraction.version, "router_version": VERSION,
            "suggested_route": recommendation.route, "reviewed_route": selected_route,
            "overridden": selected_route != recommendation.route,
            "reason": reason.strip(), "reviewer_alias": reviewer.strip(), "verified": True,
            "critical_completeness": recommendation.completeness,
            "rule_ids": recommendation.rule_ids, "conflict_fields": extraction.conflicts}


def append_event(event: dict, path: Path) -> None:
    """Append once per analysis in one process; not a multiprocess audit database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        if path.exists():
            prior = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            if any(e["analysis_id"] == event["analysis_id"] for e in prior):
                raise ValueError("This analysis already has a recorded review. Analyze again for a new review.")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

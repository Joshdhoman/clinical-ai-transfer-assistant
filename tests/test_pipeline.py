import json

import pytest

from transfer_assistant.audit import append_event, review_event
from transfer_assistant.examples import EXAMPLES
from transfer_assistant.extraction import RuleBasedExtractor
from transfer_assistant.modeling import FEATURES, feature_frame, train_model
from transfer_assistant.routing import recommend
from transfer_assistant.schema import ABSTAIN, missing_fields
from transfer_assistant.synthetic import generate_dataset, validate_dataset


@pytest.fixture
def extractor():
    return RuleBasedExtractor()


@pytest.mark.parametrize("name,expected", list(zip(list(EXAMPLES), ["ICU", "telemetry", "medical/surgical", "specialty review", ABSTAIN, ABSTAIN])))
def test_demo_routes(extractor, name, expected):
    assert recommend(extractor.extract(EXAMPLES[name])).route == expected


def test_unknown_is_not_negative(extractor):
    missing = extractor.extract("Age: 55. Currently on room air.")
    explicit = extractor.extract("Age: 55. Currently on room air. No vasopressors.")
    assert missing.values["vasopressor_use"] is None
    assert explicit.values["vasopressor_use"] is False
    assert "vasopressor_use" in missing_fields(missing)["critical"]
    assert recommend(missing).route == ABSTAIN


def test_sentence_after_numeric_age_and_vitals(extractor):
    r = extractor.extract("Age: 54. Service: cardiology. Pulse 83. Isolation: none. BP 123/70. Urgency: urgent.")
    assert r.values["requested_service"] == "cardiology"
    assert r.values["isolation_requirements"] == "none"
    assert r.values["transfer_urgency"] == "urgent"


def test_overlapping_labels_and_case_preservation(extractor):
    r = extractor.extract("Service needed: pulmonology. Referrer asks for ICU bed. Referrer: Synthetic Valley ED. Working diagnosis: COPD exacerbation.")
    assert r.values["requested_service"] == "pulmonology"
    assert r.values["referring_facility"] == "Synthetic Valley ED"
    assert r.values["requested_level_of_care"] == "ICU"
    assert r.values["presenting_problem"] == "COPD exacerbation"


def test_conflict_and_evidence(extractor):
    r = extractor.extract("No vasopressors. Currently on norepinephrine.")
    assert r.values["vasopressor_use"] is None
    assert "vasopressor_use" in r.conflicts
    assert len(r.evidence["vasopressor_use"]) == 2


def test_negation_history_and_hypothetical(extractor):
    r = extractor.extract("History of prior mechanical ventilation. If deterioration occurs consider norepinephrine. Not intubated. Currently on room air. No vasopressors.")
    assert r.values["oxygen_support"] == "room air"
    assert r.values["vasopressor_use"] is False


def test_icu_priority_and_capacity_invariance(extractor):
    note = EXAMPLES["ICU support + capacity constraint"]
    a = recommend(extractor.extract(note))
    b = recommend(extractor.extract(note.replace("unavailable", "available")))
    assert a.route == b.route == "ICU"
    assert any("Capacity constraint" in w for w in a.warnings)
    assert not any("Capacity constraint" in w for w in b.warnings)
    r = extractor.extract(note.replace("No specialty evaluation required", "Specialty evaluation required"))
    assert recommend(r).route == "ICU"
    assert any("also required" in w for w in recommend(r).warnings)


def test_pediatric_abstention_even_with_support(extractor):
    assert recommend(extractor.extract("Age: 12. Currently on norepinephrine.")).route == ABSTAIN


@pytest.mark.parametrize("note", ["", "  ", "x"*12001])
def test_input_validation(extractor, note):
    with pytest.raises(ValueError):
        extractor.extract(note)


def test_dataset_reproducibility_validation_and_split():
    a, b = generate_dataset(), generate_dataset()
    assert a.equals(b)
    assert validate_dataset(a)["passed"]
    assert a.groupby("template_family").split.nunique().eq(1).all()
    corrupt = a.copy()
    corrupt.loc[0, "patient_age"] = 200
    with pytest.raises(ValueError):
        validate_dataset(corrupt)


def test_ml_contract_and_unseen_category(extractor):
    data = generate_dataset(160)
    x = feature_frame([extractor.extract(n) for n in data.free_text_transfer_note])
    assert set(x) == set(FEATURES)
    assert not {"routing_target", "scenario_id", "transfer_outcome", "bed_availability", "patient_age"} & set(x)
    train, val = data.split.eq("train"), data.split.eq("validation")
    model, selection = train_model(x[train], data.routing_target[train], x[val], data.routing_target[val])
    unknown = feature_frame([extractor.extract("SYNTHETIC CASE. Unsupported terminology.")])
    assert len(model.predict(unknown)) == 1
    assert selection["selected_depth"] in [3, 4, 5, 6]


def test_review_requires_verification_reason_and_valid_route(extractor, tmp_path):
    note = EXAMPLES["Telemetry referral"]
    r = extractor.extract(note)
    rec = recommend(r)
    with pytest.raises(ValueError):
        review_event(note, r, rec, "ICU", "Reason is recorded", "Demo", False, "analysis")
    with pytest.raises(ValueError):
        review_event(note, r, rec, "ICU", "short", "Demo", True, "analysis")
    with pytest.raises(ValueError):
        review_event(note, r, rec, "arbitrary route", "Reason is recorded", "Demo", True, "analysis")
    event = review_event(note, r, rec, "ICU", "Synthetic escalation reviewed", "Demo", True, "analysis")
    path = tmp_path / "events.jsonl"
    append_event(event, path)
    assert json.loads(path.read_text())["overridden"]
    assert note not in path.read_text()
    assert "extracted_values" not in event
    with pytest.raises(ValueError):
        append_event(event, path)

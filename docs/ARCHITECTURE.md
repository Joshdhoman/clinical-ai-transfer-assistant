# Architecture and operational design

## Boundaries

This is an adult, English-language, synthetic-only workflow demonstration. No EHR integration, live capacity feed, transport dispatch, patient matching, acceptance authority, or treatment advice is implemented.

## Contracts

- Generator emits complete scenario facts, a fictional note, independently retained note-visible annotations, and a scenario routing target.
- Extractor accepts only the note, returns normalized values, supporting text spans, and conflicts. Unknown is distinct from an explicit negative.
- Missing-information detector operates on extracted values; it never fills absent facts from scenario truth.
- Router consumes extraction output and emits a provisional routing category, rationale, rule IDs, evidence completeness, and human-review warnings. It may abstain when essential support information is unknown.
- ML uses the same extracted features. It is a comparison, not an override of the deterministic safety checks.
- Final coordinator decision captures whether the suggested route was accepted or changed and why. Changes to the source note invalidate the previous result. No automatic transfer action exists.
- Local decision records omit raw notes and are explicitly not a production compliance system.

## Label and evaluation design

Scenario catalog labels are assigned before note rendering. The labels reflect simplified operational assumptions, not actual clinical adjudication. Note omission creates a deliberate gap between observable referral data and complete scenario state. Eight template families are assigned to development/training (0–4), validation (5), and held-out testing (6–7). No test results select the model hyperparameters. Templates still share a generator and vocabulary: this only reduces template leakage, not simulation bias.

Evaluation separates field extraction against note-visible facts from end-to-end routing against complete scenario labels. Abstentions are counted as misses in overall accuracy and separately reported as coverage. An independently authored challenge set probes negation, conflicts, historical/hypothetical statements, unknowns, and unsupported input.

## Future LLM adapter

Implement the `Extractor` protocol with schema-constrained output and note-grounded evidence. Keep credentials server-side, turn off provider retention where contractually available, and require an approved data boundary before any real clinical text is considered. Validate contradictions, negation, hallucination, and prompt injection with a separately adjudicated test set. Keep routing and human review downstream of the same typed interface. No API call is included in this prototype.

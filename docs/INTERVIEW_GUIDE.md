# Five-minute portfolio walkthrough

1. **Workflow problem:** explain the coordinator's work of interpreting an incomplete referral, confirming support needs, locating accepting services, and separating capacity from acuity. Avoid claiming measured time savings.
2. **Data contract:** show complete scenario facts versus note-visible annotations and missingness. Explain why an absent pressor statement cannot mean no pressors.
3. **Live review:** select the ICU/capacity example, confirm synthetic data, analyze, inspect evidence, and show that bed unavailability does not lower the suggestion. Load the incomplete example to demonstrate withholding.
4. **Human control:** make a final coordinator decision by accepting the suggested route or choosing a different route, document why, and show the local decision record. Explain what a real authenticated audit system would require.
5. **Honest results:** open Evaluation, discuss rules coverage versus the tree's full coverage, then inspect failures. Explain generator bias, the limits of template holdout, and the need for independently adjudicated clinical validation.

## Analyst decisions to discuss

- Translate workflow pain into measurable requirements before selecting an LLM.
- Keep deterministic clinical/operational policy separate from extraction technology.
- Show evidence, conflicts, missingness, and disagreement instead of a bare route.
- Evaluate extraction separately from downstream routing so failures can be attributed.
- Treat a missed ICU suggestion and a deliberately withheld route differently in review, while counting both transparently in metrics.
- Propose workflow outcome measurement: referral completeness, callbacks, coordinator review time, disagreement burden, and clinician-adjudicated routing safety. No benefit is assumed from model accuracy alone.

## Scope and next iteration

Map a real institution's policies with authorized stakeholders; collect no PHI for the portfolio. Add a schema-constrained LLM extractor only after testing whether supported NLP coverage is actually the bottleneck. A real pilot would require approved data handling, clinical validation, security, governance, and human-factors work.

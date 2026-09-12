"""Human review workbench. Run: python -m streamlit run app.py."""

import json
import os
import uuid
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from transfer_assistant.audit import append_event, review_event
from transfer_assistant.examples import EXAMPLES
from transfer_assistant.extraction import RuleBasedExtractor
from transfer_assistant.modeling import explain_tree
from transfer_assistant.routing import recommend
from transfer_assistant.schema import ABSTAIN, FIELDS, ROUTES, missing_fields

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Clinical AI | Transfer operations", page_icon=":material/hub:", layout="wide")
st.session_state.setdefault("note", next(iter(EXAMPLES.values())))
st.session_state.setdefault("analysis", None)
st.session_state.setdefault("review_saved", None)


def invalidate():
    st.session_state.analysis = None
    st.session_state.review_saved = None


def load_example():
    st.session_state.note = EXAMPLES[st.session_state.example]
    invalidate()


@st.cache_resource(max_entries=1)
def load_model(modified: float):
    # Load only a trusted local artifact produced by scripts/build_project.py.
    return joblib.load(ROOT / "artifacts/routing_tree.joblib")


@st.cache_data(max_entries=4)
def load_csv(relative: str, modified: float):
    return pd.read_csv(ROOT / relative)


with st.sidebar:
    st.markdown("### :material/hub: Transfer operations")
    st.caption("CLINICAL AI WORKFLOW ASSISTANT")
    page = st.radio("Workspace", ["Request review", "Evaluation", "Governance"], label_visibility="collapsed")
    st.space("medium")
    st.badge("Synthetic data only", icon=":material/science:", color="blue")
    st.markdown("**A coordinator stays in control.**")
    st.caption("Extract → clarify → suggest → review")
    st.caption("Local prototype · No paid API required")
    st.caption("Independent portfolio project. No Rush affiliation or endorsement.")

st.caption("HEALTHCARE OPERATIONS / DECISION SUPPORT")
st.title("Clinical AI Workflow Assistant")
st.write("Patient transfer operations — turn a referral narrative into a reviewable handoff.")
st.warning("Synthetic cases only. Not validated for clinical use. Every suggestion requires human review.", icon=":material/health_and_safety:")

if page == "Request review":
    left, right = st.columns([1.05, 1.2], gap="large")
    with left:
        st.subheader("01 / Transfer request")
        st.selectbox("Load a synthetic example", list(EXAMPLES), key="example", on_change=load_example)
        note = st.text_area("Synthetic referral note", key="note", height=300, max_chars=12000, on_change=invalidate)
        synthetic = st.checkbox("I confirm this text is entirely synthetic", key="synthetic_confirm")
        if st.button("Analyze request", type="primary", icon=":material/clinical_notes:", width="stretch"):
            if not synthetic:
                st.error("Confirm that the input contains no real patient information.")
            else:
                try:
                    extraction = RuleBasedExtractor().extract(note)
                    recommendation = recommend(extraction)
                    st.session_state.analysis = {"note": note, "extraction": extraction,
                                                 "recommendation": recommendation, "id": str(uuid.uuid4())}
                    st.session_state.review_saved = None
                except ValueError as exc:
                    st.error(str(exc))
        st.caption("Notes are processed locally and are not written to the audit log. This demo does not detect or de-identify PHI.")
    analysis = st.session_state.analysis
    # Also check content equality at submission; stale analyses must never be reviewed.
    if analysis and analysis["note"] != note:
        invalidate()
        analysis = None
    with right:
        st.subheader("02 / Provisional routing")
        if not analysis:
            with st.container(border=True):
                st.markdown("### Ready for a synthetic referral")
                st.write("Analyze a request to inspect the supporting evidence, clarification needs, and routing rationale.")
                st.caption("Four pathways: ICU · telemetry · medical/surgical · specialty review")
        else:
            extraction, rec = analysis["extraction"], analysis["recommendation"]
            missing = missing_fields(extraction)
            with st.container(border=True):
                st.badge("Human review required", color="orange", icon=":material/person_check:")
                st.markdown(f"### {rec.route.capitalize() if rec.route != 'ICU' else 'ICU'}")
                st.write("**Confidence:** " + rec.confidence)
                st.progress(rec.completeness, text=f"Routing-field completeness: {rec.completeness:.0%}")
                st.caption("Completeness measures documentation, not the probability that a route is correct.")
                for rule, reason in zip(rec.rule_ids, rec.reasons):
                    st.write(f"**{rule}** · {reason}")
            for warning in rec.warnings[1:]:
                st.warning(warning)
    if analysis:
        extraction, rec = analysis["extraction"], analysis["recommendation"]
        missing = missing_fields(extraction)
        evidence_tab, clarify_tab, model_tab = st.tabs(["Extracted evidence", "Clarification checklist", "ML comparison"])
        with evidence_tab:
            table = [{"Field": f.replace("_", " ").capitalize(),
                      "Value": "Unknown / conflicting" if extraction.values[f] is None else str(extraction.values[f]),
                      "Source evidence": " | ".join(extraction.evidence.get(f, [])) or "Not found"} for f in FIELDS]
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        with clarify_tab:
            c1, c2 = st.columns(2)
            for container, group in [(c1, "critical"), (c2, "operational")]:
                with container:
                    st.markdown(f"#### {group.capitalize()} information")
                    if missing[group]:
                        for name in missing[group]:
                            st.write(f"• Verify {name.replace('_', ' ')} with the referring team.")
                    else:
                        st.success("All tracked fields in this group are documented.")
            st.caption("This checklist is not exhaustive: confirm trends, mental status, current therapies, accepting clinician, transport needs, and local transfer requirements outside this demo.")
        with model_tab:
            model_path = ROOT / "artifacts/routing_tree.joblib"
            if model_path.exists():
                comparison = explain_tree(load_model(model_path.stat().st_mtime), extraction)
                st.write(f"**Experimental tree suggestion:** {comparison['route']}")
                if comparison["route"] != rec.route:
                    st.warning("Rules and ML disagree. Review the source evidence; the model does not resolve missing information.")
                st.caption(comparison["caution"])
                st.code("\n".join(comparison["decision_path"]), language="text")
            else:
                st.info("The optional comparison model is not built. Run python scripts/build_project.py; rule-based review is available now.")
        st.subheader("03 / Human review and override")
        if st.session_state.review_saved:
            event = st.session_state.review_saved
            st.success(f"Review recorded: {event['reviewed_route']}. No transfer action was taken.")
            st.download_button("Download review event", json.dumps(event, indent=2), file_name="synthetic_review.json", mime="application/json")
        else:
            with st.form("human_review"):
                c1, c2 = st.columns(2)
                selected = c1.selectbox("Reviewed routing", [ABSTAIN] + ROUTES,
                                        index=([ABSTAIN] + ROUTES).index(rec.route))
                reviewer = c2.text_input("Fictional reviewer alias", value="Demo coordinator", max_chars=80)
                reason = st.text_area("Review rationale / override reason (synthetic only)", max_chars=1000,
                                      placeholder="Explain what you verified and why you confirmed or changed the route.")
                verified = st.checkbox("I reviewed the source evidence, missing fields, warnings, and final routing.")
                save = st.form_submit_button("Record human review", type="primary")
            if save:
                try:
                    event = review_event(note, extraction, rec, selected, reason, reviewer, verified, analysis["id"])
                    append_event(event, Path(os.environ.get("TRANSFER_AUDIT_PATH", str(ROOT / "audit/reviews.jsonl"))))
                    st.session_state.review_saved = event
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(f"Review was not recorded: {exc}")
            st.caption("Local audit stores route, reason, fictional alias, versions, and note hash. Do not enter identifiers in the rationale. It has no authentication or tamper resistance.")

elif page == "Evaluation":
    st.subheader("Measured performance on synthetic requests")
    metrics_path = ROOT / "reports/metrics.json"
    if not metrics_path.exists():
        st.info("Run python scripts/build_project.py to generate the evaluation reports.")
    else:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.caption("625 training · 125 validation · 250 test requests · template families separated")
        with st.container(horizontal=True):
            st.metric("Rules accuracy", f"{metrics['rules']['accuracy']:.1%}", border=True)
            st.metric("Tree accuracy", f"{metrics['decision_tree']['accuracy']:.1%}", border=True)
            st.metric("Rules coverage", f"{metrics['rules']['coverage']:.1%}", border=True)
            st.metric("Test referrals", metrics["rules"]["n"], border=True)
        st.info("Overall accuracy counts withheld suggestions as misses. Higher ML coverage can hide unsupported inferences from incomplete referrals.")
        rows = [{"Approach": name, **{key: metrics[name][key] for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "coverage", "icu_false_positives", "icu_false_negatives"]}}
                for name in ["majority_baseline", "rules", "decision_tree"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.image(str(ROOT / "reports/figures/confusion_matrices.png"))
        relative = "reports/failures.csv"
        failures = load_csv(relative, (ROOT / relative).stat().st_mtime)
        st.markdown("#### Inspect actual failures")
        approach = st.selectbox("Approach", ["rules", "decision_tree"])
        st.dataframe(failures[failures.approach.eq(approach)], hide_index=True)
        relative = "reports/subgroups.csv"
        subgroups = load_csv(relative, (ROOT / relative).stat().st_mtime)
        group = subgroups[subgroups.dimension.eq("completeness_group")]
        st.plotly_chart(px.bar(group, x="group", y="accuracy", color="approach", barmode="group",
                               title="Documentation completeness and routing accuracy", range_y=[0, 1],
                               color_discrete_sequence=["#087E83", "#315B83"]), width="stretch")
        st.caption("Simulation-specific results. Templates share an authored generator. Test diagnostics informed parser bug fixes; this is a development benchmark, not independent external validation.")

else:
    st.subheader("Human oversight by design")
    st.markdown((ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8"))

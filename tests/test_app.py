from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_analysis_override_and_stale_state(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSFER_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.checkbox(key="synthetic_confirm").check().run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["analysis"]["recommendation"].route == "ICU"
    # Review validation should reject an unchecked acknowledgment.
    app.button[1].click().run()
    assert any("Confirm human review" in e.value for e in app.error)
    app.selectbox[1].select("specialty review")
    app.text_area[1].input("Synthetic alternate pathway reviewed with fictional receiving team.")
    app.checkbox[1].check()
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state["review_saved"]["overridden"]
    assert (tmp_path / "audit.jsonl").exists()
    app.text_area(key="note").input("SYNTHETIC CASE. Age: 71.").run()
    assert app.session_state["analysis"] is None
    assert app.session_state["review_saved"] is None


def test_evaluation_and_governance_render():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.radio[0].set_value("Evaluation").run()
    assert not app.exception
    assert len(app.metric) == 4
    app.radio[0].set_value("Governance").run()
    assert not app.exception

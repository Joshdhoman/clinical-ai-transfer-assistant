"""Create the five reproducible narrative analyses; execute with execute_notebooks.py."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
get_ipython().run_line_magic('matplotlib', 'inline')

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists())
DATA = ROOT / 'data/synthetic/transfer_requests.csv'
plt.rcParams.update({'figure.figsize': (10, 4), 'axes.spines.top': False, 'axes.spines.right': False})
'''


def write(name, title, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}}
    nb.cells = [nbf.v4.new_markdown_cell(title), nbf.v4.new_code_cell(SETUP)]
    nb.cells.extend(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text) for kind, text in cells)
    (ROOT / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, ROOT / "notebooks" / name)


write("01_generate_validate_data.ipynb", "# 01 · Generate and validate synthetic transfers\n\n**Question:** Can we construct a reproducible referral cohort with explicit provenance, observable note labels, and validation? Entirely fictional; no real patient source.", [
    ("md", "## Scenario design\nSixteen authored adult scenarios vary service, respiratory support, pressors, monitoring, and specialty needs. The target is assigned before prose rendering. It is a simulation assumption, not a clinical label. Eight prose families partition 625/125/250 records into train/validation/test."),
    ("code", "from transfer_assistant.synthetic import generate_dataset, validate_dataset, write_dataset, SEED\ndf = write_dataset(ROOT)\nvalidation = validate_dataset(df)\nassert df.equals(generate_dataset(seed=SEED))\ndisplay(validation)"),
    ("code", "display(df[['request_id','patient_age','presenting_problem','requested_service','oxygen_support','vasopressor_use','routing_target','split']].head(8))\ndisplay(pd.crosstab(df.split, df.routing_target))"),
    ("md", "## Complete scenario versus referral evidence\nA negative is observable information; a missing value is not. Conflicts are separately recorded but unavailable to routing until clarified. Compare one incomplete referral's source truth with its visible annotation."),
    ("code", "row = df[df.omitted_fields.ne('[]')].iloc[0]\nprint(row.free_text_transfer_note)\nvisible = json.loads(row.visible_annotations)\ndisplay(pd.DataFrame([{'field': k, 'source_truth': str(row[k]), 'note_visible': str(v)} for k,v in visible.items()]))"),
    ("code", "family_sets = {s: set(g.template_family) for s,g in df.groupby('split')}\nassert not family_sets['train'] & family_sets['test']\nassert not family_sets['validation'] & family_sets['test']\ndisplay(family_sets)"),
    ("md", "## Interpretation\nThese checks establish consistency and lineage, not clinical realism. Documentation omission rates and physiological ranges are authored assumptions. Never use hidden source truth to fill inference-time gaps. See the data dictionary for every field and generation parameter.")])

write("02_exploratory_analysis.ipynb", "# 02 · Explore referral operations\n\n**Question:** How do simulated acuity, documentation completeness, and capacity interact? This notebook explores the training partition; final test results belong in notebook 05.", [
    ("code", "all_data = pd.read_csv(DATA)\ndf = all_data[all_data.split.eq('train')].copy()\nprint(f'Training referrals: {len(df)}')\ndisplay(df[['patient_age','systolic_bp','heart_rate','spo2']].describe())"),
    ("code", "fig, axes = plt.subplots(1,2,figsize=(12,4))\ndf.routing_target.value_counts().plot.barh(ax=axes[0],color='#087E83',title='Synthetic route mix')\ndf.patient_age.plot.hist(ax=axes[1],bins=12,color='#315B83',title='Synthetic adult ages')\nfig.tight_layout()\nplt.show()"),
    ("code", "from transfer_assistant.schema import FIELDS\nvisible = pd.DataFrame([json.loads(s) for s in df.visible_annotations])\nmissing = visible.isna().mean().sort_values()\nmissing.plot.barh(color='#B0792B',title='Unavailable note fields (absence or conflict)',figsize=(10,6))\nplt.xlabel('Fraction of training referrals')\nplt.show()"),
    ("code", "df['unavailable_count'] = [sum(v is None for v in json.loads(s).values()) for s in df.visible_annotations]\ndisplay(df.groupby('referring_facility').agg(n=('request_id','size'),mean_unavailable=('unavailable_count','mean')))\ndisplay(pd.crosstab(df.bed_availability,df.transfer_outcome,normalize='index').round(3))"),
    ("md", "## Operational interpretation\nThe generator makes documentation completeness depend on fictional referral source and outcomes depend on capacity. These relationships are designed, not discovered healthcare facts. The prototype keeps capacity outside acuity prediction to avoid normalizing lower-acuity placement when a bed is unavailable. Facility-specific missingness can still create unequal abstention burden."),
    ("code", "display(df.groupby('routing_target')[['systolic_bp','heart_rate','spo2']].mean().round(1))"),
    ("md", "## Modeling risk\nStrong scenario–vital associations make this synthetic problem easier than clinical practice. The model may infer missing support from proxies. Exclude outcome, scenario ID, requested bed, and hidden truth; analyze incomplete-referral failures separately.")])

write("03_nlp_extraction.ipynb", "# 03 · Extract evidence and detect missing information\n\n**Question:** Which facts can a deterministic extractor recover, and where should it admit uncertainty?", [
    ("code", "from transfer_assistant.extraction import RuleBasedExtractor\nfrom transfer_assistant.examples import EXAMPLES\nfrom transfer_assistant.schema import missing_fields\nextractor = RuleBasedExtractor()\nnote = EXAMPLES['Conflicting pressor documentation']\nresult = extractor.extract(note)\nprint(note)\ndisplay(pd.DataFrame([{'field':k,'value':str(v),'evidence':' | '.join(result.evidence.get(k, []))} for k,v in result.values.items()]))\ndisplay({'conflicts':result.conflicts,'missing':missing_fields(result)})"),
    ("md", "## Observable truth, not hidden truth\nScore against note-visible annotations. An omitted or contradictory field is expected to be unavailable. Value precision counts incorrect non-null extraction as a false positive; value recall also counts it as a false negative. Exact-match accuracy includes correctly unknown fields."),
    ("code", "from transfer_assistant.evaluation import extraction_metrics\ndata = pd.read_csv(DATA)\nvalidation = data[data.split.eq('validation')].reset_index(drop=True)\nresults = [extractor.extract(n) for n in validation.free_text_transfer_note]\nfields, errors, missing_scores = extraction_metrics(validation, results)\ndisplay(fields)\ndisplay(missing_scores)\ndisplay(errors.head())"),
    ("code", "for text in ['No vasopressors.', 'Vasopressor use: unknown.', 'History of prior mechanical ventilation.', 'If deterioration occurs, consider norepinephrine.']:\n    parsed = extractor.extract(text)\n    print(text, '=>', {k:parsed.values[k] for k in ['oxygen_support','vasopressor_use']})"),
    ("md", "## Open vocabulary and LLM extension\nThe shared `Extractor` protocol permits a later LLM implementation, but it must return the same nullable schema and source evidence. No schema-valid output is automatically true. Preserve downstream missingness, conflict validation, routing policy, and the final coordinator decision. Broader abbreviations and complex temporality need independently authored tests and clinically adjudicated evaluation."),
    ("code", "display(pd.read_csv(ROOT/'reports/challenge_results.csv'))"),
    ("md", "## Limits\nThe authored challenge suite is a development diagnostic, not a population sample. Supported generator prose can score perfectly while unrecognized real-world phrasing fails. Keep the actual failed cases visible; do not convert expectations to match the parser.")])

write("04_routing_model.ipynb", "# 04 · Compare policy rules with an interpretable model\n\n**Question:** Can a shallow tree reproduce scenario routing, and what does it learn when the referral is incomplete?", [
    ("code", "from transfer_assistant.extraction import RuleBasedExtractor\nfrom transfer_assistant.modeling import feature_frame, train_model, FEATURES\nfrom transfer_assistant.routing import recommend\nfrom transfer_assistant.schema import ROUTES\nfrom sklearn.metrics import classification_report\nfrom sklearn.tree import export_text\ndata = pd.read_csv(DATA)\nextractor = RuleBasedExtractor()\nparsed = [extractor.extract(n) for n in data.free_text_transfer_note]\nx = feature_frame(parsed)\ntrain, validation = data.split.eq('train'), data.split.eq('validation')\nmodel, selection = train_model(x[train],data.routing_target[train],x[validation],data.routing_target[validation])\ndisplay(selection)\nprint('Input features:', FEATURES)"),
    ("md", "## Model selection without test fitting\nTraining-only one-hot encoding and median imputation feed a class-weighted decision tree, with depth selected on validation macro-F1. Missing categorical values are explicit `unknown`; numerical imputation adds missingness indicators. The model excludes age, facility, capacity, requested bed, IDs, and outcomes."),
    ("code", "y = data.routing_target[validation]\nrules = [recommend(r).route for r,use in zip(parsed,validation) if use]\nprint('RULES: validation')\nprint(classification_report(y,rules,labels=ROUTES,zero_division=0))\nprint('TREE: validation')\nprint(classification_report(y,model.predict(x[validation]),labels=ROUTES,zero_division=0))"),
    ("code", "names = model.named_steps['preprocess'].get_feature_names_out()\nprint(export_text(model.named_steps['tree'],feature_names=list(names)))\nimportance = pd.Series(model.named_steps['tree'].feature_importances_,index=names).sort_values()\nimportance[importance>0].plot.barh(color='#087E83',title='Tree impurity-based feature importance')\nplt.show()"),
    ("code", "from transfer_assistant.modeling import explain_tree\nfrom transfer_assistant.examples import EXAMPLES\nincomplete = extractor.extract(EXAMPLES['Incomplete referral'])\ndisplay({'rules':recommend(incomplete).route,'tree':explain_tree(model,incomplete)})"),
    ("md", "## Tradeoff\nThe tree can use correlated service and vitals to fill an inferential gap; the deterministic workflow can withhold a lower-acuity recommendation. Full model coverage is not evidence that missing facts were recovered. Tree leaf proportions are uncalibrated, particularly with balanced class weights. The app keeps the tree as a comparison.")])

write("05_model_evaluation.ipynb", "# 05 · Evaluate routing, omissions, and failure modes\n\n**Question:** What actually happened on the held-out prose families, and what would block clinical use?", [
    ("md", "## Evaluation provenance\nRun `python scripts/build_project.py` before this notebook. All numbers below are loaded from that executed pipeline, not manually entered. Test families were excluded from fitting and hyperparameter selection; parser debugging used development test diagnostics, so this is not an untouched external evaluation."),
    ("code", "metrics = json.loads((ROOT/'reports/metrics.json').read_text())\nmanifest = json.loads((ROOT/'artifacts/run_manifest.json').read_text())\ndisplay(manifest)\ndisplay(pd.DataFrame([{ 'approach':name, **{k:m[k] for k in ['accuracy','macro_precision','macro_recall','macro_f1','coverage','selective_accuracy','icu_false_positives','icu_false_negatives','icu_fn_abstentions']}} for name,m in metrics.items() if 'coverage' in m]))"),
    ("code", "from IPython.display import Image\ndisplay(Image(filename=str(ROOT/'reports/figures/confusion_matrices.png')))\ndisplay(pd.DataFrame(metrics['decision_tree']['per_class']).T)"),
    ("code", "predictions = pd.read_csv(ROOT/'reports/test_predictions.csv')\nfailures = pd.read_csv(ROOT/'reports/failures.csv')\ndisplay(failures.groupby(['approach','failure_type']).size().rename('count').reset_index())\nfor row in failures.groupby(['approach','failure_type']).head(1).itertuples():\n    print(f'\\n{row.request_id} | {row.approach} | {row.expected} -> {row.predicted}')\n    print(row.note)"),
    ("md", "## ICU error definitions\nAn ICU false positive is any non-ICU scenario labeled ICU. A false negative is any ICU scenario not labeled ICU, including a withheld suggestion. The separate abstention count prevents conflating an intentional clarification request with lower-acuity routing. Neither error count measures actual patient harm in this fictional cohort."),
    ("code", "display(pd.read_csv(ROOT/'reports/extraction_metrics.csv'))\ndisplay(metrics['missing_information'])\ndisplay(pd.read_csv(ROOT/'reports/subgroups.csv'))\ndisplay(json.loads((ROOT/'reports/challenge_summary.json').read_text()))\nchallenge = pd.read_csv(ROOT/'reports/challenge_results.csv')\ndisplay(challenge[~challenge.passed])"),
    ("md", "## What these results can support\nThey demonstrate a reproducible engineering workflow, transparent policy logic, diagnostic error analysis, and a final coordinator decision interface. They cannot establish clinical safety, representativeness, equity, time savings, or return on investment. Even high synthetic scores can result from circular scenario assumptions and strongly correlated physiology. See the model card for an external adjudication, prospective shadow testing, human-factors, governance, and monitoring plan.")])

print("Created five notebooks.")

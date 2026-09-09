"""Author the five narrative notebooks. Execute them with execute_notebooks.py.

Kept as a script so the notebooks are regenerated from one place and stay
consistent. Re-run after editing, then `python scripts/execute_notebooks.py`.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

SETUP = """import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

get_ipython().run_line_magic("matplotlib", "inline")
pd.set_option("display.max_columns", 40)
plt.rcParams.update({"figure.figsize": (9, 4), "axes.spines.top": False, "axes.spines.right": False})

# Anchor on this project specifically: it sits in a subdirectory of a repository
# that has its own pyproject.toml, so "nearest pyproject.toml" is not enough.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "transfer_decline").is_dir())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
"""


def write(name: str, title: str, cells: list[tuple[str, str]]) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    nb.cells = [nbf.v4.new_markdown_cell(title), nbf.v4.new_code_cell(SETUP)]
    for kind, text in cells:
        nb.cells.append(
            nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)
        )
    (ROOT / "notebooks").mkdir(exist_ok=True)
    nbf.write(nb, ROOT / "notebooks" / name)


write(
    "01_data_preparation.ipynb",
    "# 01 · Prepare the synthetic transfer-request extract\n\n"
    "**Question:** can the raw export be turned into a clean, de-duplicated cohort "
    "with a well-defined target and a defensible train/validation/test split?\n\n"
    "All data is synthetic. No real patient, facility, or payer information is used anywhere.",
    [
        ("md",
         "## The raw file, warts and all\n"
         "The extract imitates a transfer-center report pulled from Epic. It carries the "
         "messiness such a pull usually has: inconsistent capitalisation and stray "
         "whitespace in `transport_mode` and `referring_facility_type`, missing values in "
         "a few columns, and some double-entered requests."),
        ("code",
         "from transfer_decline.data import load_raw\n"
         "raw = load_raw(ROOT)\n"
         "print(raw.shape)\n"
         "print('exact duplicate rows:', raw.duplicated().sum())\n"
         "display(raw['transport_mode'].value_counts())\n"
         "display(raw['referring_facility_type'].value_counts())\n"
         "display(raw.isna().sum()[lambda s: s > 0])"),
        ("md",
         "## Cleaning\n"
         "`clean()` drops the exact duplicates, trims and case-normalises the two text "
         "columns, derives the binary target, and adds calendar features from the request "
         "timestamp. Missing values are **left in place** — imputation belongs in the "
         "modelling pipeline where it is fit on the training split only.\n\n"
         "**Target definition:** `declined = 1` when `disposition == 'Declined'`. "
         "`Accepted - Not Transferred` counts as an acceptance: the transfer center said "
         "yes even though the patient never arrived."),
        ("code",
         "from transfer_decline.data import clean, time_split, validate\n"
         "df = clean(raw)\n"
         "df['split'] = time_split(df)\n"
         "print('rows after clean:', len(df))\n"
         "display(df['disposition'].value_counts())\n"
         "print('overall decline rate: {:.1%}'.format(df['declined'].mean()))\n"
         "display(df.groupby('split')['declined'].agg(n='size', rate='mean'))"),
        ("md",
         "## Why a chronological split\n"
         "The model is meant to score a request as it arrives, using only history. A "
         "time-ordered split — earliest 60% train, next 20% validation, last 20% test — "
         "matches that and stops near-identical requests from the same shift landing in "
         "two partitions. The validation split is what the operating threshold is tuned on "
         "in notebook 05; the test split is never touched until the final evaluation."),
        ("code",
         "checks = validate(df)\n"
         "print(json.dumps(checks, indent=2, default=str))\n"
         "assert checks['no_exact_duplicates']\n"
         "assert checks['target_is_binary']\n"
         "assert checks['splits_are_time_ordered']"),
        ("code",
         "prepared_path = ROOT / 'data' / 'synthetic' / 'prepared_transfer_requests.csv'\n"
         "df.to_csv(prepared_path, index=False)\n"
         "print('wrote', prepared_path.relative_to(ROOT))"),
        ("md",
         "## What this establishes\n"
         "A reproducible cohort with a clear target and an honest split. It does **not** "
         "establish clinical or operational realism — the decline rate, the missingness "
         "pattern, and the payer mix are all properties of the synthetic generator, not "
         "findings about any real transfer center."),
    ],
)

write(
    "02_exploratory_analysis.ipynb",
    "# 02 · Explore the request funnel and demand\n\n"
    "**Question:** where do requests come from, when, and where do declines concentrate?\n\n"
    "This notebook looks at the **training split only**. Test-set numbers live in notebook 05.",
    [
        ("code",
         "df = pd.read_csv(ROOT / 'data' / 'synthetic' / 'prepared_transfer_requests.csv', parse_dates=['request_datetime'])\n"
         "train = df[df['split'] == 'train'].copy()\n"
         "print('training requests:', len(train))"),
        ("md", "## The funnel"),
        ("code",
         "funnel = train['disposition'].value_counts()\n"
         "display(funnel)\n"
         "ax = funnel.plot.barh(color='#2f6f4e', title='Request disposition (training split)')\n"
         "ax.invert_yaxis(); plt.xlabel('requests'); plt.show()\n"
         "print('decline reasons:')\n"
         "display(train.loc[train['declined'] == 1, 'decline_reason'].value_counts())"),
        ("md", "## Demand: who sends, and when"),
        ("code",
         "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
         "train['referring_facility_type'].value_counts().plot.barh(ax=axes[0], color='#315b83', title='Referring facility type')\n"
         "axes[0].invert_yaxis()\n"
         "order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']\n"
         "train['request_datetime'].dt.day_name().value_counts().reindex(order).plot.bar(ax=axes[1], color='#b0792b', title='Requests by weekday')\n"
         "plt.tight_layout(); plt.show()"),
        ("code",
         "hourly = train.groupby('request_hour')['declined'].agg(n='size', rate='mean')\n"
         "fig, ax1 = plt.subplots()\n"
         "ax1.bar(hourly.index, hourly['n'], color='#d8d2c4'); ax1.set_xlabel('hour of day'); ax1.set_ylabel('requests')\n"
         "ax2 = ax1.twinx(); ax2.plot(hourly.index, hourly['rate'], color='#c0392b', marker='o'); ax2.set_ylabel('decline rate')\n"
         "plt.title('Volume and decline rate by hour (training split)'); plt.show()"),
        ("md", "## Where declines concentrate"),
        ("code",
         "for col in ['requested_level_of_care', 'requested_service_line', 'acuity_score']:\n"
         "    display(train.groupby(col)['declined'].agg(n='size', rate='mean').sort_values('rate', ascending=False))"),
        ("code",
         "import numpy as np\n"
         "train['medsurg_occ_band'] = pd.cut(train['medsurg_occupancy_pct'], [0, 0.85, 0.95, 1.01], labels=['<85%','85-95%','>95%'])\n"
         "display(train.groupby('medsurg_occ_band', observed=True)['declined'].agg(n='size', rate='mean'))\n"
         "train.groupby('medsurg_occ_band', observed=True)['declined'].mean().plot.bar(color='#2f6f4e', title='Decline rate by med-surg occupancy band'); plt.show()"),
        ("md",
         "## Reading it\n"
         "Declines track capacity (occupancy, ED boarding) and acuity far more than time of "
         "day. `No bed capacity` is the dominant decline reason. These are the relationships "
         "the generator was built with — the value here is that the later model agrees with "
         "a picture we can already see, not that we have discovered how transfer centers "
         "behave."),
    ],
)

write(
    "03_feature_engineering.ipynb",
    "# 03 · Build the feature matrix\n\n"
    "**Question:** which columns are legitimately available at request time, and how are "
    "they encoded without leaking across the split?",
    [
        ("md",
         "## Leakage: what has to stay out\n"
         "Anything recorded after the transfer-center decision, or that encodes the decision "
         "itself, cannot be a feature. Using it would let the model 'predict' the decision "
         "from its own consequences."),
        ("code",
         "from transfer_decline.data import POST_DECISION_COLUMNS\n"
         "from transfer_decline.features import FEATURE_COLUMNS, FORBIDDEN, NUMERIC_FEATURES, CATEGORICAL_FEATURES\n"
         "print('excluded as post-decision / leakage:')\n"
         "for c in POST_DECISION_COLUMNS: print('  -', c)\n"
         "print()\n"
         "print('payer is also excluded from the model — audited separately in notebook 04')\n"
         "assert not (set(FEATURE_COLUMNS) & FORBIDDEN)"),
        ("code",
         "df = pd.read_csv(ROOT / 'data' / 'synthetic' / 'prepared_transfer_requests.csv')\n"
         "print(f'{len(FEATURE_COLUMNS)} features')\n"
         "print('numeric:', NUMERIC_FEATURES)\n"
         "print('categorical:', CATEGORICAL_FEATURES)\n"
         "from transfer_decline.features import build_feature_frame\n"
         "X = build_feature_frame(df)\n"
         "display(X.head())"),
        ("md",
         "## Encoding, fit on training only\n"
         "`make_encoder()` is a `ColumnTransformer`: median imputation + standardisation for "
         "numerics (with a missing-indicator column), most-frequent imputation + one-hot for "
         "categoricals, rare levels (< 20) folded together. It is `fit` on the training rows "
         "and only `transform`-ed on validation and test."),
        ("code",
         "from transfer_decline.features import make_encoder, TARGET\n"
         "train = df[df['split'] == 'train']\n"
         "encoder = make_encoder().fit(build_feature_frame(train), train[TARGET])\n"
         "matrix = encoder.transform(build_feature_frame(df.head()))\n"
         "print('encoded width:', matrix.shape[1])\n"
         "print(list(encoder.get_feature_names_out())[:12], '...')"),
        ("md",
         "## Note\n"
         "The encoder object is not saved here — the modelling pipeline in notebook 05 "
         "rebuilds and refits it as its first step, so training-only fitting is guaranteed "
         "by construction."),
    ],
)

write(
    "04_payer_equity_audit.ipynb",
    "# 04 · Payer-equity audit\n\n"
    "**Question:** is the decline decision associated with payer — and does any raw "
    "difference survive adjustment for case mix?",
    [
        ("md",
         "## Two tests\n"
         "1. **Chi-square** tests of independence: payer × decision, and med-surg occupancy "
         "band × decision (a check that the test picks up the association the generator "
         "*does* build in).\n"
         "2. **Multivariable logistic regression** of `declined` on payer, adjusting for "
         "acuity, ICU and med-surg occupancy, ED boarding, service line, and level of care. "
         "This separates 'this payer is declined more' from 'this payer's requests are "
         "sicker / arrive when the hospital is full'."),
        ("code",
         "df = pd.read_csv(ROOT / 'data' / 'synthetic' / 'prepared_transfer_requests.csv')\n"
         "from transfer_decline.equity import chi_square_tests, decline_rate_by_payer, payer_logit, payer_logit_summary\n"
         "display(decline_rate_by_payer(df))"),
        ("code",
         "chi = chi_square_tests(df)\n"
         "print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'table'} for k, v in chi.items()}, indent=2))"),
        ("md",
         "The occupancy test is significant — as it should be, since capacity drives "
         "declines in the generator. The payer test is the one under audit."),
        ("code",
         "display(payer_logit(df))\n"
         "summary = payer_logit_summary(df)\n"
         "print(json.dumps({k: v for k, v in summary.items() if k != 'odds_ratios'}, indent=2))"),
        ("md",
         "## Reading it\n"
         "In this synthetic data payer is assigned independently of the decision, so the "
         "audit comes back null: no payer term is significant after adjustment. The model in "
         "notebook 05 still **excludes payer entirely** — a clean audit today is not a "
         "guarantee for tomorrow, and a model that never sees payer cannot learn to use it. "
         "That is a design safeguard, not a reaction to a finding.\n\n"
         "On real data this same audit could show a disparity; it would then be a starting "
         "point for investigation, not a conclusion about intent."),
    ],
)

write(
    "05_decline_model.ipynb",
    "# 05 · Predict declines and choose an operating threshold\n\n"
    "**Question:** how well can a logistic-regression model flag likely declines at "
    "request time, and where should the decision threshold sit?\n\n"
    "Run `python scripts/build_project.py` first — this notebook reads the executed "
    "pipeline's artifacts rather than recomputing them.",
    [
        ("code",
         "metrics = json.loads((ROOT / 'reports' / 'metrics.json').read_text())\n"
         "selection = json.loads((ROOT / 'artifacts' / 'model_selection.json').read_text())\n"
         "manifest = json.loads((ROOT / 'artifacts' / 'run_manifest.json').read_text())\n"
         "print(json.dumps(manifest, indent=2))"),
        ("md",
         "## Threshold chosen on validation, not test\n"
         "The default 0.5 cutoff is just the midpoint of a probability scale. What matters is "
         "the cost of each error: a **missed decline** (no early warning) is set to cost 5×, "
         "a **false alarm** (a coordinator double-checks a request that was fine) 1×. The "
         "threshold that minimises expected weighted cost is picked on the **validation** "
         "split, then frozen."),
        ("code",
         "print(json.dumps(selection, indent=2))\n"
         "curve = pd.read_csv(ROOT / 'reports' / 'threshold_cost_curve.csv')\n"
         "ax = curve.plot(x='threshold', y='weighted_cost', legend=False, color='#2f6f4e')\n"
         "ax.axvline(selection['threshold'], color='#c0392b', ls='--')\n"
         "ax.axvline(0.5, color='#888', ls=':')\n"
         "ax.set_ylabel('weighted error cost (validation)'); plt.title('Threshold selection'); plt.show()"),
        ("md", "## Held-out test performance"),
        ("code",
         "rows = []\n"
         "for split in ['train', 'validation', 'test']:\n"
         "    m = metrics['splits'][split]\n"
         "    rows.append({'split': split, **{k: m[k] for k in ['n','roc_auc','pr_auc','brier','precision','recall','f1','specificity','weighted_cost']}})\n"
         "display(pd.DataFrame(rows).set_index('split').round(3))\n"
         "print('test confusion matrix at operating threshold:', metrics['splits']['test']['confusion_matrix'])\n"
         "print('test confusion matrix at default 0.50     :', metrics['splits']['test_default_threshold']['confusion_matrix'])"),
        ("code",
         "from IPython.display import Image\n"
         "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'confusion_test.png')))\n"
         "display(Image(filename=str(ROOT / 'reports' / 'figures' / 'calibration_test.png')))"),
        ("md", "## What drives the score"),
        ("code",
         "coef = pd.read_csv(ROOT / 'reports' / 'coefficients.csv')\n"
         "display(coef.head(15).round(3))"),
        ("md",
         "## Operational reading\n"
         "The model recovers the capacity-and-acuity story from notebook 02: occupancy, ED "
         "boarding, ICU level of care, and higher acuity push decline odds up. Moving the "
         "threshold below 0.5 trades some precision for materially higher recall on declines, "
         "which is the point when a missed decline is the costlier error.\n\n"
         "**Limits.** Synthetic data with designed relationships; a single logistic model; "
         "illustrative rather than measured error costs; the chronological split still shares "
         "a generator across periods. This supports a reproducible workflow and a defensible "
         "threshold argument — not a claim about real transfer-center performance or about "
         "any payer group."),
    ],
)

print("wrote 5 notebooks to notebooks/")

# HarborLink course materials

Teaching materials for **chapters 1 to 7** of an AI and Data Science lesson. The rendered pages are in `_book/`. Open `_book/index.html`.

Chapters 2, 3, and 4 use one fictitious subscription company. Those datasets are entirely synthetic. Chapter 5 then uses a customer-level telco table plus generated communications.

## What this repository contains

| Path | Role |
| --- | --- |
| `index.qmd` | Chapter 1: what each chapter does, and which file to open |
| `lessons/02b-data-loading.qmd` | Chapter 2: loading and extracting data |
| `lessons/02c-exploratory-analysis.qmd` | Chapter 3: one analysis table from several sources |
| `lessons/02d-eda-visualisation.qmd` | Chapter 4: EDA and visualisation |
| `lessons/02e-churn-text-ml.qmd` | Chapter 5: multimodal churn model |
| `lessons/02f-campaign-targeting.qmd` | Chapter 6: economic targeting under a campaign cost |
| `lessons/02g-cloud-automl.qmd` | Chapter 7: local logistic regression and cloud AutoML |
| `_book/` | The HTML pages, with chapters 1 to 7 in the left-hand menu |
| `notebooks/` | Practice notebooks for the same demonstrations |
| `data/raw/` | Small chapter 2 extracts |
| `data/raw/analysis/` | Larger panel used to build the analysis table, plus `*_variables.csv` descriptions |
| `data/raw/telco/` | Telco profiles and communication CSVs used in chapter 5 |
| `data/processed/harborlink_invoices.csv` | Invoice-level table used for EDA |
| `data/processed/harborlink_invoices_variables.csv` | Forwarded descriptions of the merged columns |
| `data/processed/harborlink_invoices_dictionary.xlsx` | Glossary workbook built from those descriptions |
| `data/processed/harborlink_invoices_summary.xlsx` | Column summaries with hover notes and histograms |
| `data/processed/telco/` | Metrics, plots and LIME outputs from the model |
| `python/generate_data.py` | Rebuilds the small chapter 2 files |
| `python/generate_analysis_data.py` | Rebuilds the larger panel |
| `python/data_dictionary.py` | General: data + descriptions → dictionary and summary workbooks |
| `python/build_analysis_frame.py` | Reproducible join pipeline (data and descriptions) |
| `python/eda_visualisation.py` | Helper to reload the invoice table |
| `python/mock_api.py` | Local API |
| `python/train_churn_model.py` | Retrains a tabular-plus-text model from paths and column lists you pass in |
| `python/workshop_targeting.py` | Expected-cost targeting: individual decisions and a budget |
| `python/workshop_automl.py` | Local exports for a tabular AutoML comparison, plus the approved cloud run |
| `python/campaign_scenario.py` | Builds the saved test scores and the simulated campaign probabilities for chapter 6 |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python -m ipykernel install --user \
  --name data-extraction-lesson \
  --display-name "Python (data-extraction-lesson)"
```

## Regenerating data

```bash
python python/generate_data.py
python python/generate_analysis_data.py
python python/build_analysis_frame.py
```

The chapter 5 tables are already in `data/raw/telco/`. To rebuild them from the IBM source file:

```bash
python python/generate_profiles.py
python python/generate_communications.py
# train_churn_model.py needs TABULAR_FILE, TEXT_FILE, ID_COLUMN, TARGET_COLUMN,
# POSITIVE_LABEL, NUMERIC_FEATURES and CATEGORICAL_FEATURES in the environment.
```

`generate_communications_2.py` is an optional LLM rewrite and needs extra packages (`torch`, `transformers`, `accelerate`) that are not in the default environment.

## Teaching

Open the Quarto file and run the cells in order. Optionally: `quarto render`.

The documents request the `data-extraction-lesson` kernel.

- **Chapter 1** — the map of the book (`index.qmd`, `_book/index.html`).
- **Chapter 2** — sources and extraction (`data/raw/`).
- **Chapter 3** — joins (`data/raw/analysis/` → `harborlink_invoices.csv`), forwarding each source's `*_variables.csv` into an Excel dictionary.
- **Chapter 4** — EDA on that CSV with matplotlib and seaborn. The emphasis is the **meaning** of each graph, not a catalogue of chart types.
- **Chapter 5** — multimodal churn model on `data/raw/telco/` (tabular fields + communications).
- **Chapter 6** — whom to target with a retention campaign, using the saved chapter 5 test scores. The render does not refit the model.
- **Chapter 7** — the same customers, scored by tabular AutoML as well as by the local logistic regressions. `quarto render lessons/02g-cloud-automl.qmd` reads saved files and does not call Google Cloud. The paid training command is `python python/workshop_automl.py run --approve-spend`, after the setup in that lesson.

## Dependencies

| Package | Why it is needed |
| --- | --- |
| `pandas` | DataFrames, loading and joins |
| `openpyxl` | Excel files |
| `requests` | Local API demonstration |
| `numpy` | Numerical arrays in chapters 4, 5, 6, and 7 |
| `pymupdf` | PDF extraction in chapter 2 |
| `reportlab`, `Pillow` | Synthetic invoice PDFs |
| `matplotlib`, `seaborn` | Graphs in chapters 3 and 4 |
| `scikit-learn` | Encoding, logistic regression, and metrics in chapters 5 and 7 |
| `scipy` | Budgeted targeting in chapter 6 |
| `lime` | Local explanations written by `python/train_churn_model.py` |
| `google-cloud-aiplatform`, `google-cloud-storage` | The paid AutoML run in chapter 7 |
| `jupyter`, `ipykernel` | Notebooks and Quarto |

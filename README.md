# HarborLink course materials

Teaching materials for **Sections 2.b, 2.c, 2.d and 2.e** of an AI and Data Science lesson.

The HarborLink documents (2.b–2.d) use one fictitious subscription company. Those datasets are entirely synthetic. Section 2.e then uses a customer-level telco table plus generated communications.

## What this repository contains

| Path | Role |
| --- | --- |
| `lessons/02b-data-loading.qmd` | Section 2.b: loading and extracting data |
| `lessons/02c-exploratory-analysis.qmd` | Section 2.c: one analysis table from several sources |
| `lessons/02d-eda-visualisation.qmd` | Section 2.d: EDA and visualisation |
| `lessons/02e-churn-text-ml.qmd` | Section 2.e: multimodal churn model |
| `notebooks/` | Practice notebooks for the same demonstrations |
| `data/raw/` | Small 2.b extracts |
| `data/raw/analysis/` | Larger panel used to build the analysis table, plus `*_variables.csv` descriptions |
| `data/raw/telco/` | Telco profiles and communication CSVs used in 2.e |
| `data/processed/harborlink_invoices.csv` | Invoice-level table used for EDA |
| `data/processed/harborlink_invoices_variables.csv` | Forwarded descriptions of the merged columns |
| `data/processed/harborlink_invoices_dictionary.xlsx` | Glossary workbook built from those descriptions |
| `data/processed/harborlink_invoices_summary.xlsx` | Column summaries with hover notes and histograms |
| `data/processed/telco/` | Metrics, plots and LIME outputs from the model |
| `python/generate_data.py` | Rebuilds the small 2.b files |
| `python/generate_analysis_data.py` | Rebuilds the larger panel |
| `python/data_dictionary.py` | General: data + descriptions → dictionary and summary workbooks |
| `python/build_analysis_frame.py` | Reproducible join pipeline (data and descriptions) |
| `python/eda_visualisation.py` | Helper to reload the invoice table |
| `python/mock_api.py` | Local API |
| `python/train_churn_model.py` | Retrains the 2.e model from the telco CSVs |

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

The 2.e tables are already in `data/raw/telco/`. To rebuild them from the IBM source file:

```bash
python python/generate_profiles.py
python python/generate_communications.py
python python/train_churn_model.py
```

`generate_communications_2.py` is an optional LLM rewrite and needs extra packages (`torch`, `transformers`, `accelerate`) that are not in the default environment.

## Teaching

Open the Quarto file and run the cells in order. Optionally: `quarto render`.

The documents request the `data-extraction-lesson` kernel.

- **2.b** — sources and extraction (`data/raw/`).
- **2.c** — joins (`data/raw/analysis/` → `harborlink_invoices.csv`), forwarding each source's `*_variables.csv` into an Excel dictionary.
- **2.d** — EDA on that CSV with matplotlib and seaborn. The emphasis is the **meaning** of each graph, not a catalogue of chart types.
- **2.e** — multimodal churn model on `data/raw/telco/` (tabular fields + communications).

## Dependencies

| Package | Why it is needed |
| --- | --- |
| `pandas` | DataFrames, loading and joins |
| `openpyxl` | Excel files |
| `requests` | Local API demonstration |
| `pymupdf` | PDF extraction in 2.b |
| `reportlab`, `Pillow` | Synthetic invoice PDFs |
| `matplotlib`, `seaborn` | Exploratory graphs in 2.c |
| `scikit-learn` | Encoding, logistic regression and metrics in 2.e |
| `lime` | Local explanations in 2.e |
| `jupyter`, `ipykernel` | Notebooks and Quarto |

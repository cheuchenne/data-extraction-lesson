"""Compare saved local probabilities with tabular AutoML, without spending by accident.

The Quarto document only reads files this module has already written.
``prepare`` builds those files on your machine. ``run`` is the only command
that creates Google Cloud resources, and it refuses to do so until you
approve the spend. ``cleanup`` deletes the resources that ``run`` recorded.

The cloud path uses the current tabular classification API:

- Python package ``google-cloud-aiplatform``
- ``TabularDataset`` and ``AutoMLTabularTrainingJob``
- batch prediction on the model resource, with no deployed endpoint

Official pages used for this workflow
--------------------------------------
https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/overview
https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/train-model
https://cloud.google.com/vertex-ai/docs/tabular-data/data-splits
https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/prepare-data
https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/get-batch-predictions
https://cloud.google.com/vertex-ai/docs/tabular-data/bp-tabular
https://cloud.google.com/vertex-ai/pricing
https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.AutoMLTabularTrainingJob

The product pages are titled Gemini Enterprise Agent Platform. The API host
and the Python classes are still Vertex AI. This file uses that current API.
It does not call the retired AutoML Tables client.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import python.workshop_churn as workshop
import python.workshop_targeting as targeting

# The Quarto document and ``use_course_example`` assign these.
CLOUD_MODE = "LOAD"
RANDOM_STATE = 42
TEST_SIZE = 0.25
DECISION_THRESHOLD = 0.5
N_TEXT_COMPONENTS = 64
HOLDOUT_VALIDATION_FRACTION = 0.5
PLATFORM_VALIDATE_FRACTION = 0.10
PLATFORM_TEST_FRACTION = 0.10
BUDGET_MILLI_NODE_HOURS = 1000
INCLUDE_TEXT_ONLY = False
BUDGET = 8000.0
OPTIMIZATION_OBJECTIVE = "minimize-log-loss"

ID_COLUMN = "id"
TARGET_COLUMN = "target"
POSITIVE_LABEL = "1"
TEXT_COLUMN = "text"
TEXT_ORDER_COLUMN = "text_index"
TEXT_TYPE_COLUMN = ""
TEXT_FEATURE = "text_feature"
SPLIT_COLUMN = "ml_split"
OBSERVED_COLUMN = "observed_outcome"
MODEL_ROWS_ID_COLUMN = "id"
# The tabular data-splits page requires these three words.
# The Python SDK docstring still says training/validation/test.
# Invalid values are ignored, so the wrong word drops the row instead of
# training on it. This file follows the tabular data-splits page.
SPLIT_LABELS = ("TRAIN", "VALIDATE", "TEST")

NUMERIC_FEATURES: list[str] = []
CATEGORICAL_FEATURES: list[str] = []
TABULAR_FILE = ""
TEXT_FILE = ""
ECONOMICS_FILE = ""
ECONOMICS_ID_COLUMN = "id"
ECONOMICS_P1_COLUMN = "p1"
ECONOMICS_COST_COLUMN = "cost"
ECONOMICS_LOSS_COLUMN = "loss"
SAVED_PROBABILITY_FILE = ""
SAVED_PROBABILITY_ID = "id"
SAVED_PROBABILITY_COLUMN = "p0"
SAVED_OUTCOME_COLUMN = "observed_outcome"
MODEL_ROWS_FILE = ""

OUTPUT_DIR = ROOT / "data" / "processed" / "automl"
GCS_PREFIX = "automl"
REGION = "us-central1"
MINIMUM_ROWS = 1000

LOCAL_TABULAR = "p_logistic_tabular"
LOCAL_TEXT = "p_logistic_text"
AUTOML_TABULAR = "p_automl_tabular"
AUTOML_TEXT = "p_automl_text"
AUTOML_TEXT_ONLY = "p_automl_text_only"

PROBABILITY_FILE_NAME = "validation_probabilities.csv"
MANIFEST_NAME = "manifest.json"
RESOURCES_NAME = "cloud_resources.json"
CLEANUP_SCRIPT_NAME = "cleanup_commands.sh"

NAVY = "#1f4e79"
CLAY = "#b85c38"
GREEN = "#2f6f4e"
GREY = "#7a7a7a"

_LOADED: dict | None = None


class AutoMLWorkshopError(Exception):
    """A problem the participant can correct from the workshop instructions."""


def _ensure_output_dir() -> None:
    """Accept the folder path as written in the lesson, including a relative path."""
    global OUTPUT_DIR
    path = Path(OUTPUT_DIR)
    if not path.is_absolute():
        path = ROOT / path
    OUTPUT_DIR = path


def use_course_example() -> None:
    """Fill the capital-letter settings with the course example.

    The lesson assigns the same names again after this call. Change the
    outcome by editing ``TARGET_COLUMN`` there.
    """
    global ID_COLUMN, TARGET_COLUMN, POSITIVE_LABEL
    global TEXT_COLUMN, TEXT_ORDER_COLUMN, TEXT_TYPE_COLUMN, TEXT_FEATURE
    global NUMERIC_FEATURES, CATEGORICAL_FEATURES
    global TABULAR_FILE, TEXT_FILE
    global ECONOMICS_FILE, ECONOMICS_ID_COLUMN, ECONOMICS_P1_COLUMN
    global ECONOMICS_COST_COLUMN, ECONOMICS_LOSS_COLUMN
    global SAVED_PROBABILITY_FILE, SAVED_PROBABILITY_ID
    global SAVED_PROBABILITY_COLUMN, SAVED_OUTCOME_COLUMN
    global MODEL_ROWS_FILE, MODEL_ROWS_ID_COLUMN, OBSERVED_COLUMN
    global OUTPUT_DIR, GCS_PREFIX, SPLIT_COLUMN
    global RANDOM_STATE, TEST_SIZE, DECISION_THRESHOLD, N_TEXT_COMPONENTS
    global HOLDOUT_VALIDATION_FRACTION, BUDGET_MILLI_NODE_HOURS
    global INCLUDE_TEXT_ONLY, BUDGET, OPTIMIZATION_OBJECTIVE

    ID_COLUMN = "customerID"
    TARGET_COLUMN = "Churn"
    POSITIVE_LABEL = "Yes"
    TEXT_COLUMN = "text"
    TEXT_ORDER_COLUMN = "text_index"
    TEXT_TYPE_COLUMN = "communication_type"
    TEXT_FEATURE = "customer_text"
    SPLIT_COLUMN = "ml_split"
    OBSERVED_COLUMN = "observed_outcome"
    NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
    CATEGORICAL_FEATURES = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    TABULAR_FILE = "data/raw/telco/telco_customers.csv"
    TEXT_FILE = "data/raw/telco/customer_communications.csv"
    ECONOMICS_FILE = "data/processed/telco/campaign_example_population.csv"
    ECONOMICS_ID_COLUMN = "customer_id"
    ECONOMICS_P1_COLUMN = "p1"
    ECONOMICS_COST_COLUMN = "example_cost"
    ECONOMICS_LOSS_COLUMN = "example_loss"
    SAVED_PROBABILITY_FILE = "data/processed/telco/test_probabilities_with_text.csv"
    SAVED_PROBABILITY_ID = "customer_id"
    SAVED_PROBABILITY_COLUMN = "p0"
    SAVED_OUTCOME_COLUMN = "observed_outcome"
    MODEL_ROWS_FILE = "data/processed/telco/model_rows.csv"
    MODEL_ROWS_ID_COLUMN = "customer_id"
    OUTPUT_DIR = ROOT / "data" / "processed" / "telco" / "automl"
    GCS_PREFIX = "telco-automl"
    RANDOM_STATE = 42
    TEST_SIZE = 0.25
    DECISION_THRESHOLD = 0.5
    N_TEXT_COMPONENTS = 64
    HOLDOUT_VALIDATION_FRACTION = 0.5
    BUDGET_MILLI_NODE_HOURS = 1000
    INCLUDE_TEXT_ONLY = False
    BUDGET = 8000.0
    OPTIMIZATION_OBJECTIVE = "minimize-log-loss"


def active_settings() -> pd.DataFrame:
    """Show the settings that control the split, the exports, and the budget."""
    rows = [
        ("CLOUD_MODE", CLOUD_MODE),
        ("TABULAR_FILE", TABULAR_FILE),
        ("TEXT_FILE", TEXT_FILE),
        ("ID_COLUMN", ID_COLUMN),
        ("TARGET_COLUMN", TARGET_COLUMN),
        ("POSITIVE_LABEL", POSITIVE_LABEL),
        ("NUMERIC_FEATURES", ", ".join(NUMERIC_FEATURES)),
        ("CATEGORICAL_FEATURES", ", ".join(CATEGORICAL_FEATURES)),
        ("TEXT_FEATURE", TEXT_FEATURE),
        ("SPLIT_COLUMN", SPLIT_COLUMN),
        ("OBSERVED_COLUMN", OBSERVED_COLUMN),
        ("MODEL_ROWS_ID_COLUMN", MODEL_ROWS_ID_COLUMN),
        ("GCS_PREFIX", GCS_PREFIX),
        ("RANDOM_STATE", RANDOM_STATE),
        ("TEST_SIZE", TEST_SIZE),
        ("HOLDOUT_VALIDATION_FRACTION", HOLDOUT_VALIDATION_FRACTION),
        ("PLATFORM_VALIDATE_FRACTION", PLATFORM_VALIDATE_FRACTION),
        ("PLATFORM_TEST_FRACTION", PLATFORM_TEST_FRACTION),
        ("DECISION_THRESHOLD", DECISION_THRESHOLD),
        ("OPTIMIZATION_OBJECTIVE", OPTIMIZATION_OBJECTIVE),
        ("BUDGET_MILLI_NODE_HOURS", BUDGET_MILLI_NODE_HOURS),
        ("INCLUDE_TEXT_ONLY", INCLUDE_TEXT_ONLY),
        ("BUDGET", BUDGET),
        ("ECONOMICS_FILE", ECONOMICS_FILE),
        ("SAVED_PROBABILITY_FILE", SAVED_PROBABILITY_FILE),
        ("MODEL_ROWS_FILE", MODEL_ROWS_FILE),
        ("OUTPUT_DIR", _display_output_dir()),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value"])


def documentation_links() -> pd.DataFrame:
    """Official pages for the API this module calls."""
    rows = [
        ("Overview", "https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/overview"),
        ("Train a model", "https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/train-model"),
        ("Data splits", "https://cloud.google.com/vertex-ai/docs/tabular-data/data-splits"),
        ("Prepare data", "https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/prepare-data"),
        ("Batch prediction", "https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/get-batch-predictions"),
        ("Text in a table", "https://cloud.google.com/vertex-ai/docs/tabular-data/bp-tabular"),
        ("Pricing", "https://cloud.google.com/vertex-ai/pricing"),
        ("Python training job", "https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.AutoMLTabularTrainingJob"),
    ]
    return pd.DataFrame(rows, columns=["Topic", "URL"])


def cost_table(plan: list[dict] | None = None) -> pd.DataFrame:
    """Cost components for the jobs this workshop is allowed to start.

    Dollar rates are on the pricing page and differ by region. This table
    states the unit and the cap used here. It does not start a job.
    """
    hours = int(BUDGET_MILLI_NODE_HOURS) / 1000.0
    scheduled = plan if plan is not None else _cost_rows_from_manifest()
    rows = []
    for experiment in scheduled:
        action = experiment.get("action", "train")
        if action == "skip":
            detail = "Already saved. No new job."
        elif action == "predict":
            detail = "Batch prediction only. The recorded model is reused."
        elif action == "wait":
            detail = "Wait for the recorded training pipeline, then batch predict."
        else:
            detail = f"Training cap {hours:g} node hour, then one batch prediction."
        rows.append({
            "Component": experiment["label"],
            "What is billed": "Training node hours, up to the cap, and batch-prediction node hours.",
            "This run": detail,
        })
    rows.extend([
        {
            "Component": "Online endpoint",
            "What is billed": "Node hours from deployment until the model is undeployed.",
            "This run": "Not created.",
        },
        {
            "Component": "Cloud Storage",
            "What is billed": "Storage of the uploaded CSV files and the prediction output.",
            "This run": (
                f"The files of one run are stored at "
                f"gs://<bucket>/{GCS_PREFIX}/<run id>/."
            ),
        },
        {
            "Component": "Model registry",
            "What is billed": "A model that is not deployed is not charged for prediction.",
            "This run": "Created. Delete it with the cleanup command when you are finished.",
        },
    ])
    return pd.DataFrame(rows)


def explain_costs(plan: list[dict] | None = None) -> str:
    """Full sentences for the same facts as ``cost_table``.

    Input: the optional plan ``cost_table`` accepts. With no argument, the
    experiment list comes from ``manifest.json`` when that file exists.
    Returns one paragraph per component. Does not start a job and does not
    compute a dollar amount.
    """
    lines = []
    for row in cost_table(plan).to_dict(orient="records"):
        lines.append(
            f"{row['Component']}. What is billed: {row['What is billed']} "
            f"This run: {row['This run']}"
        )
    return "\n\n".join(lines)


def _display_output_dir() -> str:
    path = Path(OUTPUT_DIR)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def gather_sources() -> pd.DataFrame:
    """One row per customer: the tabular columns plus the joined messages.

    Inputs are the capital-letter settings ``TABULAR_FILE``, ``TEXT_FILE``,
    ``ID_COLUMN``, ``TARGET_COLUMN``, ``TEXT_COLUMN``, ``TEXT_ORDER_COLUMN``
    and ``TEXT_TYPE_COLUMN``. No argument. No split. No Google call.

    Returns that table. The joined messages are the column ``TEXT_FEATURE``.
    """
    _ensure_output_dir()
    _apply_workshop_settings()
    table, messages = workshop.load_datasets(_path(TABULAR_FILE), _path(TEXT_FILE))
    documents = workshop._documents_for_ids(messages, table[ID_COLUMN])
    gathered = _attach_text(table, documents)
    print(f"Gathered {len(gathered)} customers. Joined messages are in {TEXT_FEATURE}.")
    return gathered


def separate_fit_test(gathered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the gathered table the way Chapter 5 split it.

    Inputs: the table from ``gather_sources``, plus ``RANDOM_STATE``,
    ``TEST_SIZE``, ``ID_COLUMN``, ``TARGET_COLUMN`` and ``POSITIVE_LABEL``.
    Returns ``(fit, test)``. A test row was not used to fit the logistic models.
    """
    _apply_workshop_settings()
    split = workshop.prepare_train_test(gathered)
    fit = split["train"].reset_index(drop=True)
    test = split["test"].reset_index(drop=True)
    return fit, test


def _holdout_test(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same cut ``_assign_groups`` uses on the Chapter 5 test rows."""
    y_test = workshop._positive_array(test[TARGET_COLUMN])
    comparison, final_test = train_test_split(
        test,
        test_size=1.0 - float(HOLDOUT_VALIDATION_FRACTION),
        random_state=int(RANDOM_STATE),
        stratify=y_test,
    )
    return comparison.reset_index(drop=True), final_test.reset_index(drop=True)


def _read_model_rows() -> pd.DataFrame | None:
    if not str(MODEL_ROWS_FILE).strip():
        return None
    path = _path(MODEL_ROWS_FILE)
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={MODEL_ROWS_ID_COLUMN: str})
    required = [MODEL_ROWS_ID_COLUMN, "role", LOCAL_TABULAR, LOCAL_TEXT, SAVED_OUTCOME_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AutoMLWorkshopError(f"{MODEL_ROWS_FILE} is missing {missing}.")
    return frame


def retrieve_results(
    fit: pd.DataFrame,
    test: pd.DataFrame,
    write: bool = False,
) -> dict:
    """Attach the saved scores and the campaign costs, then keep the test rows.

    Inputs: the fit table and the test table from ``separate_fit_test``.
    Also reads ``MODEL_ROWS_FILE`` when that file exists, otherwise
    ``SAVED_PROBABILITY_FILE`` and a new tabular fit on this split.
    ``ECONOMICS_FILE`` supplies ``p1``, ``cost`` and ``loss`` for the test rows.
    ``HOLDOUT_VALIDATION_FRACTION`` chooses which test rows go into the
    comparison file.

    Returns a dictionary.

    - ``fit``: fit rows. Scores are filled when ``MODEL_ROWS_FILE`` exists.
    - ``test``: every test row, with scores and costs.
    - ``comparison``: the test rows written for Gemini to complete.
    - ``final_test``: the other test rows. They are not scored.

    ``write=True`` stores ``comparison`` as ``validation_probabilities.csv``.
    A cloud column already in that file is kept. ``write=False`` reads the
    file when it exists and does not fit a model.
    """
    _ensure_output_dir()
    _apply_workshop_settings()
    fit = fit.reset_index(drop=True).copy()
    test = test.reset_index(drop=True).copy()
    fit[ID_COLUMN] = fit[ID_COLUMN].astype(str)
    test[ID_COLUMN] = test[ID_COLUMN].astype(str)
    saved_rows = _read_model_rows()
    probability_path = OUTPUT_DIR / PROBABILITY_FILE_NAME

    if not write and not probability_path.exists():
        raise AutoMLWorkshopError(
            f"{PROBABILITY_FILE_NAME} is not in {OUTPUT_DIR.relative_to(ROOT)}. "
            'Set CLOUD_MODE to "SAVE" to build it on this computer. That does not call Google.'
        )
    if not write:
        comparison = pd.read_csv(probability_path, dtype={ID_COLUMN: str})
        fit_out = fit[[ID_COLUMN]].copy()
        fit_out["role"] = "fit"
        test_out = _test_rows_from_saved(test, saved_rows, comparison)
        if saved_rows is not None:
            fit_scores = saved_rows.loc[saved_rows["role"] == "fit"].rename(columns={MODEL_ROWS_ID_COLUMN: ID_COLUMN})
            fit_out = fit_out.merge(
                fit_scores[[ID_COLUMN, "p_logistic_tabular", "p_logistic_text", "observed_outcome"]],
                on=ID_COLUMN,
                how="left",
                validate="one_to_one",
            )
        print(f"Loaded {PROBABILITY_FILE_NAME}. No model was fitted.")
        return {
            "fit": fit_out,
            "test": test_out,
            "comparison": comparison,
            "final_test": pd.DataFrame(),
            "groups": None,
        }

    comparison_rows, final_test = _holdout_test(test)
    annotated = pd.concat([fit, test], ignore_index=True)
    split = {"train": fit, "test": test}
    _require_saved_test_alignment(split)
    groups = _assign_groups(split, annotated)
    if set(groups["validation"][ID_COLUMN].astype(str)) != set(comparison_rows[ID_COLUMN].astype(str)):
        raise AutoMLWorkshopError("The comparison rows and the holdout split disagree.")
    probabilities = _local_probability_frame(split, groups)
    if saved_rows is not None:
        saved_test = saved_rows.loc[saved_rows["role"] == "test"].rename(columns={MODEL_ROWS_ID_COLUMN: ID_COLUMN})
        check = probabilities.merge(
            saved_test[[ID_COLUMN, LOCAL_TEXT]],
            on=ID_COLUMN,
            how="left",
            suffixes=("", "_saved"),
        )
        if check[LOCAL_TEXT + "_saved"].isna().any():
            raise AutoMLWorkshopError(f"A comparison row is missing from {MODEL_ROWS_FILE}.")
        gap = np.max(np.abs(check[LOCAL_TEXT].to_numpy(dtype=float) - check[LOCAL_TEXT + "_saved"].to_numpy(dtype=float)))
        if gap > 1e-8:
            raise AutoMLWorkshopError(
                f"{MODEL_ROWS_FILE} does not match the saved communications score. Largest gap: {gap:.3g}."
            )
        print(f"Communications scores match {MODEL_ROWS_FILE}.")
    if write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _write_probability_file(probabilities)
        print(f"Wrote {len(probabilities)} comparison rows to {PROBABILITY_FILE_NAME}.")
    fit_out = fit[[ID_COLUMN]].copy()
    fit_out["role"] = "fit"
    if saved_rows is not None:
        fit_scores = saved_rows.loc[saved_rows["role"] == "fit"].rename(columns={MODEL_ROWS_ID_COLUMN: ID_COLUMN})
        fit_out = fit_out.merge(
            fit_scores[[ID_COLUMN, LOCAL_TABULAR, LOCAL_TEXT, "observed_outcome"]],
            on=ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
    else:
        print(
            f"{MODEL_ROWS_FILE} is not on disk. "
            "Chapter 5 writes it with save_model_rows. Fit-row scores are not in the comparison file."
        )
    test_out = _test_rows_from_saved(test, saved_rows, probabilities)
    return {
        "fit": fit_out,
        "test": test_out,
        "comparison": probabilities if not write else pd.read_csv(probability_path, dtype={ID_COLUMN: str}),
        "final_test": final_test,
        "groups": groups,
    }


def _test_rows_from_saved(
    test: pd.DataFrame,
    saved_rows: pd.DataFrame | None,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Every test row, with the scores and costs that are already on disk."""
    base = test[[ID_COLUMN]].copy()
    base["role"] = "test"
    if saved_rows is not None:
        text = saved_rows.loc[saved_rows["role"] == "test"].rename(columns={MODEL_ROWS_ID_COLUMN: ID_COLUMN})
        base = base.merge(
            text[[ID_COLUMN, LOCAL_TABULAR, LOCAL_TEXT, "observed_outcome"]],
            on=ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
    else:
        saved = _saved_test_probabilities().rename(columns={
            SAVED_PROBABILITY_ID: ID_COLUMN,
            SAVED_PROBABILITY_COLUMN: LOCAL_TEXT,
            SAVED_OUTCOME_COLUMN: "observed_outcome",
        })
        base = base.merge(
            saved[[ID_COLUMN, LOCAL_TEXT, "observed_outcome"]],
            on=ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
        known = comparison[[ID_COLUMN, LOCAL_TABULAR]].copy()
        base = base.merge(known, on=ID_COLUMN, how="left", validate="one_to_one")
    economics = _economics_for_ids(base[ID_COLUMN])
    base = base.merge(economics, on=ID_COLUMN, how="left", validate="one_to_one")
    return base


def _write_probability_file(probabilities: pd.DataFrame) -> None:
    """Write the comparison rows and keep a cloud column that is already saved."""
    path = OUTPUT_DIR / PROBABILITY_FILE_NAME
    frame = probabilities.copy()
    if path.exists():
        old = pd.read_csv(path, dtype={ID_COLUMN: str})
        extra = [
            column for column in (AUTOML_TABULAR, AUTOML_TEXT, AUTOML_TEXT_ONLY)
            if column in old.columns and column not in frame.columns
        ]
        if extra and set(old[ID_COLUMN].astype(str)) == set(frame[ID_COLUMN].astype(str)):
            frame = frame.merge(old[[ID_COLUMN, *extra]], on=ID_COLUMN, how="left", validate="one_to_one")
    frame.to_csv(path, index=False)


def build_gemini_files(retrieved: dict, write: bool = False) -> pd.DataFrame:
    """Write the training upload and the scoring upload.

    Inputs: the dictionary from ``retrieve_results`` with ``write=True``
    (it holds ``groups``). Also reads ``PLATFORM_VALIDATE_FRACTION``,
    ``PLATFORM_TEST_FRACTION``, ``BUDGET_MILLI_NODE_HOURS``,
    ``INCLUDE_TEXT_ONLY`` and ``OPTIMIZATION_OBJECTIVE``.

    The fit rows become ``training_*.csv``, with ``ml_split`` added.
    The comparison rows become ``predict_validation_*.csv``, with no target.
    ``manifest.json`` stores the training cap and those paths.

    ``write=False`` does not change files. It returns the first saved training
    rows. ``write=True`` returns those rows after writing. Neither call reads
    ``CLOUD_MODE``, so a later ``load_workshop`` is a separate step.
    """
    if not write:
        return _preview_training("tabular_text", rows=2)
    groups = retrieved.get("groups")
    if groups is None:
        raise AutoMLWorkshopError(
            'build_gemini_files(write=True) needs retrieve_results(..., write=True) first.'
        )
    _validate_local_settings()
    experiments = _experiment_specs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = _write_experiment_files(groups, experiments)
    probabilities = pd.read_csv(OUTPUT_DIR / PROBABILITY_FILE_NAME, dtype={ID_COLUMN: str})
    manifest = _manifest(groups, experiments, written, probabilities)
    _write_json(OUTPUT_DIR / MANIFEST_NAME, manifest)
    print(f"Saved the Gemini files in {OUTPUT_DIR.relative_to(ROOT)}. No Google call was made.")
    return _preview_training("tabular_text", rows=2)


def _preview_training(experiment: str = "tabular_text", rows: int = 2) -> pd.DataFrame:
    """Read the first rows of a training file. Does not load the probability file."""
    path = OUTPUT_DIR / f"training_{experiment}.csv"
    if not path.exists():
        raise AutoMLWorkshopError(
            f"{path.name} is not in {OUTPUT_DIR.relative_to(ROOT)}. "
            'Set CLOUD_MODE to "SAVE" and run build_gemini_files again. That does not call Google.'
        )
    frame = pd.read_csv(path, nrows=int(rows), dtype={ID_COLUMN: str})
    if TEXT_FEATURE in frame.columns:
        frame[TEXT_FEATURE] = (
            frame[TEXT_FEATURE].fillna("").astype(str).str.replace("\n", " ", regex=False).str.slice(0, 140)
        )
    return frame


def prepare() -> pd.DataFrame:
    """Build local exports and local probabilities. Does not call Google Cloud."""
    global _LOADED
    _ensure_output_dir()
    _LOADED = None
    _validate_local_settings()
    gathered = gather_sources()
    fit, test = separate_fit_test(gathered)
    retrieved = retrieve_results(fit, test, write=True)
    build_gemini_files(retrieved, write=True)
    print("No Google Cloud resource was created.")
    audit = verify_saved_files()
    return audit


def load_workshop() -> dict:
    """Load the saved comparison file. A render uses this path.

    ``SAVE`` rebuilds the local files and still does not call Google Cloud.
    ``RUN`` is rejected here. The paid command is ``python python/workshop_automl.py run``.
    """
    global _LOADED
    _ensure_output_dir()
    mode = str(CLOUD_MODE).strip().upper()
    if mode == "RUN":
        print(
            "This document does not start a Google Cloud job. "
            "Read the cost table, then run python python/workshop_automl.py run --approve-spend."
        )
        mode = "LOAD"
    if mode == "SAVE":
        prepare()
        _LOADED = None
    elif mode != "LOAD":
        raise AutoMLWorkshopError('CLOUD_MODE must be "LOAD", "SAVE", or "RUN".')
    if _LOADED is not None:
        return _LOADED
    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    probability_path = OUTPUT_DIR / PROBABILITY_FILE_NAME
    if not manifest_path.exists() or not probability_path.exists():
        raise AutoMLWorkshopError(
            "Saved AutoML workshop files were not found in "
            f"{OUTPUT_DIR.relative_to(ROOT)}. "
            'Set CLOUD_MODE to "SAVE" for a local rebuild, or run '
            "python python/workshop_automl.py prepare. "
            "That command does not call Google Cloud."
        )
    manifest = _read_json(manifest_path)
    frame = pd.read_csv(probability_path, dtype={ID_COLUMN: str})
    _require_probability_frame(frame, manifest)
    automl_columns = [column for column in (AUTOML_TABULAR, AUTOML_TEXT, AUTOML_TEXT_ONLY) if column in frame.columns]
    if automl_columns:
        message = "Loaded saved AutoML probabilities from " + PROBABILITY_FILE_NAME + "."
    else:
        message = (
            "AutoML probabilities are not in the saved file. "
            "The tables below use the local logistic regressions. "
            "A render does not start a training job."
        )
    print(message)
    _LOADED = {
        "manifest": manifest,
        "split_summary": pd.DataFrame(manifest["split_summary"]),
        "probabilities": frame,
        "message": message,
        "automl_columns": automl_columns,
    }
    return _LOADED


def import_saved_scores(experiment: str = "tabular") -> pd.DataFrame:
    """Copy one finished batch prediction into the comparison file.

    Input: ``"tabular"``, ``"tabular_text"``, or ``"text"`` when that
    experiment is enabled. Reads ``manifest.json`` for the column name and
    ``cloud_resources.json`` for ``gcs_predictions``. Downloads the files
    named ``prediction.results-``. Writes ``p_automl_tabular`` or
    ``p_automl_text`` into ``validation_probabilities.csv``.

    Does not train a model and does not start a batch prediction.
    Run it again after a later job. The column already in the file is replaced.

    Returns the comparison file.
    """
    _ensure_output_dir()
    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    probability_path = OUTPUT_DIR / PROBABILITY_FILE_NAME
    if not manifest_path.exists() or not probability_path.exists():
        raise AutoMLWorkshopError(
            f"{MANIFEST_NAME} and {PROBABILITY_FILE_NAME} must already be in "
            f"{OUTPUT_DIR.relative_to(ROOT)}."
        )
    manifest = _read_json(manifest_path)
    match = [item for item in manifest["experiments"] if item["key"] == experiment]
    if not match:
        known = ", ".join(item["key"] for item in manifest["experiments"])
        raise AutoMLWorkshopError(
            f"manifest.json has no experiment {experiment!r}. Known keys: {known}."
        )
    run_record, prefix = _recorded_prediction_prefix(experiment)
    _, storage, credentials = _import_cloud()
    output = _download_result_csvs(storage, credentials, run_record, prefix, experiment)
    scores, score_column = _extract_positive_score(
        output, manifest["target_column"], manifest["positive_label"]
    )
    output = output.copy()
    output[match[0]["probability_column"]] = scores
    frame = pd.read_csv(probability_path, dtype={ID_COLUMN: str})
    _merge_automl_column(frame, output, match[0], manifest, score_column)
    print(
        f"Wrote {match[0]['probability_column']} for {len(frame)} comparison customers "
        f"from {prefix}."
    )
    return pd.read_csv(probability_path, dtype={ID_COLUMN: str})


def training_preview(experiment: str = "tabular_text", rows: int = 3) -> pd.DataFrame:
    """Show a few training rows. Long text is truncated for display."""
    manifest = load_workshop()["manifest"]
    match = [item for item in manifest["experiments"] if item["key"] == experiment]
    if not match:
        raise AutoMLWorkshopError(f"The saved manifest has no experiment {experiment!r}.")
    path = ROOT / match[0]["training_file"]
    frame = pd.read_csv(path, nrows=int(rows), dtype={ID_COLUMN: str})
    if TEXT_FEATURE in frame.columns:
        frame[TEXT_FEATURE] = (
            frame[TEXT_FEATURE].fillna("").astype(str).str.replace("\n", " ", regex=False).str.slice(0, 140)
        )
    return frame


def metrics_table(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score every saved probability column on the validation customers."""
    frame = _frame(frame)
    y = frame[OBSERVED_COLUMN].to_numpy(dtype=int)
    rows = []
    for column, label, _color, _style in _model_series(frame):
        scores = frame[column].to_numpy(dtype=float)
        _require_unit_interval(scores, column)
        flags = (scores >= float(DECISION_THRESHOLD)).astype(int)
        rows.append({
            "Model": label,
            "ROC-AUC": roc_auc_score(y, scores),
            "PR-AUC": average_precision_score(y, scores),
            "Accuracy": accuracy_score(y, flags),
            "Precision": precision_score(y, flags, zero_division=0),
            "Recall": recall_score(y, flags, zero_division=0),
            "Log-loss": log_loss(y, scores),
            "Brier": brier_score_loss(y, scores),
        })
    return pd.DataFrame(rows)


def plot_roc(frame: pd.DataFrame | None = None):
    """Validation ROC curves. The legend shows ROC-AUC."""
    frame = _frame(frame)
    y = frame[OBSERVED_COLUMN].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for column, label, color, style in _model_series(frame):
        scores = frame[column].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        ax.plot(fpr, tpr, color=color, linestyle=style, linewidth=2, label=f"{label} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1)
    ax.set_xlabel("Share of negative outcomes flagged")
    ax.set_ylabel("Share of positive outcomes flagged")
    ax.set_title("Validation customers")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_calibration(frame: pd.DataFrame | None = None):
    """Mean predicted probability against the observed positive rate."""
    frame = _frame(frame)
    y = frame[OBSERVED_COLUMN].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for column, label, color, style in _model_series(frame):
        scores = frame[column].to_numpy(dtype=float)
        fraction_positive, mean_predicted = calibration_curve(
            y, scores, n_bins=8, strategy="quantile"
        )
        ax.plot(
            mean_predicted,
            fraction_positive,
            color=color,
            linestyle=style,
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration on the validation customers")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def economic_table(frame: pd.DataFrame | None = None, budget: float | None = None) -> pd.DataFrame:
    """Targeting results when only the probability of the positive label changes."""
    frame = _frame(frame)
    budget = float(BUDGET if budget is None else budget)
    metrics = metrics_table(frame).set_index("Model")["ROC-AUC"]
    rows = []
    for column, label, _color, _style in _model_series(frame):
        open_plan, capped = _plans(frame, column, budget)
        rows.append({
            "Model": label,
            "ROC-AUC": float(metrics.loc[label]),
            "Customers targeted": open_plan["n_targeted"],
            "Campaign spend": open_plan["campaign_spend"],
            "Expected cost": open_plan["expected_cost"],
            "Expected net benefit": open_plan["benefit"],
            "Budget customers": capped["n_targeted"],
            "Budget spend": capped["campaign_spend"],
            "Budget expected cost": capped["expected_cost"],
            "Budget expected net benefit": capped["benefit"],
        })
    return pd.DataFrame(rows)


def realized_cost_table(frame: pd.DataFrame | None = None, budget: float | None = None) -> pd.DataFrame:
    """Historical cost of each contact list, with ``p1`` held fixed.

    The contact list is the Chapter 6 rule applied to the score. With no
    campaign, a customer who left costs ``loss`` and a customer who stayed
    costs 0. For a selected customer who left, that loss is replaced by
    ``cost + loss * p1``. For a selected customer who stayed, the cost is
    ``cost``. The net benefit is the no-campaign total minus the
    with-campaign total. The same quantities are reported under ``BUDGET``.

    ``p1`` is an input. The table does not estimate the effect of a campaign.
    """
    frame = _frame(frame)
    budget = float(BUDGET if budget is None else budget)
    outcome = frame[OBSERVED_COLUMN].to_numpy(dtype=int)
    p1 = frame["p1"].to_numpy(dtype=float)
    contact = frame["cost"].to_numpy(dtype=float)
    churn_loss = frame["loss"].to_numpy(dtype=float)
    without = float(np.sum(churn_loss * outcome))
    rows = []
    for column, label, _color, _style in _model_series(frame):
        open_plan, capped = _plans(frame, column, budget)
        open_cost = _realized_campaign_cost(open_plan["target"], outcome, p1, contact, churn_loss)
        capped_cost = _realized_campaign_cost(capped["target"], outcome, p1, contact, churn_loss)
        rows.append({
            "Model": label,
            "Cost without campaign": without,
            "Leavers contacted": open_cost["leavers"],
            "Stayers contacted": open_cost["stayers"],
            "Cost with campaign": open_cost["cost"],
            "Net benefit": without - open_cost["cost"],
            "Budget leavers contacted": capped_cost["leavers"],
            "Budget stayers contacted": capped_cost["stayers"],
            "Budget cost with campaign": capped_cost["cost"],
            "Budget net benefit": without - capped_cost["cost"],
        })
    return pd.DataFrame(rows)


def _realized_campaign_cost(
    target: np.ndarray,
    outcome: np.ndarray,
    p1: np.ndarray,
    contact: np.ndarray,
    churn_loss: np.ndarray,
) -> dict:
    """Replace the loss of a contacted leaver by ``cost + loss * p1``."""
    chosen = np.asarray(target, dtype=int) == 1
    left = np.asarray(outcome, dtype=int) == 1
    stayed = ~left
    cost = np.zeros(len(chosen), dtype=float)
    cost[left & ~chosen] = churn_loss[left & ~chosen]
    cost[left & chosen] = contact[left & chosen] + churn_loss[left & chosen] * p1[left & chosen]
    cost[stayed & chosen] = contact[stayed & chosen]
    return {
        "leavers": int((left & chosen).sum()),
        "stayers": int((stayed & chosen).sum()),
        "cost": float(np.sum(cost)),
    }


def plot_expected_cost(summary: pd.DataFrame):
    """Unconstrained expected cost for each saved model."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    positions = np.arange(len(summary))
    ax.barh(positions, summary["Expected cost"], color=NAVY, height=0.7)
    ax.set_yticks(positions, summary["Model"])
    ax.set_xlabel("Expected cost")
    ax.set_title("Same costs, losses, and campaign probabilities")
    fig.tight_layout()
    plt.close(fig)
    return fig


def ranking_versus_cost(frame: pd.DataFrame | None = None, budget: float | None = None) -> pd.DataFrame:
    """Whether the higher AUC is also the lower expected cost."""
    frame = _frame(frame)
    budget = float(BUDGET if budget is None else budget)
    summary = economic_table(frame, budget).set_index("Model")
    pairs = [
        (
            "The two logistic regressions",
            "Logistic regression, tabular",
            "Logistic regression, tabular and text",
        ),
        ("Tabular data", "Logistic regression, tabular", "AutoML, tabular"),
        (
            "Tabular data and communications",
            "Logistic regression, tabular and text",
            "AutoML, tabular and text",
        ),
    ]
    rows = []
    for comparison, local_name, automl_name in pairs:
        if local_name not in summary.index or automl_name not in summary.index:
            rows.append({
                "Comparison": comparison,
                "Higher ROC-AUC": "Not saved",
                "Lower expected cost": "Not saved",
                "Lower budgeted expected cost": "Not saved",
                "Different unconstrained decisions": "",
            })
            continue
        local_column = _column_for_label(frame, local_name)
        automl_column = _column_for_label(frame, automl_name)
        local_plan, _local_capped = _plans(frame, local_column, budget)
        automl_plan, _automl_capped = _plans(frame, automl_column, budget)
        changed = int(np.sum(local_plan["target"] != automl_plan["target"]))
        rows.append({
            "Comparison": comparison,
            "Higher ROC-AUC": _better_name(summary, local_name, automl_name, "ROC-AUC", higher=True),
            "Lower expected cost": _better_name(summary, local_name, automl_name, "Expected cost", higher=False),
            "Lower budgeted expected cost": _better_name(
                summary, local_name, automl_name, "Budget expected cost", higher=False
            ),
            "Different unconstrained decisions": changed,
        })
    return pd.DataFrame(rows)


def decision_note(comparison: pd.DataFrame) -> str:
    """State the AUC and cost result that is in the comparison table."""
    sentences = [
        "ROC-AUC measures ranking.",
        "Expected cost uses the probability values, the loss, the contact cost, and the campaign probability.",
    ]
    for row in comparison.to_dict(orient="records"):
        auc_name = str(row["Higher ROC-AUC"])
        cost_name = str(row["Lower expected cost"])
        budget_name = str(row["Lower budgeted expected cost"])
        label = str(row["Comparison"]).lower()
        if auc_name == "Not saved":
            sentences.append(f"In {label}, the AutoML probabilities are not in the saved file.")
            continue
        if auc_name == "Tie" and cost_name == "Tie":
            sentences.append(f"In {label}, ROC-AUC is tied and expected cost is tied.")
        elif auc_name == "Tie":
            sentences.append(f"In {label}, ROC-AUC is tied. {cost_name} has the lower expected cost.")
        elif cost_name == "Tie":
            sentences.append(f"In {label}, {auc_name} has the higher ROC-AUC. Expected cost is tied.")
        elif auc_name == cost_name:
            sentences.append(
                f"In {label}, {auc_name} has the higher ROC-AUC and the lower expected cost."
            )
        else:
            sentences.append(
                f"In {label}, {auc_name} has the higher ROC-AUC. "
                f"{cost_name} has the lower expected cost."
            )
        if budget_name not in {"", "Not saved", "nan"} and budget_name != cost_name:
            if budget_name == "Tie":
                sentences.append("Under the budget, expected cost is tied.")
            else:
                sentences.append(f"Under the budget, {budget_name} has the lower expected cost.")
        changed = row["Different unconstrained decisions"]
        sentences.append(
            f"The unconstrained targeting decision differs for {changed} validation customers."
        )
    return " ".join(sentences)


def verify_saved_files() -> pd.DataFrame:
    """Check the local exports. AutoML columns are checked only when they exist."""
    _ensure_output_dir()
    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    probability_path = OUTPUT_DIR / PROBABILITY_FILE_NAME
    if not manifest_path.exists() or not probability_path.exists():
        raise AutoMLWorkshopError(
            "Nothing is saved yet. Run python python/workshop_automl.py prepare. "
            "That command does not call Google Cloud."
        )
    manifest = _read_json(manifest_path)
    frame = pd.read_csv(probability_path, dtype={ID_COLUMN: str})
    checks = []

    def add(name: str, passed: bool, detail: str, required: bool = True) -> None:
        checks.append({
            "Check": name,
            "Result": "Pass" if passed else "Fail",
            "Detail": detail,
            "Required": "Yes" if required else "No",
        })

    validation_ids = frame[ID_COLUMN].astype(str).tolist()
    add(
        "One row per validation customer",
        frame[ID_COLUMN].is_unique and len(frame) == int(manifest["n_validation"]),
        f"{len(frame)} rows, {frame[ID_COLUMN].nunique()} identifiers.",
    )
    add(
        "Validation customers match the manifest",
        validation_ids == list(manifest["validation_ids"]),
        "The saved order is the validation split order.",
    )
    train_ids = set(_read_ids(ROOT / manifest["training_ids_file"]))
    final_ids = _read_ids(ROOT / manifest["final_test_file"])
    add(
        "Training, validation, and final test do not overlap",
        set(validation_ids).isdisjoint(train_ids)
        and set(validation_ids).isdisjoint(final_ids)
        and set(final_ids).isdisjoint(train_ids),
        f"Training {len(train_ids)}, validation {len(validation_ids)}, final test {len(final_ids)}.",
    )
    add(
        "Final test is the rest of the previous test set",
        set(validation_ids) | set(final_ids) == set(manifest["previous_test_ids"])
        and len(validation_ids) + len(final_ids) == len(manifest["previous_test_ids"]),
        "The previous test partition is unchanged as the union of these two sets.",
    )
    for experiment in manifest["experiments"]:
        training_ids = _read_ids(ROOT / experiment["training_file"])
        predict_ids = _read_ids(ROOT / experiment["predict_file"])
        add(
            f"{experiment['key']}: training file is the previous training set",
            training_ids == list(manifest["training_ids"]),
            f"{len(training_ids)} rows.",
        )
        add(
            f"{experiment['key']}: prediction file is the validation set",
            predict_ids == validation_ids,
            "The final test customers are not in this file.",
        )
        header = pd.read_csv(ROOT / experiment["training_file"], nrows=0).columns.tolist()
        add(
            f"{experiment['key']}: split labels exist and the target is not a feature",
            SPLIT_COLUMN in header
            and TARGET_COLUMN in header
            and set(experiment["features"]).isdisjoint({ID_COLUMN, TARGET_COLUMN, SPLIT_COLUMN}),
            "Features: " + ", ".join(experiment["features"][:4]) + ("..." if len(experiment["features"]) > 4 else ""),
        )
    split_counts = pd.read_csv(ROOT / manifest["experiments"][0]["training_file"], usecols=[SPLIT_COLUMN])
    found = set(split_counts[SPLIT_COLUMN].astype(str))
    add(
        "Platform split uses TRAIN, VALIDATE, and TEST inside the training file",
        found == set(SPLIT_LABELS),
        ", ".join(f"{label} {int((split_counts[SPLIT_COLUMN] == label).sum())}" for label in SPLIT_LABELS),
    )
    for column in (LOCAL_TABULAR, LOCAL_TEXT):
        scores = frame[column].to_numpy(dtype=float)
        add(
            f"{column} is between 0 and 1",
            np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all(),
            f"Minimum {scores.min():.4f}, maximum {scores.max():.4f}.",
        )
    add(
        "Positive class is the saved positive label, coded as 1",
        set(frame[OBSERVED_COLUMN].astype(int)) <= {0, 1}
        and int(frame[OBSERVED_COLUMN].sum()) == int(manifest["n_validation_positive"]),
        f"Positive label {manifest['positive_label']!r}.",
    )
    saved = _saved_test_probabilities()
    aligned = frame.merge(
        saved,
        left_on=ID_COLUMN,
        right_on=SAVED_PROBABILITY_ID,
        how="left",
        validate="one_to_one",
    )
    same_scores = np.allclose(
        aligned[LOCAL_TEXT].to_numpy(dtype=float),
        aligned[SAVED_PROBABILITY_COLUMN].to_numpy(dtype=float),
    )
    same_outcome = np.array_equal(
        aligned[OBSERVED_COLUMN].to_numpy(dtype=int),
        aligned[SAVED_OUTCOME_COLUMN].to_numpy(dtype=int),
    )
    add(
        "Combined logistic probabilities are the previous test scores",
        same_scores and same_outcome and aligned[SAVED_PROBABILITY_COLUMN].notna().all(),
        "Matched on the customer identifier. The combined model was not refitted.",
    )
    automl_columns = [column for column in (AUTOML_TABULAR, AUTOML_TEXT, AUTOML_TEXT_ONLY) if column in frame.columns]
    if not automl_columns:
        add(
            "AutoML probabilities",
            True,
            "Not saved. No cloud job has been run.",
            required=False,
        )
    else:
        for column in automl_columns:
            scores = frame[column].to_numpy(dtype=float)
            add(
                f"{column} is between 0 and 1",
                np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all(),
                f"Minimum {float(np.min(scores)):.4f}, maximum {float(np.max(scores)):.4f}.",
            )
        meta_path = OUTPUT_DIR / "automl_probability_meta.json"
        if meta_path.exists():
            meta = _read_json(meta_path)
            add(
                "AutoML positive class is the workshop positive label",
                meta.get("positive_label") == manifest["positive_label"]
                and meta.get("positive_coded_as") == 1,
                f"Score column pattern {meta.get('score_column', '')}.",
            )
    economics_ok = {"p1", "cost", "loss"}.issubset(frame.columns)
    if economics_ok:
        p1 = frame["p1"].to_numpy(dtype=float)
        cost = frame["cost"].to_numpy(dtype=float)
        loss = frame["loss"].to_numpy(dtype=float)
        economics_ok = (
            np.isfinite(p1).all() and (p1 >= 0).all() and (p1 <= 1).all()
            and np.isfinite(cost).all() and (cost >= 0).all()
            and np.isfinite(loss).all() and (loss >= 0).all()
        )
    add(
        "Campaign cost, loss, and p1 can be read by the targeting functions",
        economics_ok,
        "The targeting call uses these columns and changes only p0.",
    )
    if economics_ok:
        plan, capped = _plans(frame, LOCAL_TEXT, float(BUDGET))
        add(
            "Targeting accepts the validation file",
            len(plan["target"]) == len(frame)
            and set(np.unique(plan["target"])).issubset({0, 1})
            and capped["campaign_spend"] <= float(BUDGET) + 1e-6,
            f"Unconstrained targets {plan['n_targeted']}. Budget spend {capped['campaign_spend']:.2f}.",
        )
    _self_check_score_parser()
    result = pd.DataFrame(checks)
    failed = result[(result["Required"] == "Yes") & (result["Result"] != "Pass")]
    if len(failed):
        names = "; ".join(failed["Check"].tolist())
        raise AutoMLWorkshopError("Saved-file checks failed: " + names)
    print(f"{len(result)} checks passed.")
    return result


def run_cloud(approve_spend: bool = False, force: bool = False) -> int:
    """Train and predict in Google Cloud after an explicit approval."""
    _ensure_output_dir()
    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    if not manifest_path.exists():
        print("Run python python/workshop_automl.py prepare first. No cloud call was made.")
        return 1
    manifest = _read_json(manifest_path)
    if int(manifest["n_training"]) < MINIMUM_ROWS:
        print(
            f"The training file has {manifest['n_training']} rows. "
            f"The current data page requires at least {MINIMUM_ROWS}. No cloud call was made."
        )
        return 1
    resources = _read_resources()
    frame = pd.read_csv(OUTPUT_DIR / PROBABILITY_FILE_NAME, dtype={ID_COLUMN: str})
    plan = _cloud_plan(manifest, resources, frame, force=force)
    if all(item["action"] == "skip" for item in plan):
        print("Saved AutoML probabilities are already on disk. No cloud call was made.")
        return 0
    _print_approval_block(manifest, plan)
    if not _spend_approved(approve_spend):
        print("Set AUTOML_APPROVE_SPEND=yes and pass --approve-spend to continue.")
        print("Approval was not given. No cloud call was made.")
        return 2
    project, region, bucket = _cloud_environment()
    aiplatform, storage, credentials = _import_cloud()
    aiplatform.init(
        project=project,
        location=region,
        staging_bucket=f"gs://{bucket}",
        credentials=credentials,
    )
    reuse = (not force) and _latest_open_run(resources) is not None and any(
        item["action"] in {"wait", "predict"} for item in plan
    )
    if reuse:
        run_record = _latest_open_run(resources)
    else:
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_record = {
            "run_id": run_id,
            "project": project,
            "location": region,
            "bucket": bucket,
            "prefix": f"{GCS_PREFIX}/{run_id}",
            "deleted": False,
            "experiments": {},
        }
        resources["runs"].append(run_record)
        _write_resources(resources)
    print(f"Approval received. Project {project}, region {region}, bucket {bucket}.")
    print("Experiments run one after another. A failure stops the sequence before the next training job.")
    try:
        for item in plan:
            if item["action"] == "skip":
                print(f"{item['label']}: already saved.")
                continue
            _run_one_experiment(
                item, manifest, run_record, resources, frame, aiplatform, storage, credentials
            )
            _write_resources(resources)
    finally:
        _write_cleanup_script(resources)
        _write_resources(resources)
    print("Cloud results are saved locally. The model is not deployed to an endpoint.")
    print("Delete the recorded resources when you are finished:")
    print("  export AUTOML_APPROVE_DELETE=yes")
    print("  python python/workshop_automl.py cleanup --approve-delete")
    return 0


def cleanup_cloud(approve_delete: bool = False) -> int:
    """Delete only the resources recorded by ``run_cloud``."""
    _ensure_output_dir()
    resources = _read_resources()
    pending = [run for run in resources.get("runs", []) if not run.get("deleted")]
    if not pending:
        print("No recorded cloud resources are waiting to be deleted.")
        return 0
    print("This deletes only the resources recorded in cloud_resources.json.")
    for run in pending:
        print(f"Run {run['run_id']} in project {run['project']}, region {run['location']}.")
    if not _delete_approved(approve_delete):
        print("Deletion was not approved. Nothing was deleted.")
        return 2
    aiplatform, storage, credentials = _import_cloud()
    for run in pending:
        aiplatform.init(
            project=run["project"],
            location=run["location"],
            credentials=credentials,
        )
        _delete_run(run, aiplatform, storage, credentials)
        run["deleted"] = True
        _write_resources(resources)
        print(f"Recorded resources for run {run['run_id']} were deleted.")
    print("List the project to confirm that no endpoint is still deployed:")
    print(f"  gcloud ai endpoints list --region={pending[-1]['location']}")
    print(f"  gcloud ai models list --region={pending[-1]['location']}")
    return 0


def _cost_rows_from_manifest() -> list[dict]:
    _ensure_output_dir()
    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        return [
            {"key": item["key"], "label": item["label"], "action": "train"}
            for item in manifest["experiments"]
        ]
    return _planned_experiments()


def _planned_experiments() -> list[dict]:
    rows = [
        {"key": "tabular", "label": "Experiment A, tabular AutoML", "action": "train"},
        {"key": "tabular_text", "label": "Experiment B, tabular AutoML plus text", "action": "train"},
    ]
    if INCLUDE_TEXT_ONLY:
        rows.append({"key": "text", "label": "Experiment C, text only", "action": "train"})
    return rows


def _validate_local_settings() -> None:
    if not 0 < float(HOLDOUT_VALIDATION_FRACTION) < 1:
        raise AutoMLWorkshopError("HOLDOUT_VALIDATION_FRACTION must be between 0 and 1.")
    if int(BUDGET_MILLI_NODE_HOURS) < 1000 or int(BUDGET_MILLI_NODE_HOURS) > 72000:
        raise AutoMLWorkshopError(
            "BUDGET_MILLI_NODE_HOURS must be from 1000 to 72000. 1000 is one node hour."
        )
    if SPLIT_COLUMN in NUMERIC_FEATURES or SPLIT_COLUMN in CATEGORICAL_FEATURES:
        raise AutoMLWorkshopError(f"{SPLIT_COLUMN} is the platform split column. Do not use it as a feature.")
    if TEXT_FEATURE in NUMERIC_FEATURES or TEXT_FEATURE in CATEGORICAL_FEATURES:
        raise AutoMLWorkshopError(f"{TEXT_FEATURE} is reserved for the aggregated communications.")
    reserved = {ID_COLUMN, TARGET_COLUMN, SPLIT_COLUMN, TEXT_FEATURE}
    overlap = reserved & (set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES))
    if overlap:
        raise AutoMLWorkshopError("These columns cannot be predictors: " + ", ".join(sorted(overlap)) + ".")


def _apply_workshop_settings() -> None:
    workshop.ID_COLUMN = ID_COLUMN
    workshop.TARGET_COLUMN = TARGET_COLUMN
    workshop.POSITIVE_LABEL = POSITIVE_LABEL
    workshop.TEXT_COLUMN = TEXT_COLUMN
    workshop.TEXT_ORDER_COLUMN = TEXT_ORDER_COLUMN
    workshop.TEXT_TYPE_COLUMN = TEXT_TYPE_COLUMN
    workshop.NUMERIC_FEATURES = list(NUMERIC_FEATURES)
    workshop.CATEGORICAL_FEATURES = list(CATEGORICAL_FEATURES)
    workshop.RANDOM_STATE = int(RANDOM_STATE)
    workshop.TEST_SIZE = float(TEST_SIZE)
    workshop.N_TEXT_COMPONENTS = int(N_TEXT_COMPONENTS)
    workshop.TEXT_MODE = "RUN"
    workshop.DECISION_THRESHOLD = float(DECISION_THRESHOLD)


def _require_saved_test_alignment(split: dict) -> None:
    saved = _saved_test_probabilities()
    test_ids = split["test"][ID_COLUMN].astype(str).tolist()
    saved_ids = saved[SAVED_PROBABILITY_ID].astype(str).tolist()
    if test_ids != saved_ids:
        raise AutoMLWorkshopError(
            "The current training/test split does not match "
            f"{SAVED_PROBABILITY_FILE}. "
            f"Use the same RANDOM_STATE ({RANDOM_STATE}) and TEST_SIZE ({TEST_SIZE}) "
            "that wrote that file in the earlier workshop. "
            "The customer order in that file is the test order."
        )
    observed = workshop._positive_array(split["test"][TARGET_COLUMN])
    if not np.array_equal(observed, saved[SAVED_OUTCOME_COLUMN].to_numpy(dtype=int)):
        raise AutoMLWorkshopError("The saved test file and the source table disagree on the positive label.")


def _attach_text(table: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    docs = documents.rename(columns={"document": TEXT_FEATURE}).copy()
    docs[ID_COLUMN] = docs[ID_COLUMN].astype(str)
    annotated = table.copy()
    annotated[ID_COLUMN] = annotated[ID_COLUMN].astype(str)
    annotated = annotated.merge(docs, on=ID_COLUMN, how="left", validate="one_to_one")
    if len(annotated) != len(table):
        raise AutoMLWorkshopError("Joining the communications changed the number of customers.")
    annotated[TEXT_FEATURE] = annotated[TEXT_FEATURE].fillna("")
    return annotated


def _assign_groups(split: dict, annotated: pd.DataFrame) -> dict:
    train = split["train"].copy()
    test = split["test"].copy()
    train[ID_COLUMN] = train[ID_COLUMN].astype(str)
    test[ID_COLUMN] = test[ID_COLUMN].astype(str)
    text_map = annotated.set_index(ID_COLUMN)[TEXT_FEATURE]
    train[TEXT_FEATURE] = train[ID_COLUMN].map(text_map)
    test[TEXT_FEATURE] = test[ID_COLUMN].map(text_map)
    if train[TEXT_FEATURE].isna().any() or test[TEXT_FEATURE].isna().any():
        raise AutoMLWorkshopError("A customer lost the aggregated communications during the split.")
    y_test = workshop._positive_array(test[TARGET_COLUMN])
    validation, final_test = train_test_split(
        test,
        test_size=1.0 - float(HOLDOUT_VALIDATION_FRACTION),
        random_state=int(RANDOM_STATE),
        stratify=y_test,
    )
    validation = validation.reset_index(drop=True)
    final_test = final_test.reset_index(drop=True)
    labeled = _label_platform_split(train)
    _require_disjoint(labeled, validation, final_test, test)
    return {
        "train": labeled.reset_index(drop=True),
        "validation": validation,
        "final_test": final_test,
        "previous_test_ids": test[ID_COLUMN].astype(str).tolist(),
    }


def _label_platform_split(train: pd.DataFrame) -> pd.DataFrame:
    y = workshop._positive_array(train[TARGET_COLUMN])
    try:
        rest, platform_test = train_test_split(
            train,
            test_size=float(PLATFORM_TEST_FRACTION),
            random_state=int(RANDOM_STATE),
            stratify=y,
        )
        y_rest = workshop._positive_array(rest[TARGET_COLUMN])
        relative = float(PLATFORM_VALIDATE_FRACTION) / (1.0 - float(PLATFORM_TEST_FRACTION))
        platform_train, platform_validate = train_test_split(
            rest,
            test_size=relative,
            random_state=int(RANDOM_STATE),
            stratify=y_rest,
        )
    except ValueError as exc:
        raise AutoMLWorkshopError(
            "The platform split inside the training set could not be stratified. "
            "Each part needs both target labels."
        ) from exc
    labels = {}
    for frame, label in (
        (platform_train, "TRAIN"),
        (platform_validate, "VALIDATE"),
        (platform_test, "TEST"),
    ):
        for identifier in frame[ID_COLUMN].astype(str):
            labels[identifier] = label
    labeled = train.copy()
    labeled[SPLIT_COLUMN] = labeled[ID_COLUMN].astype(str).map(labels)
    if labeled[SPLIT_COLUMN].isna().any() or set(labeled[SPLIT_COLUMN]) != set(SPLIT_LABELS):
        raise AutoMLWorkshopError("The platform split did not label every training customer.")
    return labeled


def _require_disjoint(train: pd.DataFrame, validation: pd.DataFrame, final_test: pd.DataFrame, previous_test: pd.DataFrame) -> None:
    train_ids = set(train[ID_COLUMN].astype(str))
    validation_ids = set(validation[ID_COLUMN].astype(str))
    final_ids = set(final_test[ID_COLUMN].astype(str))
    previous_ids = set(previous_test[ID_COLUMN].astype(str))
    if validation_ids & train_ids or final_ids & train_ids or validation_ids & final_ids:
        raise AutoMLWorkshopError("A customer appears in more than one of training, validation, and final test.")
    if validation_ids | final_ids != previous_ids:
        raise AutoMLWorkshopError("Validation and final test do not rebuild the previous test set.")


def _experiment_specs() -> list[dict]:
    specs = [
        {
            "key": "tabular",
            "label": "Experiment A, tabular AutoML",
            "probability_column": AUTOML_TABULAR,
            "include_text": False,
            "text_only": False,
        },
        {
            "key": "tabular_text",
            "label": "Experiment B, tabular AutoML plus text",
            "probability_column": AUTOML_TEXT,
            "include_text": True,
            "text_only": False,
        },
    ]
    if INCLUDE_TEXT_ONLY:
        specs.append({
            "key": "text",
            "label": "Experiment C, text only",
            "probability_column": AUTOML_TEXT_ONLY,
            "include_text": True,
            "text_only": True,
        })
    for spec in specs:
        spec["features"] = _feature_columns(spec)
        spec["column_specs"] = _column_specs(spec)
    return specs


def _feature_columns(spec: dict) -> list[str]:
    if spec["text_only"]:
        return [TEXT_FEATURE]
    columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    if spec["include_text"]:
        columns.append(TEXT_FEATURE)
    return columns


def _column_specs(spec: dict) -> dict:
    if spec["text_only"]:
        return {TEXT_FEATURE: "text"}
    specs = {column: "numeric" for column in NUMERIC_FEATURES}
    specs.update({column: "categorical" for column in CATEGORICAL_FEATURES})
    if spec["include_text"]:
        specs[TEXT_FEATURE] = "text"
    return specs


def _write_experiment_files(groups: dict, experiments: list[dict]) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    for spec in experiments:
        columns = spec["features"]
        training = groups["train"][[ID_COLUMN, *columns, TARGET_COLUMN, SPLIT_COLUMN]].copy()
        predict = groups["validation"][[ID_COLUMN, *columns]].copy()
        training_path = OUTPUT_DIR / f"training_{spec['key']}.csv"
        predict_path = OUTPUT_DIR / f"predict_validation_{spec['key']}.csv"
        training.to_csv(training_path, index=False)
        predict.to_csv(predict_path, index=False)
        written[spec["key"]] = {
            "training_file": str(training_path.relative_to(ROOT)),
            "predict_file": str(predict_path.relative_to(ROOT)),
        }
    final_path = OUTPUT_DIR / "final_test_ids.csv"
    groups["final_test"][[ID_COLUMN]].to_csv(final_path, index=False)
    train_ids_path = OUTPUT_DIR / "training_ids.csv"
    groups["train"][[ID_COLUMN]].to_csv(train_ids_path, index=False)
    written["final_test_file"] = str(final_path.relative_to(ROOT))
    written["training_ids_file"] = str(train_ids_path.relative_to(ROOT))
    return written


def _local_probability_frame(split: dict, groups: dict) -> pd.DataFrame:
    tabular = workshop.run_tabular_model(split)
    if list(tabular["test_customer_ids"]) != groups["previous_test_ids"]:
        raise AutoMLWorkshopError("The tabular model scored a different test set from the saved file.")
    tabular_scores = pd.DataFrame({
        ID_COLUMN: [str(value) for value in tabular["test_customer_ids"]],
        LOCAL_TABULAR: np.asarray(tabular["prob_test"], dtype=float),
    })
    saved = _saved_test_probabilities()
    validation = groups["validation"][[ID_COLUMN, TARGET_COLUMN]].copy()
    validation[ID_COLUMN] = validation[ID_COLUMN].astype(str)
    frame = validation.merge(tabular_scores, on=ID_COLUMN, how="left", validate="one_to_one")
    frame = frame.merge(
        saved[[SAVED_PROBABILITY_ID, SAVED_PROBABILITY_COLUMN, SAVED_OUTCOME_COLUMN]],
        left_on=ID_COLUMN,
        right_on=SAVED_PROBABILITY_ID,
        how="left",
        validate="one_to_one",
    )
    if frame[LOCAL_TABULAR].isna().any() or frame[SAVED_PROBABILITY_COLUMN].isna().any():
        raise AutoMLWorkshopError("A validation customer has no local probability.")
    frame[OBSERVED_COLUMN] = workshop._positive_array(frame[TARGET_COLUMN])
    if not np.array_equal(frame[OBSERVED_COLUMN].to_numpy(dtype=int), frame[SAVED_OUTCOME_COLUMN].to_numpy(dtype=int)):
        raise AutoMLWorkshopError("The saved outcome does not match the validation labels.")
    frame = frame.rename(columns={SAVED_PROBABILITY_COLUMN: LOCAL_TEXT})
    economics = _economics_for_ids(frame[ID_COLUMN])
    frame = frame.merge(economics, on=ID_COLUMN, how="left", validate="one_to_one")
    keep = [ID_COLUMN, OBSERVED_COLUMN, LOCAL_TABULAR, LOCAL_TEXT, "p1", "cost", "loss"]
    result = frame[keep].copy()
    _require_unit_interval(result[LOCAL_TABULAR].to_numpy(dtype=float), LOCAL_TABULAR)
    _require_unit_interval(result[LOCAL_TEXT].to_numpy(dtype=float), LOCAL_TEXT)
    return result


def _economics_for_ids(identifiers: pd.Series) -> pd.DataFrame:
    economics = pd.read_csv(_path(ECONOMICS_FILE), dtype={ECONOMICS_ID_COLUMN: str})
    required = [ECONOMICS_ID_COLUMN, ECONOMICS_P1_COLUMN, ECONOMICS_COST_COLUMN, ECONOMICS_LOSS_COLUMN]
    missing = [column for column in required if column not in economics.columns]
    if missing:
        raise AutoMLWorkshopError(
            f"{ECONOMICS_FILE} is missing {missing}. "
            "The targeting comparison needs a campaign probability, a contact cost, and a loss."
        )
    if economics[ECONOMICS_ID_COLUMN].duplicated().any():
        raise AutoMLWorkshopError(f"{ECONOMICS_FILE} repeats a customer identifier.")
    renamed = economics.rename(columns={
        ECONOMICS_ID_COLUMN: ID_COLUMN,
        ECONOMICS_P1_COLUMN: "p1",
        ECONOMICS_COST_COLUMN: "cost",
        ECONOMICS_LOSS_COLUMN: "loss",
    })
    wanted = pd.DataFrame({ID_COLUMN: identifiers.astype(str).tolist()})
    merged = wanted.merge(renamed[[ID_COLUMN, "p1", "cost", "loss"]], on=ID_COLUMN, how="left", validate="one_to_one")
    if merged[["p1", "cost", "loss"]].isna().any().any():
        raise AutoMLWorkshopError("A validation customer is missing from the campaign file.")
    return merged


def _manifest(groups: dict, experiments: list[dict], written: dict, probabilities: pd.DataFrame) -> dict:
    previous_test = pd.concat(
        [groups["validation"][TARGET_COLUMN], groups["final_test"][TARGET_COLUMN]],
        ignore_index=True,
    )
    summary_frames = [
        ("Previous training", groups["train"][TARGET_COLUMN]),
        ("Previous test", previous_test),
        ("Validation, used for comparison", groups["validation"][TARGET_COLUMN]),
        ("Final test, not scored", groups["final_test"][TARGET_COLUMN]),
    ]
    summary = []
    for label, labels in summary_frames:
        y = workshop._positive_array(labels)
        summary.append({
            "Group": label,
            "Rows": int(len(labels)),
            "Positive": int(y.sum()),
            "Positive rate": float(y.mean()),
        })
    split_counts = groups["train"][SPLIT_COLUMN].value_counts()
    for label in SPLIT_LABELS:
        mask = groups["train"][SPLIT_COLUMN] == label
        y = workshop._positive_array(groups["train"].loc[mask, TARGET_COLUMN])
        summary.append({
            "Group": f"Platform {label} inside previous training",
            "Rows": int(mask.sum()),
            "Positive": int(y.sum()),
            "Positive rate": float(y.mean()),
        })
    stored_experiments = []
    for spec in experiments:
        paths = written[spec["key"]]
        stored_experiments.append({
            "key": spec["key"],
            "label": spec["label"],
            "probability_column": spec["probability_column"],
            "features": spec["features"],
            "column_specs": spec["column_specs"],
            "training_file": paths["training_file"],
            "predict_file": paths["predict_file"],
            "include_text": spec["include_text"],
            "text_only": spec["text_only"],
        })
    return {
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "positive_label": POSITIVE_LABEL,
        "positive_coded_as": 1,
        "text_feature": TEXT_FEATURE,
        "split_column": SPLIT_COLUMN,
        "split_labels": list(SPLIT_LABELS),
        "optimization_objective": OPTIMIZATION_OBJECTIVE,
        "budget_milli_node_hours": int(BUDGET_MILLI_NODE_HOURS),
        "include_text_only": bool(INCLUDE_TEXT_ONLY),
        "random_state": int(RANDOM_STATE),
        "test_size": float(TEST_SIZE),
        "holdout_validation_fraction": float(HOLDOUT_VALIDATION_FRACTION),
        "n_training": int(len(groups["train"])),
        "n_validation": int(len(groups["validation"])),
        "n_final_test": int(len(groups["final_test"])),
        "n_validation_positive": int(probabilities[OBSERVED_COLUMN].sum()),
        "n_empty_text_training": int((groups["train"][TEXT_FEATURE].astype(str).str.strip() == "").sum()),
        "training_ids": groups["train"][ID_COLUMN].astype(str).tolist(),
        "validation_ids": groups["validation"][ID_COLUMN].astype(str).tolist(),
        "previous_test_ids": list(groups["previous_test_ids"]),
        "training_ids_file": written["training_ids_file"],
        "final_test_file": written["final_test_file"],
        "probability_file": PROBABILITY_FILE_NAME,
        "split_summary": summary,
        "platform_label_counts": {label: int(split_counts.get(label, 0)) for label in SPLIT_LABELS},
        "experiments": stored_experiments,
        "text_aggregation": (
            f"Messages for one {ID_COLUMN} are ordered by {TEXT_ORDER_COLUMN} "
            "and joined with a blank line. The same rule builds the local text features."
        ),
    }


def _saved_test_probabilities() -> pd.DataFrame:
    path = _path(SAVED_PROBABILITY_FILE)
    if not path.exists():
        raise AutoMLWorkshopError(
            f"{SAVED_PROBABILITY_FILE} was not found. "
            "The previous workshop writes it with python/campaign_scenario.py."
        )
    frame = pd.read_csv(path, dtype={SAVED_PROBABILITY_ID: str})
    required = [SAVED_PROBABILITY_ID, SAVED_PROBABILITY_COLUMN, SAVED_OUTCOME_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AutoMLWorkshopError(f"{SAVED_PROBABILITY_FILE} is missing {missing}.")
    if frame[SAVED_PROBABILITY_ID].duplicated().any():
        raise AutoMLWorkshopError(f"{SAVED_PROBABILITY_FILE} repeats a customer identifier.")
    return frame


def _require_probability_frame(frame: pd.DataFrame, manifest: dict) -> None:
    required = [ID_COLUMN, OBSERVED_COLUMN, LOCAL_TABULAR, LOCAL_TEXT, "p1", "cost", "loss"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AutoMLWorkshopError(f"{PROBABILITY_FILE_NAME} is missing {missing}.")
    if frame[ID_COLUMN].astype(str).tolist() != list(manifest["validation_ids"]):
        raise AutoMLWorkshopError(
            "The probability file is not the saved validation set. "
            'Set CLOUD_MODE to "SAVE" and rebuild the local files.'
        )


def _frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return load_workshop()["probabilities"]
    return frame


def _model_series(frame: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    catalog = [
        (LOCAL_TABULAR, "Logistic regression, tabular", NAVY, "-"),
        (LOCAL_TEXT, "Logistic regression, tabular and text", CLAY, "-"),
        (AUTOML_TABULAR, "AutoML, tabular", NAVY, "--"),
        (AUTOML_TEXT, "AutoML, tabular and text", CLAY, "--"),
        (AUTOML_TEXT_ONLY, "AutoML, text only", GREEN, "--"),
    ]
    return [item for item in catalog if item[0] in frame.columns]


def _column_for_label(frame: pd.DataFrame, label: str) -> str:
    for column, name, _color, _style in _model_series(frame):
        if name == label:
            return column
    raise AutoMLWorkshopError(f"The probability frame has no model {label!r}.")


def _plans(frame: pd.DataFrame, column: str, budget: float) -> tuple[dict, dict]:
    identifiers = frame[ID_COLUMN].astype(str).tolist()
    p0 = frame[column].to_numpy(dtype=float)
    p1 = frame["p1"].to_numpy(dtype=float)
    cost = frame["cost"].to_numpy(dtype=float)
    loss = frame["loss"].to_numpy(dtype=float)
    open_plan = targeting.allocate(p0, p1, cost, loss, ids=identifiers)
    capped = targeting.allocate_under_budget(
        p0, p1, cost, loss, float(budget), ids=identifiers, mode="RUN"
    )
    return open_plan, capped


def _better_name(summary: pd.DataFrame, left: str, right: str, column: str, higher: bool) -> str:
    left_value = float(summary.loc[left, column])
    right_value = float(summary.loc[right, column])
    if np.isclose(left_value, right_value):
        return "Tie"
    if higher:
        return left if left_value > right_value else right
    return left if left_value < right_value else right


def _require_unit_interval(scores: np.ndarray, name: str) -> None:
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise AutoMLWorkshopError(f"{name} has a value outside 0 to 1.")


def _self_check_score_parser() -> None:
    sample = pd.DataFrame({
        ID_COLUMN: ["a", "b"],
        f"{TARGET_COLUMN}_{POSITIVE_LABEL}_score": [0.25, 0.75],
        f"{TARGET_COLUMN}_No_score": [0.75, 0.25],
    })
    scores, column = _extract_positive_score(sample, TARGET_COLUMN, POSITIVE_LABEL)
    if column != f"{TARGET_COLUMN}_{POSITIVE_LABEL}_score" or not np.allclose(scores, [0.25, 0.75]):
        raise AutoMLWorkshopError("The positive-class score parser failed its local check.")
    plural = pd.DataFrame({
        ID_COLUMN: ["a", "b"],
        f"{TARGET_COLUMN}_{POSITIVE_LABEL}_scores": [0.2, 0.8],
        f"{TARGET_COLUMN}_No_scores": [0.8, 0.2],
    })
    scores, column = _extract_positive_score(plural, TARGET_COLUMN, POSITIVE_LABEL)
    if column != f"{TARGET_COLUMN}_{POSITIVE_LABEL}_scores" or not np.allclose(scores, [0.2, 0.8]):
        raise AutoMLWorkshopError("The positive-class score parser failed its local check.")


def _extract_positive_score(frame: pd.DataFrame, target: str, positive: str) -> tuple[pd.Series, str]:
    singular = f"{target}_{positive}_score"
    plural = f"{target}_{positive}_scores"
    lookup = {str(column).lower(): column for column in frame.columns}
    column = lookup.get(singular.lower()) or lookup.get(plural.lower())
    if column is None:
        raise AutoMLWorkshopError(
            f"The prediction file has no column {singular} or {plural}. "
            f"Found columns: {list(frame.columns)}. "
            f"The positive class is {positive!r}, coded as 1 in the local files."
        )
    scores = pd.to_numeric(frame[column], errors="coerce")
    if scores.isna().any():
        raise AutoMLWorkshopError(f"{column} contains a value that is not a probability.")
    companions = [
        name for name in frame.columns
        if str(name).lower().startswith(f"{target}_".lower())
        and (str(name).lower().endswith("_scores") or str(name).lower().endswith("_score"))
        and str(name).lower() != str(column).lower()
    ]
    if companions:
        other = pd.to_numeric(frame[companions[0]], errors="coerce")
        if other.notna().all():
            gap = np.max(np.abs(scores.to_numpy(dtype=float) + other.to_numpy(dtype=float) - 1.0))
            if gap > 0.02:
                raise AutoMLWorkshopError(
                    f"{column} and {companions[0]} do not add to 1. Largest gap {gap:.4f}."
                )
    return scores, str(column)


def _cloud_plan(manifest: dict, resources: dict, frame: pd.DataFrame, force: bool) -> list[dict]:
    latest = _latest_open_run(resources)
    plan = []
    for experiment in manifest["experiments"]:
        column = experiment["probability_column"]
        recorded = (latest or {}).get("experiments", {}).get(experiment["key"], {})
        if column in frame.columns and frame[column].notna().all() and not force:
            action = "skip"
        elif recorded.get("model") and not force:
            action = "predict"
        elif recorded.get("training_pipeline") and not force:
            action = "wait"
        else:
            action = "train"
        plan.append({**experiment, "action": action, "recorded": recorded})
    return plan


def _print_approval_block(manifest: dict, plan: list[dict]) -> None:
    hours = int(manifest["budget_milli_node_hours"]) / 1000.0
    print("Billable Google Cloud work is requested.")
    print(f"Training objective: {manifest['optimization_objective']}.")
    print(f"Training cap: {hours:g} node hour per new model. Early stopping can stop sooner.")
    print("You are charged if you cancel a training job. Another training failure is not billed.")
    print("Batch prediction is billed for node time. Replica count is capped at 1.")
    print("No endpoint is created. An endpoint would be billed until you undeploy it.")
    print("Rates are on https://cloud.google.com/vertex-ai/pricing for the region you select.")
    print(explain_costs(plan))


def _spend_approved(flag: bool) -> bool:
    approved = os.environ.get("AUTOML_APPROVE_SPEND", "").strip().lower() in {"yes", "true", "1"}
    return bool(flag) and approved


def _delete_approved(flag: bool) -> bool:
    approved = os.environ.get("AUTOML_APPROVE_DELETE", "").strip().lower() in {"yes", "true", "1"}
    return bool(flag) and approved


def _cloud_environment() -> tuple[str, str, str]:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    region = os.environ.get("GOOGLE_CLOUD_REGION", REGION).strip() or REGION
    bucket = os.environ.get("GOOGLE_CLOUD_BUCKET", "").strip()
    bucket = bucket.removeprefix("gs://").strip("/")
    missing = [
        name for name, value in (
            ("GOOGLE_CLOUD_PROJECT", project),
            ("GOOGLE_CLOUD_BUCKET", bucket),
        ) if not value
    ]
    if missing:
        raise AutoMLWorkshopError(
            "Set " + ", ".join(missing) + " before the paid command. "
            "Also set GOOGLE_CLOUD_REGION if you are not using us-central1. "
            "The bucket must be in that region. No cloud call was made."
        )
    return project, region, bucket


def _import_cloud():
    try:
        import google.auth
        from google.cloud import aiplatform
        from google.cloud import storage
    except ImportError as exc:
        raise AutoMLWorkshopError(
            "Install the cloud libraries before a paid run: "
            "pip install google-cloud-aiplatform google-cloud-storage. "
            "They are not required to render the lesson."
        ) from exc
    try:
        credentials, _detected = google.auth.default()
    except Exception as exc:
        raise AutoMLWorkshopError(
            "Application Default Credentials were not found. "
            "Run gcloud auth application-default login. "
            "Do not put a service-account key in this repository."
        ) from exc
    return aiplatform, storage, credentials


def _run_one_experiment(item, manifest, run_record, resources, frame, aiplatform, storage, credentials) -> None:
    key = item["key"]
    recorded = {} if item["action"] == "train" else dict(item.get("recorded") or {})
    if item["action"] == "train":
        gcs_training = _upload_file(
            storage, credentials, run_record, ROOT / item["training_file"], f"training_{key}.csv"
        )
        dataset = aiplatform.TabularDataset.create(
            display_name=f"{GCS_PREFIX}-{key}"[:120],
            gcs_source=[gcs_training],
        )
        recorded["dataset"] = dataset.resource_name
        recorded["gcs_training"] = gcs_training
        run_record["experiments"][key] = recorded
        _write_resources(resources)
        job = aiplatform.AutoMLTabularTrainingJob(
            display_name=f"{GCS_PREFIX}-{key}"[:120],
            optimization_prediction_type="classification",
            optimization_objective=manifest["optimization_objective"],
            column_specs=item["column_specs"],
        )
        model_or_future = job.run(
            dataset=dataset,
            target_column=manifest["target_column"],
            model_display_name=f"{GCS_PREFIX}-{key}-model"[:120],
            predefined_split_column_name=manifest["split_column"],
            budget_milli_node_hours=int(manifest["budget_milli_node_hours"]),
            disable_early_stopping=False,
            sync=False,
        )
        job.wait_for_resource_creation()
        recorded["training_pipeline"] = job.resource_name
        run_record["experiments"][key] = recorded
        _write_resources(resources)
        print(f"{item['label']}: training pipeline {job.resource_name}")
        model = _resolve_future(model_or_future)
        if model is None:
            model = job.get_model(sync=True)
        recorded["model"] = model.resource_name
        run_record["experiments"][key] = recorded
        _write_resources(resources)
    elif item["action"] == "wait":
        job = aiplatform.AutoMLTabularTrainingJob.get(recorded["training_pipeline"])
        model = job.get_model(sync=True)
        recorded["model"] = model.resource_name
        run_record["experiments"][key] = recorded
        _write_resources(resources)
    else:
        model = aiplatform.Model(recorded["model"])
    gcs_predict = _upload_file(
        storage, credentials, run_record, ROOT / item["predict_file"], f"predict_{key}.csv"
    )
    destination = f"gs://{run_record['bucket']}/{run_record['prefix']}/{key}/predictions/"
    recorded["gcs_predictions"] = destination
    run_record["experiments"][key] = recorded
    _write_resources(resources)
    batch_or_future = model.batch_predict(
        job_display_name=f"{GCS_PREFIX}-{key}-batch"[:120],
        gcs_source=[gcs_predict],
        instances_format="csv",
        gcs_destination_prefix=destination,
        predictions_format="csv",
        starting_replica_count=1,
        max_replica_count=1,
        sync=False,
    )
    if hasattr(batch_or_future, "wait_for_resource_creation"):
        batch_or_future.wait_for_resource_creation()
    batch = _resolve_future(batch_or_future)
    recorded["batch_prediction_job"] = batch.resource_name
    recorded["gcs_predictions"] = destination
    run_record["experiments"][key] = recorded
    _write_resources(resources)
    print(f"{item['label']}: batch prediction {batch.resource_name}")
    output = _download_prediction_csvs(storage, credentials, run_record, batch, key)
    scores, score_column = _extract_positive_score(
        output, manifest["target_column"], manifest["positive_label"]
    )
    output = output.copy()
    output[item["probability_column"]] = scores
    _merge_automl_column(frame, output, item, manifest, score_column)
    print(f"{item['label']}: saved {item['probability_column']} for {len(frame)} validation customers.")


def _resolve_future(value):
    if hasattr(value, "done") and hasattr(value, "result") and callable(value.result):
        return value.result()
    return value


def _upload_file(storage, credentials, run_record, path: Path, name: str) -> str:
    client = storage.Client(project=run_record["project"], credentials=credentials)
    bucket = client.bucket(run_record["bucket"])
    if not bucket.exists():
        raise AutoMLWorkshopError(
            f"gs://{run_record['bucket']} was not found. "
            f"Create it in {run_record['location']} before training."
        )
    blob_name = f"{run_record['prefix']}/{name}"
    bucket.blob(blob_name).upload_from_filename(path)
    return f"gs://{run_record['bucket']}/{blob_name}"


def _is_prediction_result_csv(name: str) -> bool:
    """Keep the scored shards. Leave error tables and temporary files unread."""
    if "beam-temp" in name:
        return False
    base = Path(name).name
    return base == "prediction.results.csv" or (
        base.startswith("prediction.results-") and base.endswith(".csv")
    )


def _download_result_csvs(storage, credentials, run_record, prefix: str, experiment: str) -> pd.DataFrame:
    client = storage.Client(project=run_record["project"], credentials=credentials)
    bucket_name, blob_prefix = _split_gcs_uri(prefix)
    frames = []
    for blob in client.list_blobs(bucket_name, prefix=blob_prefix):
        if not _is_prediction_result_csv(blob.name):
            continue
        local = OUTPUT_DIR / "cloud_download" / run_record["run_id"] / experiment / Path(blob.name).name
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local)
        frames.append(pd.read_csv(local, dtype={ID_COLUMN: str}))
    if not frames:
        raise AutoMLWorkshopError(
            f"No prediction.results file was found at {prefix}. "
            "The batch prediction has not written its scores yet."
        )
    return pd.concat(frames, ignore_index=True)


def _recorded_prediction_prefix(experiment: str) -> tuple[dict, str]:
    resources = _read_resources()
    for run in reversed(resources.get("runs", [])):
        if run.get("deleted"):
            continue
        recorded = (run.get("experiments") or {}).get(experiment) or {}
        prefix = recorded.get("gcs_predictions")
        if prefix:
            return run, prefix
    raise AutoMLWorkshopError(
        f"cloud_resources.json has no finished batch prediction for {experiment!r}. "
        "That column can be copied after the batch job writes its files."
    )


def _download_prediction_csvs(storage, credentials, run_record, batch, experiment: str) -> pd.DataFrame:
    output_prefix = _batch_output_prefix(batch)
    return _download_result_csvs(storage, credentials, run_record, output_prefix, experiment)


def _batch_output_prefix(batch) -> str:
    info = getattr(batch, "output_info", None)
    if info is not None and getattr(info, "gcs_output_directory", None):
        return info.gcs_output_directory
    resource = getattr(batch, "gca_resource", None)
    if resource is not None and resource.output_info.gcs_output_directory:
        return resource.output_info.gcs_output_directory
    raise AutoMLWorkshopError("The batch prediction finished without a Cloud Storage output directory.")


def _merge_automl_column(frame: pd.DataFrame, output: pd.DataFrame, item: dict, manifest: dict, score_column: str) -> None:
    id_column = _match_id_column(output.columns, manifest["id_column"])
    slim = output[[id_column, item["probability_column"]]].copy()
    slim[id_column] = slim[id_column].astype(str)
    if slim[id_column].duplicated().any():
        raise AutoMLWorkshopError("The batch prediction repeated a customer identifier.")
    merged = frame[[ID_COLUMN]].merge(
        slim,
        left_on=ID_COLUMN,
        right_on=id_column,
        how="left",
        validate="one_to_one",
    )
    if merged[item["probability_column"]].isna().any():
        raise AutoMLWorkshopError("The batch prediction missed a validation customer.")
    if len(merged) != len(frame):
        raise AutoMLWorkshopError("The batch prediction changed the number of validation customers.")
    scores = merged[item["probability_column"]].to_numpy(dtype=float)
    _require_unit_interval(scores, item["probability_column"])
    frame[item["probability_column"]] = scores
    path = OUTPUT_DIR / PROBABILITY_FILE_NAME
    frame.to_csv(path, index=False)
    _write_json(OUTPUT_DIR / "automl_probability_meta.json", {
        "positive_label": manifest["positive_label"],
        "positive_coded_as": 1,
        "target_column": manifest["target_column"],
        "score_column": score_column,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    })
    global _LOADED
    _LOADED = None


def _match_id_column(columns, expected: str) -> str:
    lookup = {str(column).lower(): column for column in columns}
    if expected.lower() not in lookup:
        raise AutoMLWorkshopError(
            f"The prediction file has no identifier column {expected}. Columns: {list(columns)}."
        )
    return lookup[expected.lower()]


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise AutoMLWorkshopError(f"Expected a gs:// path, received {uri}.")
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, prefix


def _delete_run(run: dict, aiplatform, storage, credentials) -> None:
    for key, experiment in run.get("experiments", {}).items():
        _delete_named(experiment.get("batch_prediction_job"), lambda name: _delete_batch(aiplatform, name), f"{key} batch job")
        _delete_named(experiment.get("endpoint"), lambda name: _delete_endpoint(aiplatform, name), f"{key} endpoint")
        _delete_named(experiment.get("model"), lambda name: aiplatform.Model(name).delete(), f"{key} model")
        _delete_named(
            experiment.get("training_pipeline"),
            lambda name: _delete_training(aiplatform, name),
            f"{key} training pipeline",
        )
        _delete_named(experiment.get("dataset"), lambda name: aiplatform.TabularDataset(name).delete(), f"{key} dataset")
    _delete_recorded_prefix(storage, credentials, run)


def _delete_named(resource_name: str | None, deleter, label: str) -> None:
    if not resource_name:
        return
    try:
        deleter(resource_name)
        print(f"Deleted {label}: {resource_name}")
    except Exception as exc:
        print(f"Could not delete {label}: {resource_name}. {exc}")


def _delete_batch(aiplatform, name: str) -> None:
    job = aiplatform.BatchPredictionJob(name)
    try:
        job.cancel()
    except Exception:
        pass
    job.delete()


def _delete_endpoint(aiplatform, name: str) -> None:
    endpoint = aiplatform.Endpoint(name)
    endpoint.undeploy_all()
    endpoint.delete()


def _delete_training(aiplatform, name: str) -> None:
    job = aiplatform.AutoMLTabularTrainingJob.get(name)
    try:
        job.cancel()
    except Exception:
        pass
    job.delete()


def _delete_recorded_prefix(storage, credentials, run: dict) -> None:
    prefix = run.get("prefix", "")
    run_id = run.get("run_id", "")
    if not prefix or not run_id or run_id not in prefix or not prefix.startswith(GCS_PREFIX):
        raise AutoMLWorkshopError("Refusing to delete a Cloud Storage prefix that is not one recorded run.")
    client = storage.Client(project=run["project"], credentials=credentials)
    blobs = list(client.list_blobs(run["bucket"], prefix=prefix))
    for blob in blobs:
        blob.delete()
    print(f"Deleted {len(blobs)} Cloud Storage objects under gs://{run['bucket']}/{prefix}")


def _write_cleanup_script(resources: dict) -> None:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "# Generated by python/workshop_automl.py. Deletes only recorded resources.",
    ]
    for run in resources.get("runs", []):
        if run.get("deleted"):
            continue
        region = run["location"]
        lines.append(f"# Run {run['run_id']}")
        for experiment in run.get("experiments", {}).values():
            for key, command in (
                ("endpoint", "endpoints"),
                ("model", "models"),
                ("dataset", "datasets"),
            ):
                resource_name = experiment.get(key)
                if not resource_name:
                    continue
                short = resource_name.rstrip("/").split("/")[-1]
                if key == "endpoint":
                    lines.append(
                        f"# Undeploy every model on this endpoint, then delete it.\n"
                        f"gcloud ai endpoints describe {short} --region={region}\n"
                        f"# gcloud ai endpoints undeploy-model {short} --region={region} "
                        f"--deployed-model-id=DEPLOYED_MODEL_ID --quiet\n"
                        f"gcloud ai endpoints delete {short} --region={region} --quiet || true"
                    )
                else:
                    lines.append(f"gcloud ai {command} delete {short} --region={region} --quiet || true")
            pipeline = experiment.get("training_pipeline")
            if pipeline:
                short = pipeline.rstrip("/").split("/")[-1]
                lines.append(f"gcloud ai training-pipelines cancel {short} --region={region} --quiet || true")
            batch = experiment.get("batch_prediction_job")
            if batch:
                short = batch.rstrip("/").split("/")[-1]
                lines.append(f"gcloud ai batch-prediction-jobs cancel {short} --region={region} --quiet || true")
        lines.append(
            f"gcloud storage rm --recursive \"gs://{run['bucket']}/{run['prefix']}/**\" || true"
        )
    lines.append(f"gcloud ai endpoints list --region={resources['runs'][-1]['location']}")
    path = OUTPUT_DIR / CLEANUP_SCRIPT_NAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}.")


def _latest_open_run(resources: dict) -> dict | None:
    open_runs = [run for run in resources.get("runs", []) if not run.get("deleted")]
    return open_runs[-1] if open_runs else None


def _read_resources() -> dict:
    path = OUTPUT_DIR / RESOURCES_NAME
    if not path.exists():
        return {"runs": []}
    return _read_json(path)


def _write_resources(resources: dict) -> None:
    _write_json(OUTPUT_DIR / RESOURCES_NAME, resources)


def _read_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, usecols=[ID_COLUMN], dtype={ID_COLUMN: str})
    return frame[ID_COLUMN].astype(str).tolist()


def _path(value) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or run the AutoML comparison.")
    parser.add_argument("command", choices=["prepare", "run", "cleanup", "verify"])
    parser.add_argument("--approve-spend", action="store_true")
    parser.add_argument("--approve-delete", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        use_course_example()
        if args.command == "prepare":
            prepare()
            return 0
        if args.command == "verify":
            verify_saved_files()
            return 0
        if args.command == "run":
            return run_cloud(approve_spend=args.approve_spend, force=args.force)
        return cleanup_cloud(approve_delete=args.approve_delete)
    except AutoMLWorkshopError as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

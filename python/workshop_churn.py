"""Participant-facing helpers for a tabular-plus-text workshop.

The Quarto document calls these functions. Fitting, text processing and
cache handling stay here so the workshop cells can stay short.

The text step uses TF-IDF and truncated SVD, and it is fitted on the
training rows only. File paths, the identifier, the target, the positive
label and the feature lists are set by the document. This file does not
name a particular dataset.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

# The Quarto document assigns these in one cell.
RANDOM_STATE = 42
TEST_SIZE = 0.25
N_TEXT_COMPONENTS = 64
TEXT_MODE = "RUN"  # RUN, SAVE or LOAD.
DECISION_THRESHOLD = 0.5

ID_COLUMN = "id"
TARGET_COLUMN = "target"
POSITIVE_LABEL = "1"
TEXT_COLUMN = "text"
TEXT_ORDER_COLUMN = "text_index"
TEXT_TYPE_COLUMN = "text_type"
TABLE_TEXT_COLUMN = ""
NUMERIC_FEATURES: list[str] = []
CATEGORICAL_FEATURES: list[str] = []
TABULAR_FILE: Path | None = None
TEXT_FILE: Path | None = None

CACHE_DIR = ROOT / "data" / "processed" / "text_model_cache"
CACHE_VERSION = 2

TFIDF_SETTINGS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "stop_words": "english",
}

MODEL_COLORS = {
    "Tabular data only": "#1f4e79",
    "Tabular data and text": "#b85c38",
    "Text only": "#2f6f4e",
    "Tabular data and row text": "#6b5b95",
    "Row text only": "#8c6d4f",
    "Always predict negative": "#7a7a7a",
}

ILLUSTRATION_NOTES = (
    ("Slow service A", "fiber slow fiber"),
    ("Slow service B", "slow connection outage"),
    ("Billing", "monthly bill payment"),
)


class WorkshopError(Exception):
    """A problem the participant can correct from the workshop instructions."""


def active_settings() -> pd.DataFrame:
    """Show the settings that control the split, the text step and the cutoff."""
    _validate_settings()
    rows = [
        ("RANDOM_STATE", RANDOM_STATE),
        ("TEST_SIZE", TEST_SIZE),
        ("N_TEXT_COMPONENTS", int(N_TEXT_COMPONENTS)),
        ("TEXT_MODE", _text_mode()),
        ("DECISION_THRESHOLD", DECISION_THRESHOLD),
        ("ID_COLUMN", ID_COLUMN),
        ("TARGET_COLUMN", TARGET_COLUMN),
        ("POSITIVE_LABEL", POSITIVE_LABEL),
        ("TEXT_COLUMN", TEXT_COLUMN),
        ("TEXT_ORDER_COLUMN", TEXT_ORDER_COLUMN),
        ("TEXT_TYPE_COLUMN", TEXT_TYPE_COLUMN or ""),
        ("NUMERIC_FEATURES", ", ".join(NUMERIC_FEATURES)),
        ("CATEGORICAL_FEATURES", ", ".join(CATEGORICAL_FEATURES)),
    ]
    return pd.DataFrame(rows, columns=["Setting", "Value"])


def load_datasets(tabular_file, text_file) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one row-per-entity table and one row-per-text file.

    Blank values in the numeric predictors are replaced with zero.
    """
    global TABULAR_FILE, TEXT_FILE
    _require_roles()
    tabular_path = _require_file(Path(tabular_file), "tabular file")
    text_path = _require_file(Path(text_file), "text file")
    TABULAR_FILE = tabular_path
    TEXT_FILE = text_path

    table = pd.read_csv(tabular_path, dtype={ID_COLUMN: str})
    messages = pd.read_csv(text_path, dtype={ID_COLUMN: str})
    text_columns = [ID_COLUMN, TEXT_COLUMN, TEXT_ORDER_COLUMN]
    if TEXT_TYPE_COLUMN:
        text_columns.append(TEXT_TYPE_COLUMN)
    _require_columns(
        table,
        [ID_COLUMN, TARGET_COLUMN, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES],
        tabular_path,
    )
    _require_columns(messages, text_columns, text_path)

    if table[ID_COLUMN].duplicated().any():
        raise WorkshopError(
            f"The tabular file has duplicate {ID_COLUMN} values. "
            "The workshop expects one row per entity."
        )
    if messages.duplicated([ID_COLUMN, TEXT_ORDER_COLUMN]).any():
        raise WorkshopError(
            f"The text file repeats the same {TEXT_ORDER_COLUMN} for one {ID_COLUMN}. "
            "The order of that entity's texts would be ambiguous."
        )

    table = table.copy()
    filled = []
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(table[column], errors="coerce")
        n_blank = int(values.isna().sum())
        table[column] = values.fillna(0.0)
        if n_blank:
            filled.append(f"{column}: {n_blank}")
    _positive_array(table[TARGET_COLUMN])

    print(f"Tabular file: {_display_path(tabular_path)}")
    print(f"Text file: {_display_path(text_path)}")
    print("Blank numeric values replaced with zero: " + (", ".join(filled) if filled else "0"))
    return table, messages


def data_overview(table: pd.DataFrame, messages: pd.DataFrame) -> pd.DataFrame:
    """Compare the shape of the two files."""
    table_ids = set(table[ID_COLUMN].astype(str))
    message_ids = set(messages[ID_COLUMN].astype(str))
    per_id = messages.groupby(ID_COLUMN).size()
    rows = [
        ("Rows", len(table), len(messages)),
        ("Columns", table.shape[1], messages.shape[1]),
        (f"Unique {ID_COLUMN}", len(table_ids), len(message_ids)),
        ("Rows per identifier (minimum)", 1, int(per_id.min()) if len(per_id) else 0),
        ("Rows per identifier (median)", 1, float(per_id.median()) if len(per_id) else 0),
        ("Rows per identifier (maximum)", 1, int(per_id.max()) if len(per_id) else 0),
        ("Identifiers in both files", len(table_ids & message_ids), len(table_ids & message_ids)),
        ("Identifiers only in this file", len(table_ids - message_ids), len(message_ids - table_ids)),
    ]
    frame = pd.DataFrame(rows, columns=["Item", "Tabular file", "Text file"])
    for column in ("Tabular file", "Text file"):
        frame[column] = frame[column].map(
            lambda value: f"{value:g}" if isinstance(value, (int, float, np.integer, np.floating)) else value
        )
    return frame


def target_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Count rows by the target label."""
    labels = table[TARGET_COLUMN].astype(str).str.strip()
    counts = labels.value_counts()
    frame = pd.DataFrame({
        TARGET_COLUMN: counts.index.astype(str),
        "Rows": counts.to_numpy(),
    })
    frame["Share"] = frame["Rows"] / frame["Rows"].sum()
    positive = str(POSITIVE_LABEL).strip()
    frame["_order"] = frame[TARGET_COLUMN].map(lambda value: 1 if value == positive else 0)
    return frame.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def text_type_counts(messages: pd.DataFrame) -> pd.DataFrame:
    """Count texts by the recorded text type."""
    if not TEXT_TYPE_COLUMN or TEXT_TYPE_COLUMN not in messages.columns:
        raise WorkshopError("Set TEXT_TYPE_COLUMN to a column of the text file before counting text types.")
    counts = messages[TEXT_TYPE_COLUMN].value_counts()
    frame = pd.DataFrame({
        "Text type": counts.index.astype(str),
        "Messages": counts.to_numpy(),
    })
    frame["Share of messages"] = frame["Messages"] / frame["Messages"].sum()
    return frame.reset_index(drop=True)


def timing_summary(messages: pd.DataFrame) -> pd.DataFrame:
    """Show whether the text file carries a date."""
    date_columns = [
        column for column in messages.columns
        if "date" in column.lower() or "time" in column.lower()
    ]
    order_present = TEXT_ORDER_COLUMN in messages.columns
    return pd.DataFrame([{
        "Date or time columns": ", ".join(date_columns) if date_columns else "None",
        "Order column": TEXT_ORDER_COLUMN if order_present else "None",
        "Columns": ", ".join(messages.columns.astype(str)),
    }])


def prepare_train_test(table: pd.DataFrame) -> dict:
    """Keep one entity-level split for every model in the workshop.

    The split is stratified on the target, so the training group and the test
    group have similar positive rates. The test rows are not used to fit a model.
    """
    _validate_settings()
    _require_roles()
    if table[ID_COLUMN].duplicated().any():
        raise WorkshopError(f"Cannot split the table because {ID_COLUMN} is duplicated.")
    y = _positive_array(table[TARGET_COLUMN])
    try:
        train_df, test_df = train_test_split(
            table,
            test_size=float(TEST_SIZE),
            random_state=int(RANDOM_STATE),
            stratify=y,
        )
    except ValueError as exc:
        raise WorkshopError(
            "The training/test split could not be created. "
            f"Check that {TARGET_COLUMN} contains {POSITIVE_LABEL!r} and one other label, "
            "and that TEST_SIZE leaves enough rows in each group."
        ) from exc
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    overlap = set(train_df[ID_COLUMN]) & set(test_df[ID_COLUMN])
    if overlap:
        raise WorkshopError("An identifier landed in both the training group and the test group. The split was stopped.")
    split = {
        "train": train_df,
        "test": test_df,
        "random_state": int(RANDOM_STATE),
        "test_size": float(TEST_SIZE),
    }
    split["fingerprint"] = split_fingerprint(split)
    print(
        f"Training rows: {len(train_df)}. "
        f"Test rows: {len(test_df)}. "
        f"Random seed: {int(RANDOM_STATE)}."
    )
    return split


def split_summary(split: dict) -> pd.DataFrame:
    """Show how many rows, and what share are positive, in each part of the split."""
    rows = []
    for label, frame in (("Training", split["train"]), ("Test", split["test"])):
        y = _positive_array(frame[TARGET_COLUMN])
        rows.append({
            "Group": label,
            "Rows": int(len(frame)),
            "Positive": int(y.sum()),
            "Negative": int((y == 0).sum()),
            "Positive rate": float(y.mean()),
        })
    return pd.DataFrame(rows)


def run_tabular_model(split: dict) -> dict:
    """Fit a logistic regression on the structured fields."""
    return _run_model(
        split["train"],
        split["test"],
        text_columns=[],
        use_tabular=True,
        use_text=False,
        name="Tabular data only",
        short_name="Tabular",
    )


def baseline_always_negative(split: dict) -> dict:
    """Score a rule that predicts every row is negative."""
    train_ids = split["train"][ID_COLUMN].astype(str).tolist()
    test_ids = split["test"][ID_COLUMN].astype(str).tolist()
    y_train = _positive_array(split["train"][TARGET_COLUMN])
    y_test = _positive_array(split["test"][TARGET_COLUMN])
    return _pack_result(
        name="Always predict negative",
        short_name="Always negative",
        threshold=float(DECISION_THRESHOLD),
        train_ids=train_ids,
        test_ids=test_ids,
        y_train=y_train,
        y_test=y_test,
        prob_train=np.zeros(len(y_train)),
        prob_test=np.zeros(len(y_test)),
        predictor_columns=[],
        n_unmatched_train=0,
        n_unmatched_test=0,
    )


def show_metrics(result: dict) -> pd.DataFrame:
    """Show training and test measures for one model."""
    rows = []
    for group, metrics in (("Training", result["train_metrics"]), ("Test", result["test_metrics"])):
        rows.append({
            "Model": result["name"],
            "Group": group,
            "Cutoff": result["threshold"],
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "ROC-AUC": metrics["roc_auc"],
        })
    return pd.DataFrame(rows)


def confusion_table(result: dict) -> pd.DataFrame:
    """Count test rows by the actual outcome and the model's flag."""
    labels = confusion_matrix(result["y_test"], result["pred_test"], labels=[0, 1])
    return pd.DataFrame(
        labels,
        index=["Actually negative", "Actually positive"],
        columns=["Predicted negative", "Predicted positive"],
    )


def apply_threshold(result: dict, threshold: float) -> dict:
    """Reuse stored scores and apply another cutoff. The model is not refitted."""
    threshold = float(threshold)
    if not 0 < threshold < 1:
        raise WorkshopError("The cutoff must be between 0 and 1.")
    updated = dict(result)
    updated["threshold"] = threshold
    updated["pred_train"] = (np.asarray(result["prob_train"]) >= threshold).astype(int)
    updated["pred_test"] = (np.asarray(result["prob_test"]) >= threshold).astype(int)
    updated["train_metrics"] = _metric_dict(updated["y_train"], updated["prob_train"], updated["pred_train"])
    updated["test_metrics"] = _metric_dict(updated["y_test"], updated["prob_test"], updated["pred_test"])
    return updated


def plot_confusion(result: dict):
    """Draw the test confusion counts."""
    table = confusion_table(result)
    values = table.to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], list(table.columns))
    ax.set_yticks([0, 1], list(table.index))
    threshold = float(values.max()) * 0.55
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = int(values[row, col])
            color = "white" if value > threshold else "black"
            ax.text(col, row, f"{value}", ha="center", va="center", fontsize=14, color=color)
    ax.set_title(f"{result['short_name']}: test rows, cutoff {result['threshold']:.2f}")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_roc(results: list[dict]):
    """Draw test ROC curves. The legend shows the test ROC-AUC."""
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for result in results:
        fpr, tpr, _ = roc_curve(result["y_test"], result["prob_test"])
        auc = result["test_metrics"]["roc_auc"]
        ax.plot(
            fpr,
            tpr,
            color=MODEL_COLORS.get(result["name"], "#333333"),
            linewidth=2,
            label=f"{result['short_name']} ({auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="No ranking ability (0.500)")
    ax.set_xlabel("Share of negative cases flagged (false alarm rate)")
    ax.set_ylabel("Share of positive cases flagged (recall)")
    ax.set_title("Test rows: ranking positive cases above negative cases")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_probabilities(result: dict):
    """Show the distribution of test scores."""
    scores = np.asarray(result["prob_test"])
    y = np.asarray(result["y_test"])
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.hist(scores[y == 0], bins=bins, alpha=0.7, color="#1f4e79", label="Negative")
    ax.hist(scores[y == 1], bins=bins, alpha=0.7, color="#b85c38", label="Positive")
    ax.axvline(result["threshold"], color="black", linestyle="--", linewidth=1, label=f"Cutoff {result['threshold']:.2f}")
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Test rows")
    ax.set_title(f"{result['short_name']}: test scores")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_metric_bars(results: list[dict]):
    """Compare test ROC-AUC, recall and precision."""
    metric_keys = [("ROC-AUC", "roc_auc"), ("Recall", "recall"), ("Precision", "precision")]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(results), 1)
    for index, result in enumerate(results):
        values = [result["test_metrics"][key] for _, key in metric_keys]
        positions = x + index * width
        ax.bar(
            positions,
            values,
            width=width,
            color=MODEL_COLORS.get(result["name"], "#333333"),
            label=result["short_name"],
        )
        for position, value in zip(positions, values):
            ax.text(position, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width * (len(results) - 1) / 2, [label for label, _ in metric_keys])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Test score")
    ax.set_title("Test performance")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(len(results), 3),
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def illustrate_text_encoding() -> dict:
    """Show TF-IDF and a two-number compression on three teaching lines.

    These lines are not records from the files and are not used in the models.
    """
    labels = [label for label, _ in ILLUSTRATION_NOTES]
    texts = [text for _, text in ILLUSTRATION_NOTES]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    tfidf = pd.DataFrame(matrix.toarray(), index=labels, columns=terms)
    tfidf.index.name = "Teaching line"

    svd = TruncatedSVD(n_components=2, random_state=int(RANDOM_STATE))
    compressed = svd.fit_transform(matrix)
    lsa = pd.DataFrame(compressed, index=labels, columns=["example_1", "example_2"])
    similarity = cosine_similarity(matrix)
    pairs = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            pairs.append((similarity[left, right], labels[left], labels[right]))
    pairs.sort(reverse=True)
    best = pairs[0]
    print("Teaching example only. These three lines are not used in the models.")
    print(
        f"Closest pair by word-score similarity: {best[1]} and {best[2]} "
        f"({best[0]:.3f})."
    )
    print("Word-score similarities:")
    for value, left, right in pairs:
        print(f"  {left} / {right}: {value:.3f}")
    print(
        f"The workshop compresses real documents into {int(N_TEXT_COMPONENTS)} numbers. "
        "The two columns in the compressed table are the same kind of compression on this three-line example."
    )
    return {"word_scores": _display_numbers(tfidf), "compressed": _display_numbers(lsa)}


def phrase_counts(table: pd.DataFrame, messages: pd.DataFrame, phrases) -> pd.DataFrame:
    """Count chosen phrases among messages with the positive label and the negative label."""
    phrases = list(phrases)
    if not phrases:
        raise WorkshopError("Pass at least one phrase to phrase_counts.")
    merged = messages.merge(
        table[[ID_COLUMN, TARGET_COLUMN]],
        on=ID_COLUMN,
        how="left",
        validate="many_to_one",
    )
    if merged[TARGET_COLUMN].isna().any():
        raise WorkshopError(
            f"Some texts have a {ID_COLUMN} that is missing from the tabular file. "
            "The phrase count was stopped so those texts are not dropped silently."
        )
    labels = merged[TARGET_COLUMN].astype(str).str.strip()
    positive = str(POSITIVE_LABEL).strip()
    negatives = sorted(set(labels.unique()) - {positive})
    if len(negatives) != 1:
        raise WorkshopError(
            f"{TARGET_COLUMN} must contain {positive!r} and exactly one other label. Found: {sorted(labels.unique())}."
        )
    negative = negatives[0]
    rows = []
    for phrase in phrases:
        hit = merged[TEXT_COLUMN].astype(str).str.contains(phrase, regex=False)
        rows.append({
            "Phrase": phrase,
            "Messages with the positive label": int((hit & (labels == positive)).sum()),
            "Messages with the negative label": int((hit & (labels == negative)).sum()),
        })
    return pd.DataFrame(rows)


def text_excerpts(table: pd.DataFrame, messages: pd.DataFrame, text_type: str) -> pd.DataFrame:
    """Show one text of a chosen type for each target label, in identifier order."""
    if not TEXT_TYPE_COLUMN or TEXT_TYPE_COLUMN not in messages.columns:
        raise WorkshopError("Set TEXT_TYPE_COLUMN before requesting excerpts.")
    merged = messages.merge(
        table[[ID_COLUMN, TARGET_COLUMN]],
        on=ID_COLUMN,
        how="left",
        validate="many_to_one",
    )
    chosen = merged[merged[TEXT_TYPE_COLUMN].astype(str) == str(text_type)].copy()
    chosen = chosen.sort_values([TARGET_COLUMN, ID_COLUMN, TEXT_ORDER_COLUMN])
    positive = str(POSITIVE_LABEL).strip()
    labels = chosen[TARGET_COLUMN].astype(str).str.strip()
    negatives = sorted(set(merged[TARGET_COLUMN].astype(str).str.strip().unique()) - {positive})
    outcomes = [positive, *negatives[:1]]
    rows = []
    for outcome in outcomes:
        subset = chosen[labels == outcome]
        if subset.empty:
            continue
        row = subset.iloc[0]
        text = " ".join(str(row[TEXT_COLUMN]).split())
        rows.append({
            TARGET_COLUMN: outcome,
            ID_COLUMN: row[ID_COLUMN],
            "Text type": row[TEXT_TYPE_COLUMN],
            "Excerpt": text[:280],
        })
    if len(rows) < 2:
        raise WorkshopError(f"Could not find a {text_type!r} text for both target labels.")
    return pd.DataFrame(rows)


def build_text_features(split: dict, messages: pd.DataFrame) -> dict:
    """Turn texts into text_1, text_2, ... for each identifier.

    Texts for the same identifier are joined in order.
    TF-IDF and LSA are fitted on the training rows only.
    """
    if TEXT_FILE is None:
        raise WorkshopError("Call load_datasets before building text features.")
    documents = _documents_for_ids(
        messages,
        pd.concat([split["train"], split["test"]], ignore_index=True)[ID_COLUMN],
    )
    return _build_text_feature_table(
        split,
        documents,
        cache_label="texts",
        corpus_fingerprint=_file_fingerprint(Path(TEXT_FILE)),
        aggregation=f"Join each identifier's texts in {TEXT_ORDER_COLUMN} order, separated by a blank line.",
    )


def build_row_text_features(split: dict, text_column: str) -> dict:
    """Turn one text column on the tabular file into the same kind of numeric columns.

    TF-IDF and LSA are fitted on the training rows only.
    """
    if not text_column:
        raise WorkshopError("Pass the name of a text column that already sits on the tabular file.")
    frames = pd.concat([split["train"], split["test"]], ignore_index=True)
    if text_column not in frames.columns:
        raise WorkshopError(f"The tabular file has no column {text_column!r}.")
    documents = (
        frames.assign(**{ID_COLUMN: frames[ID_COLUMN].astype(str)})
        .assign(document=frames[text_column].fillna("").astype(str))
        .loc[:, [ID_COLUMN, "document"]]
        .drop_duplicates(ID_COLUMN)
    )
    digest = hashlib.sha256("\n".join(documents["document"].astype(str)).encode()).hexdigest()[:20]
    return _build_text_feature_table(
        split,
        documents,
        cache_label="row_text",
        corpus_fingerprint=digest,
        aggregation=f"Use the {text_column} text already stored on the tabular row.",
    )


def preview_text_features(text_bundle: dict, n_rows: int = 5, n_columns: int = 5) -> pd.DataFrame:
    """Show the identifier and the first text columns."""
    frame = text_bundle["features"]
    text_columns = [column for column in frame.columns if column.startswith("text_")]
    columns = [ID_COLUMN, *text_columns[:n_columns]]
    preview = frame.loc[:, columns].head(n_rows).copy()
    for column in text_columns[:n_columns]:
        preview[column] = preview[column].map(lambda value: round(float(value), 3))
    return preview


def text_feature_summary(text_bundle: dict) -> pd.DataFrame:
    """Describe the text-feature table and how it was obtained."""
    saved = text_bundle.get("cache_path") or "Not saved in this run"
    rows = [
        ("Execution mode", text_bundle["mode"]),
        ("Rows", text_bundle["n_customers"]),
        ("Text columns", text_bundle["n_components"]),
        ("Aggregation", text_bundle["aggregation"]),
        ("Training documents used to fit TF-IDF and LSA", text_bundle["n_train_documents"]),
        ("Test documents used to fit TF-IDF and LSA", text_bundle["n_test_documents_in_fit"]),
        ("Empty documents, kept in the table", text_bundle["n_empty_documents"]),
        ("Word and phrase columns before compression", text_bundle["vocabulary_size"]),
        ("Share of training word-score variation retained", round(text_bundle["explained_variance_ratio_sum"], 3)),
        ("Seconds", round(text_bundle["elapsed_seconds"], 1)),
        ("Saved file", saved),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def combine_with_text(split: dict, text_bundle: dict) -> dict:
    """Attach text columns to the training and test rows.

    The match key is the identifier. Rows without text stay in the table,
    with text columns set to zero. A row is never duplicated.
    """
    features = text_bundle["features"].copy()
    features[ID_COLUMN] = features[ID_COLUMN].astype(str)
    if features[ID_COLUMN].duplicated().any():
        raise WorkshopError(
            f"The text-feature table has duplicate {ID_COLUMN} values. "
            "The match was stopped because it would duplicate rows."
        )
    text_columns = [column for column in features.columns if column.startswith("text_")]
    if not text_columns:
        raise WorkshopError("The text-feature table has no text_ columns.")

    def _match(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        left = frame.copy()
        left[ID_COLUMN] = left[ID_COLUMN].astype(str)
        merged = left.merge(features, on=ID_COLUMN, how="left", validate="one_to_one")
        unmatched = int(merged[text_columns[0]].isna().sum())
        merged[text_columns] = merged[text_columns].fillna(0.0)
        if len(merged) != len(frame):
            raise WorkshopError(
                "The match changed the number of rows. "
                "The workshop stopped so a duplicated or dropped row is not sent to the model."
            )
        if not merged[ID_COLUMN].astype(str).tolist() == frame[ID_COLUMN].astype(str).tolist():
            raise WorkshopError("The match reordered rows. The workshop stopped.")
        return merged, unmatched

    train, unmatched_train = _match(split["train"])
    test, unmatched_test = _match(split["test"])
    check = pd.DataFrame([
        {"Check": "Rows before the match", "Training": len(split["train"]), "Test": len(split["test"])},
        {"Check": "Rows after the match", "Training": len(train), "Test": len(test)},
        {"Check": "Rows with no text, kept and filled with zeros", "Training": unmatched_train, "Test": unmatched_test},
        {"Check": f"Duplicate {ID_COLUMN} after the match", "Training": int(train[ID_COLUMN].duplicated().sum()), "Test": int(test[ID_COLUMN].duplicated().sum())},
    ])
    print(f"Matched text columns on {ID_COLUMN}. Row counts must stay the same.")
    return {
        "train": train,
        "test": test,
        "text_columns": text_columns,
        "check": check,
        "n_unmatched_train": unmatched_train,
        "n_unmatched_test": unmatched_test,
        "split_fingerprint": split["fingerprint"],
    }


def run_combined_model(
    combined: dict,
    name: str = "Tabular data and text",
    short_name: str = "Tabular + text",
) -> dict:
    """Fit a logistic regression on structured fields plus text columns."""
    return _run_model(
        combined["train"],
        combined["test"],
        text_columns=combined["text_columns"],
        use_tabular=True,
        use_text=True,
        name=name,
        short_name=short_name,
        n_unmatched_train=combined["n_unmatched_train"],
        n_unmatched_test=combined["n_unmatched_test"],
    )


def run_text_only_model(
    combined: dict,
    name: str = "Text only",
    short_name: str = "Text only",
) -> dict:
    """Fit a logistic regression on the text columns only."""
    return _run_model(
        combined["train"],
        combined["test"],
        text_columns=combined["text_columns"],
        use_tabular=False,
        use_text=True,
        name=name,
        short_name=short_name,
        n_unmatched_train=combined["n_unmatched_train"],
        n_unmatched_test=combined["n_unmatched_test"],
    )


def run_row_text_models(split: dict, text_column: str) -> dict:
    """Repeat the text, match and logistic-regression steps with one text column on the table.

    The test rows stay the same as in the message-file models.
    """
    print(f"Row-text check: building text columns from {text_column}.")
    bundle = build_row_text_features(split, text_column)
    combined_data = combine_with_text(split, bundle)
    combined = run_combined_model(
        combined_data,
        name="Tabular data and row text",
        short_name="Tabular + row text",
    )
    text_only = run_text_only_model(
        combined_data,
        name="Row text only",
        short_name="Row text only",
    )
    print("Fitted the row-text models on the training rows and scored the same test rows.")
    return {
        "text_bundle": bundle,
        "combined_data": combined_data,
        "combined": combined,
        "text_only": text_only,
        "comparison": compare_models([combined, text_only]),
    }


def save_scores(result: dict, path) -> pd.DataFrame:
    """Write the test scores so a later session can load them without fitting again.

    The file has one row per test customer. ``customer_id`` is the identifier
    from this run, whatever that column was called in the source table.
    ``p0`` is the test score. ``observed_outcome`` is 1 for the positive label
    and 0 otherwise. ``path`` is the folder and the file name. The next
    workshop loads these scores only when its score path is this same path.
    """
    identifiers = [str(value) for value in result["test_customer_ids"]]
    scores = np.asarray(result["prob_test"], dtype=float)
    observed = np.asarray(result["y_test"], dtype=int)
    if not (len(identifiers) == len(scores) == len(observed)):
        raise WorkshopError("The test identifiers, scores and outcomes do not have the same length.")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise WorkshopError("A test score is outside 0 to 1, so the file was not written.")
    frame = pd.DataFrame({
        "customer_id": identifiers,
        "p0": scores,
        "observed_outcome": observed,
    })
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Saved {len(frame)} test scores to {destination}.")
    return frame


def save_model_rows(tabular: dict, combined: dict, path) -> pd.DataFrame:
    """Write both logistic scores for every fit row and every test row.

    ``save_scores`` writes one model's test rows. This file writes both models
    and both roles, so a later method can reload the same customers.

    ``customer_id`` is the identifier. ``role`` is ``fit`` or ``test``.
    ``p_logistic_tabular`` is the structured-fields score.
    ``p_logistic_text`` is the score that also uses the communications.
    On the test rows, ``p_logistic_text`` is the same number ``save_scores``
    stores as ``p0``. ``observed_outcome`` is 1 or 0.
    """
    def half(result: dict, role: str, id_key: str, y_key: str, prob_key: str, column: str) -> pd.DataFrame:
        return pd.DataFrame({
            "customer_id": [str(value) for value in result[id_key]],
            "role": role,
            "observed_outcome": np.asarray(result[y_key], dtype=int),
            column: np.asarray(result[prob_key], dtype=float),
        })

    tabular_rows = pd.concat([
        half(tabular, "fit", "train_customer_ids", "y_train", "prob_train", "p_logistic_tabular"),
        half(tabular, "test", "test_customer_ids", "y_test", "prob_test", "p_logistic_tabular"),
    ], ignore_index=True)
    text_rows = pd.concat([
        half(combined, "fit", "train_customer_ids", "y_train", "prob_train", "p_logistic_text"),
        half(combined, "test", "test_customer_ids", "y_test", "prob_test", "p_logistic_text"),
    ], ignore_index=True)
    text_rows = text_rows.drop(columns=["role", "observed_outcome"])
    frame = tabular_rows.merge(text_rows, on="customer_id", how="left", validate="one_to_one")
    if frame["p_logistic_text"].isna().any():
        raise WorkshopError("A customer has a tabular score and no communications-model score.")
    if set(frame["role"]) != {"fit", "test"}:
        raise WorkshopError("The score file must contain fit rows and test rows.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = frame[[
        "customer_id",
        "role",
        "p_logistic_tabular",
        "p_logistic_text",
        "observed_outcome",
    ]]
    frame.to_csv(destination, index=False)
    n_fit = int((frame["role"] == "fit").sum())
    n_test = int((frame["role"] == "test").sum())
    print(f"Saved {n_fit} fit rows and {n_test} test rows to {destination}.")
    return frame


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Place test measures for several models side by side."""
    rows = []
    for result in results:
        metrics = result["test_metrics"]
        rows.append({
            "Model": result["name"],
            "Test rows": int(len(result["y_test"])),
            "Positive": int(np.asarray(result["y_test"]).sum()),
            "Cutoff": result["threshold"],
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "ROC-AUC": metrics["roc_auc"],
        })
    return pd.DataFrame(rows)


def evaluation_checks(split: dict, results: list[dict], text_bundle: dict | None = None) -> pd.DataFrame:
    """Confirm that the models share the test rows and that text was fitted on training data."""
    expected_ids = split["test"][ID_COLUMN].astype(str).tolist()
    expected_y = _positive_array(split["test"][TARGET_COLUMN])
    train_ids = set(split["train"][ID_COLUMN].astype(str))
    test_ids = set(expected_ids)
    rows = []

    def add(check: str, ok: bool, detail: str) -> None:
        rows.append({"Check": check, "Status": "Pass" if ok else "Fail", "Detail": detail})

    add(
        "Training and test rows are disjoint",
        train_ids.isdisjoint(test_ids),
        f"{len(train_ids)} training rows and {len(test_ids)} test rows",
    )
    for result in results:
        same_ids = result["test_customer_ids"] == expected_ids
        same_y = np.array_equal(np.asarray(result["y_test"]), expected_y)
        add(
            f"{result['name']}: same test rows and same target labels",
            same_ids and same_y,
            f"{len(result['test_customer_ids'])} test rows",
        )
        banned = sorted(set(result.get("predictor_columns", [])) & _banned_predictors())
        predictors = result.get("predictor_columns", [])
        if result["name"] == "Always predict negative":
            continue
        add(
            f"{result['name']}: identifier and target excluded from predictors",
            not banned,
            "None of those fields are predictors" if not banned else ", ".join(banned),
        )
        if "n_unmatched_test" in result and predictors and any(str(column).startswith("text_") for column in predictors):
            add(
                f"{result['name']}: no test row dropped for missing text",
                result.get("n_dropped_test", 0) == 0 and len(result["test_customer_ids"]) == len(expected_ids),
                f"Missing text filled with zeros: {result.get('n_unmatched_test', 0)} test rows",
            )
    if text_bundle is not None:
        fit_ok = (
            text_bundle.get("n_test_documents_in_fit") == 0
            and text_bundle.get("n_train_documents") == len(split["train"])
            and text_bundle.get("split_fingerprint") == split["fingerprint"]
        )
        add(
            "TF-IDF and LSA fitted on the training rows only",
            fit_ok,
            (
                f"Training documents fitted: {text_bundle.get('n_train_documents')}. "
                f"Test documents fitted: {text_bundle.get('n_test_documents_in_fit')}."
            ),
        )
    frame = pd.DataFrame(rows)
    failed = frame.loc[frame["Status"] == "Fail"]
    if not failed.empty:
        details = "; ".join(f"{row.Check} ({row.Detail})" for row in failed.itertuples(index=False))
        raise WorkshopError(f"The shared-evaluation checks failed: {details}")
    return frame


def explain_accuracy(result: dict, baseline: dict):
    """State the test accuracy next to the always-negative rule."""
    accuracy = result["test_metrics"]["accuracy"]
    baseline_accuracy = baseline["test_metrics"]["accuracy"]
    recall = result["test_metrics"]["recall"]
    difference = accuracy - baseline_accuracy
    if difference < 0:
        accuracy_sentence = (
            f"Test accuracy is {accuracy:.3f}, which is {abs(difference):.3f} below the always-negative rule "
            f"({baseline_accuracy:.3f}) on the same rows."
        )
    elif difference < 0.05:
        accuracy_sentence = (
            f"Test accuracy is {accuracy:.3f}. The always-negative rule reaches {baseline_accuracy:.3f} "
            f"on the same rows, so the model is only {difference:.3f} above that rule."
        )
    else:
        accuracy_sentence = (
            f"Test accuracy is {accuracy:.3f}, which is {difference:.3f} above the always-negative rule "
            f"({baseline_accuracy:.3f}) on the same rows."
        )
    text = (
        f"{accuracy_sentence} Test recall is {recall:.3f}: at the current cutoff the model flags "
        "that share of the positive test rows. Accuracy and recall answer different questions. "
        "Accuracy alone is a weak guide when most rows are negative."
    )
    return _as_markdown(text)


def explain_scores(result: dict):
    """Compare the average test score with the test positive rate."""
    mean_score = float(np.mean(result["prob_test"]))
    rate = float(np.mean(result["y_test"]))
    gap = mean_score - rate
    text = (
        f"On the test rows, the average predicted score is {mean_score:.3f}. "
        f"The positive share is {rate:.3f}. "
    )
    if gap > 0.05:
        text += (
            "The average score sits above the positive rate. The model was told to pay extra attention "
            "to the positive class, so a score of 0.70 is a ranking signal rather than a statement "
            "that the chance of the positive class is 70 percent. The cutoff that triggers an action is a business choice."
        )
    elif gap < -0.05:
        text += (
            "The average score sits below the positive rate. Use the scores to rank rows, "
            "and choose the cutoff from the number of rows the business can act on."
        )
    else:
        text += (
            "The average score is close to the positive rate on this test set. "
            "That is a useful check, and the cutoff that triggers an action remains a business choice."
        )
    return _as_markdown(text)


def explain_train_test(result: dict):
    """Compare training and test ROC-AUC for one model."""
    train_auc = result["train_metrics"]["roc_auc"]
    test_auc = result["test_metrics"]["roc_auc"]
    gap = train_auc - test_auc
    text = f"Training ROC-AUC is {train_auc:.3f}. Test ROC-AUC is {test_auc:.3f}. "
    if train_auc >= 0.99 and test_auc >= 0.99:
        text += (
            "Both scores are extremely high, including on rows the model had not seen. "
            "Treat that as a warning before acting on the score."
        )
        return _as_markdown(text)
    if gap > 0.03:
        text += (
            "The training score is higher. The model fits the rows it learned from more closely "
            "than the rows it had not seen. Use the test score when you discuss prediction."
        )
    else:
        text += (
            "The two scores are close. The test score is the one that describes rows the model had not seen."
        )
    return _as_markdown(text)


def explain_confusion(result: dict):
    """Describe the test confusion counts."""
    table = confusion_table(result)
    negative_ok = int(table.loc["Actually negative", "Predicted negative"])
    false_alarms = int(table.loc["Actually negative", "Predicted positive"])
    missed = int(table.loc["Actually positive", "Predicted negative"])
    found = int(table.loc["Actually positive", "Predicted positive"])
    text = (
        f"At a cutoff of {result['threshold']:.2f}, the model flags {found} of the "
        f"{found + missed} positive test rows and misses {missed}. "
        f"It also flags {false_alarms} negative test rows, and leaves {negative_ok} negative rows unflagged. "
        "A flag is a score at or above the cutoff. It is an input to a decision, and it is itself a decision only after the business accepts that cutoff."
    )
    return _as_markdown(text)


def explain_threshold_change(original: dict, alternative: dict):
    """Describe how precision and recall move when the cutoff changes."""
    text = (
        f"At a cutoff of {original['threshold']:.2f}, test recall is {original['test_metrics']['recall']:.3f} "
        f"and test precision is {original['test_metrics']['precision']:.3f}. "
        f"At a cutoff of {alternative['threshold']:.2f}, test recall is {alternative['test_metrics']['recall']:.3f} "
        f"and test precision is {alternative['test_metrics']['precision']:.3f}. "
        "A lower cutoff flags more rows: it usually finds more of the positive rows and also flags more negative rows. "
        "The rest of the workshop keeps the cutoff at "
        f"{float(DECISION_THRESHOLD):.2f} so the models are compared at the same rule."
    )
    return _as_markdown(text)


def explain_score_shape(result: dict):
    """Describe whether test scores sit in the middle or at the extremes."""
    scores = np.asarray(result["prob_test"])
    share_extreme = float(np.mean((scores < 0.01) | (scores > 0.99)))
    if share_extreme > 0.5:
        text = (
            f"On the test rows, {share_extreme:.0%} of the scores are below 0.01 or above 0.99. "
            "The model is very sure about most of these rows. "
            "A score that sure, on rows the model had not seen, needs a data check before anyone acts on it."
        )
    else:
        text = (
            f"On the test rows, {share_extreme:.0%} of the scores are below 0.01 or above 0.99. "
            "Many rows sit between those extremes, so the cutoff changes who would be flagged."
        )
    return _as_markdown(text)


def explain_model_comparison(results: list[dict]):
    """State the test comparison, including a large gain that needs a data check."""
    by_name = {result["name"]: result for result in results}
    required = ("Tabular data only", "Tabular data and text", "Text only")
    missing = [name for name in required if name not in by_name]
    if missing:
        raise WorkshopError(f"The comparison is missing: {', '.join(missing)}.")
    tabular = by_name["Tabular data only"]
    combined = by_name["Tabular data and text"]
    text_only = by_name["Text only"]
    tab = tabular["test_metrics"]
    comb = combined["test_metrics"]
    text = text_only["test_metrics"]
    gap = comb["roc_auc"] - tab["roc_auc"]
    paragraphs = [
        (
            f"On the same test rows, ROC-AUC is {tab['roc_auc']:.3f} for the tabular model, "
            f"{comb['roc_auc']:.3f} for the tabular model plus text, and "
            f"{text['roc_auc']:.3f} for the text-only model. "
            f"The combined model differs from the tabular model by {gap:+.3f} ROC-AUC."
        ),
        (
            f"At a cutoff of {tabular['threshold']:.2f}, test recall is {tab['recall']:.3f}, "
            f"{comb['recall']:.3f} and {text['recall']:.3f}. Test precision is {tab['precision']:.3f}, "
            f"{comb['precision']:.3f} and {text['precision']:.3f}."
        ),
    ]
    if max(comb["roc_auc"], text["roc_auc"]) >= 0.99:
        paragraphs.append(
            "A test ROC-AUC this close to 1 means the model almost always ranks a positive row "
            "above a negative row. A gain of this size is a warning that the notes may already contain the outcome. "
            "It is a weak basis for acting on the score."
        )
    elif gap > 0.02:
        paragraphs.append(
            "The combined model ranks the test rows better than the tabular model by a visible margin on this split. "
            "A visible margin on one split is a reason to ask whether the notes contain new information and whether they were available before the outcome. "
            "It becomes a business case only if the gain is large enough to pay for collecting and processing the texts, under the company's capacity to act."
        )
    elif gap > 0.005:
        paragraphs.append(
            "The combined model is slightly ahead of the tabular model on this test set. "
            "A difference of this size on one split is too small to justify, by itself, the cost of processing every note. "
            "The band of 0.005 is a reading guide for this workshop. A formal statistical test of the difference is outside this session."
        )
    elif gap < -0.005:
        paragraphs.append(
            "The combined model ranks the test rows below the tabular model. "
            "Adding these text features did not improve the ranking of rows the model had not seen."
        )
    else:
        paragraphs.append(
            "The combined model and the tabular model are tied for practical purposes on test ROC-AUC. "
            "These text features did not improve the ranking of unseen rows. "
            "The reading band of 0.005 is a workshop guide for a very small difference."
        )
    paragraphs.append(
        "ROC-AUC measures ranking across cutoffs. Recall and precision measure one chosen cutoff. "
        "A better ranking can support a decision. The offer, the budget and the choice of who may be contacted remain with the business. "
        "A higher score does not show that a field caused the positive outcome."
    )
    return _as_markdown("\n\n".join(paragraphs))


def explain_row_text_check(tabular: dict, row_text_models: dict):
    """Interpret the row-text models on the same test rows."""
    tab = tabular["test_metrics"]["roc_auc"]
    combined = row_text_models["combined"]["test_metrics"]["roc_auc"]
    only = row_text_models["text_only"]["test_metrics"]["roc_auc"]
    gap = combined - tab
    paragraphs = [
        (
            f"On the same test rows, ROC-AUC is {tab:.3f} with the structured fields, "
            f"{combined:.3f} when the row text is added, and {only:.3f} with the row text alone."
        )
    ]
    if abs(gap) < 0.01:
        paragraphs.append(
            f"Adding the row text changes test ROC-AUC by {gap:+.3f}. "
            "A change this small is what text that restates the structured fields should produce. "
            "The band of 0.01 is a reading guide for this workshop. A formal statistical test of the difference is outside this session."
        )
    elif gap >= 0.01:
        paragraphs.append(
            f"Adding the row text raises test ROC-AUC by {gap:.3f} on this split. "
            "Read that figure together with the cost of maintaining the written field. One split leaves uncertainty around the number."
        )
    else:
        paragraphs.append(
            f"Adding the row text lowers test ROC-AUC by {abs(gap):.3f} on this split. "
            "Rewriting the same facts as text did not improve the ranking."
        )
    if only + 0.02 < tab:
        paragraphs.append(
            f"The row text alone, at {only:.3f}, ranks rows less strongly than the structured fields, at {tab:.3f}. "
            "The original columns store those facts exactly. Compressing them into word scores keeps broad patterns and drops some of that detail."
        )
    elif only > tab + 0.02:
        paragraphs.append(
            f"The row text alone ranks rows more strongly than the structured fields on this split "
            f"({only:.3f} against {tab:.3f}). If that text was written from those same fields, a data check should come before anyone treats it as new information."
        )
    else:
        paragraphs.append(
            "The row text alone recovers a similar ranking to the structured fields. "
            "That fits text that repeats those fields."
        )
    paragraphs.append(
        "This check uses a text column stored on the tabular row. "
        "It asks whether restating the row in words improves the ranking. "
        "A separate message file can contain facts that the structured fields never record."
    )
    return _as_markdown("\n\n".join(paragraphs))

def split_fingerprint(split: dict) -> str:
    """Fingerprint the exact training and test rows and their target labels."""
    def pack(frame: pd.DataFrame) -> str:
        pairs = sorted(
            f"{identifier}:{label}"
            for identifier, label in zip(frame[ID_COLUMN].astype(str), frame[TARGET_COLUMN].astype(str), strict=True)
        )
        return ",".join(pairs)

    payload = f"{split['random_state']}|{split['test_size']}|{pack(split['train'])}|{pack(split['test'])}"
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _run_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_columns: list[str],
    use_tabular: bool,
    use_text: bool,
    name: str,
    short_name: str,
    n_unmatched_train: int = 0,
    n_unmatched_test: int = 0,
) -> dict:
    _validate_settings()
    _require_roles()
    y_train = _positive_array(train_df[TARGET_COLUMN])
    y_test = _positive_array(test_df[TARGET_COLUMN])
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise WorkshopError(
            "Training and test groups both need positive rows and negative rows. "
            "ROC-AUC would be undefined otherwise."
        )
    x_train, x_test, predictors = _design_matrices(
        train_df, test_df, text_columns, use_tabular=use_tabular, use_text=use_text
    )
    model = _logistic_regression()
    model.fit(x_train, y_train)
    if np.max(model.n_iter_) >= model.max_iter:
        print(
            "The logistic regression reached its iteration limit. "
            "The scores were still produced, and they deserve extra caution."
        )
    print(
        f"Fitted {name} on {len(train_df)} training rows "
        f"with {x_train.shape[1]} predictors. Scored {len(test_df)} test rows."
    )
    return _pack_result(
        name=name,
        short_name=short_name,
        threshold=float(DECISION_THRESHOLD),
        train_ids=train_df[ID_COLUMN].astype(str).tolist(),
        test_ids=test_df[ID_COLUMN].astype(str).tolist(),
        y_train=y_train,
        y_test=y_test,
        prob_train=model.predict_proba(x_train)[:, 1],
        prob_test=model.predict_proba(x_test)[:, 1],
        predictor_columns=predictors,
        n_unmatched_train=n_unmatched_train,
        n_unmatched_test=n_unmatched_test,
    )


def _tabular_preprocessor() -> ColumnTransformer:
    """Scale the numeric fields and dummy-code each categorical field.

    One level of each categorical field is left out, so a field with three
    levels becomes two dummies and a binary field becomes one dummy.
    """
    transformers = []
    if list(NUMERIC_FEATURES):
        transformers.append(("num", StandardScaler(), list(NUMERIC_FEATURES)))
    if list(CATEGORICAL_FEATURES):
        transformers.append((
            "cat",
            OneHotEncoder(drop="first", sparse_output=False),
            list(CATEGORICAL_FEATURES),
        ))
    if not transformers:
        raise WorkshopError("Set NUMERIC_FEATURES, CATEGORICAL_FEATURES, or both.")
    return ColumnTransformer(transformers=transformers)


def _design_matrices(train_df, test_df, text_columns, use_tabular: bool, use_text: bool):
    parts_train = []
    parts_test = []
    predictors: list[str] = []
    source_columns: list[str] = []
    if use_tabular:
        feature_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
        _require_frame_columns(train_df, feature_columns)
        _require_frame_columns(test_df, feature_columns)
        missing_train = train_df[feature_columns].isna().sum()
        missing_test = test_df[feature_columns].isna().sum()
        missing = missing_train.add(missing_test, fill_value=0)
        missing = missing[missing > 0]
        if not missing.empty:
            details = ", ".join(f"{name} ({int(count)})" for name, count in missing.items())
            raise WorkshopError(
                f"Missing values remain in predictor fields: {details}. "
                "The workshop does not guess a replacement for these fields."
            )
        preprocessor = _tabular_preprocessor()
        parts_train.append(np.asarray(preprocessor.fit_transform(train_df), dtype=float))
        parts_test.append(np.asarray(preprocessor.transform(test_df), dtype=float))
        if list(NUMERIC_FEATURES):
            predictors.extend(list(NUMERIC_FEATURES))
        if list(CATEGORICAL_FEATURES):
            categorical_names = list(
                preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
            )
            predictors.extend(categorical_names)
        source_columns.extend(feature_columns)
    if use_text:
        if not text_columns:
            raise WorkshopError("The text model received no text columns.")
        _require_frame_columns(train_df, text_columns)
        _require_frame_columns(test_df, text_columns)
        text_train = train_df[text_columns].to_numpy(dtype=float)
        text_test = test_df[text_columns].to_numpy(dtype=float)
        if np.isnan(text_train).any() or np.isnan(text_test).any():
            raise WorkshopError(
                "Text features still contain missing values. "
                "Missing text should be filled with zeros before the model is fitted."
            )
        parts_train.append(text_train)
        parts_test.append(text_test)
        predictors.extend(list(text_columns))
        source_columns.extend(list(text_columns))
    if not parts_train:
        raise WorkshopError("The model has no predictors.")
    _reject_banned_predictors(source_columns)
    x_train = np.hstack(parts_train) if len(parts_train) > 1 else parts_train[0]
    x_test = np.hstack(parts_test) if len(parts_test) > 1 else parts_test[0]
    return x_train, x_test, predictors


def _logistic_regression():
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=int(RANDOM_STATE),
        solver="lbfgs",
    )


def _pack_result(
    name: str,
    short_name: str,
    threshold: float,
    train_ids: list[str],
    test_ids: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    prob_train: np.ndarray,
    prob_test: np.ndarray,
    predictor_columns: list[str],
    n_unmatched_train: int,
    n_unmatched_test: int,
) -> dict:
    pred_train = (np.asarray(prob_train) >= threshold).astype(int)
    pred_test = (np.asarray(prob_test) >= threshold).astype(int)
    return {
        "name": name,
        "short_name": short_name,
        "threshold": float(threshold),
        "train_customer_ids": list(train_ids),
        "test_customer_ids": list(test_ids),
        "y_train": np.asarray(y_train),
        "y_test": np.asarray(y_test),
        "prob_train": np.asarray(prob_train, dtype=float),
        "prob_test": np.asarray(prob_test, dtype=float),
        "pred_train": pred_train,
        "pred_test": pred_test,
        "train_metrics": _metric_dict(y_train, prob_train, pred_train),
        "test_metrics": _metric_dict(y_test, prob_test, pred_test),
        "predictor_columns": list(predictor_columns),
        "n_unmatched_train": int(n_unmatched_train),
        "n_unmatched_test": int(n_unmatched_test),
        "n_dropped_train": 0,
        "n_dropped_test": 0,
    }


def _metric_dict(y_true, probabilities, predictions) -> dict:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": _safe_auc(y_true, probabilities),
    }


def _display_numbers(frame: pd.DataFrame) -> pd.DataFrame:
    tidy = frame.round(3).copy()
    numeric = tidy.select_dtypes(include="number").columns
    tidy[numeric] = tidy[numeric].mask(tidy[numeric].abs() < 5e-4, 0.0)
    return tidy


def _safe_auc(y_true, probabilities) -> float:
    if len(np.unique(probabilities)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, probabilities))


def _documents_for_ids(messages: pd.DataFrame, identifiers: pd.Series) -> pd.DataFrame:
    ordered = messages.sort_values([ID_COLUMN, TEXT_ORDER_COLUMN], kind="mergesort")
    grouped = (
        ordered.groupby(ID_COLUMN, sort=False)[TEXT_COLUMN]
        .agg(lambda values: "\n\n".join(str(value) for value in values if pd.notna(value)))
    )
    wanted = pd.Index(identifiers.astype(str).tolist(), name=ID_COLUMN)
    documents = grouped.reindex(wanted).fillna("")
    return documents.rename("document").reset_index()


def _build_text_feature_table(
    split: dict,
    documents: pd.DataFrame,
    cache_label: str,
    corpus_fingerprint: str,
    aggregation: str,
) -> dict:
    _validate_settings()
    _require_roles()
    mode = _text_mode()
    n_components = int(N_TEXT_COMPONENTS)
    expected = _expected_cache_meta(split, cache_label, corpus_fingerprint, aggregation, n_components)
    stem = f"{cache_label}_k{n_components}"
    if mode == "LOAD":
        started = time.perf_counter()
        frame, meta = _read_cache(stem, expected, cache_label, n_components)
        bundle = _bundle_from_frame(frame, meta, mode, time.perf_counter() - started, _cache_csv(stem))
        print(
            f"Loaded saved {cache_label} text features for {bundle['n_customers']} rows "
            f"and {bundle['n_components']} columns. The text step was not rerun."
        )
        return bundle

    started = time.perf_counter()
    frame, fit_meta = _fit_text_features(split, documents, n_components)
    elapsed = time.perf_counter() - started
    meta = {**expected, **fit_meta}
    cache_path = None
    if mode == "SAVE":
        cache_path = _write_cache(frame, meta, stem)
        print(f"Saved {cache_label} text features to {_display_path(cache_path)}.")
    else:
        print(f"Computed {cache_label} text features. Results were not saved.")
    bundle = _bundle_from_frame(frame, meta, mode, elapsed, cache_path)
    print(
        f"{bundle['n_components']} text columns for {bundle['n_customers']} rows. "
        f"Fitted on {bundle['n_train_documents']} training documents. "
        f"Variation retained: {bundle['explained_variance_ratio_sum']:.3f}. "
        f"Seconds: {bundle['elapsed_seconds']:.1f}."
    )
    return bundle


def _fit_text_features(split: dict, documents: pd.DataFrame, n_components: int):
    documents = documents.copy()
    documents[ID_COLUMN] = documents[ID_COLUMN].astype(str)
    if documents[ID_COLUMN].duplicated().any():
        raise WorkshopError("Each identifier needs one document before TF-IDF is fitted.")
    lookup = documents.set_index(ID_COLUMN)["document"].astype(str)
    train_ids = split["train"][ID_COLUMN].astype(str).tolist()
    test_ids = split["test"][ID_COLUMN].astype(str).tolist()
    if set(train_ids) & set(test_ids):
        raise WorkshopError("Text features were stopped because an identifier is in both groups.")
    missing_ids = [identifier for identifier in [*train_ids, *test_ids] if identifier not in lookup.index]
    if missing_ids:
        raise WorkshopError(
            f"{len(missing_ids)} identifiers have no document row. "
            "The text step expected every training and test row, including rows with empty text."
        )
    train_docs = lookup.loc[train_ids].tolist()
    test_docs = lookup.loc[test_ids].tolist()
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_SETTINGS["max_features"],
        ngram_range=TFIDF_SETTINGS["ngram_range"],
        min_df=TFIDF_SETTINGS["min_df"],
        stop_words=TFIDF_SETTINGS["stop_words"],
    )
    train_word_scores = vectorizer.fit_transform(train_docs)
    vocabulary_size = int(len(vectorizer.vocabulary_))
    max_components = min(n_components, train_word_scores.shape[0] - 1, train_word_scores.shape[1] - 1)
    if max_components < 2:
        raise WorkshopError(
            "The training text is too small to build at least two compressed columns. "
            "Check that the text file is available, or lower N_TEXT_COMPONENTS only after a data check."
        )
    if max_components != n_components:
        print(
            f"N_TEXT_COMPONENTS was {n_components}. "
            f"The training text can support {max_components}, so that smaller number was used."
        )
    svd = TruncatedSVD(n_components=max_components, random_state=int(RANDOM_STATE))
    train_features = svd.fit_transform(train_word_scores)
    test_features = svd.transform(vectorizer.transform(test_docs))
    columns = [f"text_{index}" for index in range(1, max_components + 1)]
    train_frame = pd.DataFrame(train_features, columns=columns)
    train_frame.insert(0, ID_COLUMN, train_ids)
    test_frame = pd.DataFrame(test_features, columns=columns)
    test_frame.insert(0, ID_COLUMN, test_ids)
    frame = pd.concat([train_frame, test_frame], ignore_index=True)
    empty = sum(1 for document in [*train_docs, *test_docs] if not str(document).strip())
    fit_meta = {
        "n_components": int(max_components),
        "n_train_documents": int(len(train_docs)),
        "n_test_documents_in_fit": 0,
        "n_empty_documents": int(empty),
        "vocabulary_size": vocabulary_size,
        "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
        "fit_excludes_test": True,
    }
    return frame, fit_meta


def _expected_cache_meta(split, cache_label, corpus_fingerprint, aggregation, n_components) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "cache_label": cache_label,
        "n_components": int(n_components),
        "random_state": int(split["random_state"]),
        "test_size": float(split["test_size"]),
        "split_fingerprint": split["fingerprint"],
        "corpus_fingerprint": corpus_fingerprint,
        "aggregation": aggregation,
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "positive_label": str(POSITIVE_LABEL),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "tfidf": {
            "max_features": TFIDF_SETTINGS["max_features"],
            "ngram_range": list(TFIDF_SETTINGS["ngram_range"]),
            "min_df": TFIDF_SETTINGS["min_df"],
            "stop_words": TFIDF_SETTINGS["stop_words"],
        },
    }


def _bundle_from_frame(frame: pd.DataFrame, meta: dict, mode: str, elapsed: float, cache_path: Path | None) -> dict:
    text_columns = [column for column in frame.columns if column.startswith("text_")]
    return {
        "features": frame,
        "mode": mode,
        "aggregation": meta["aggregation"],
        "n_customers": int(frame[ID_COLUMN].nunique()),
        "n_components": int(len(text_columns)),
        "n_train_documents": int(meta["n_train_documents"]),
        "n_test_documents_in_fit": int(meta["n_test_documents_in_fit"]),
        "n_empty_documents": int(meta["n_empty_documents"]),
        "vocabulary_size": int(meta["vocabulary_size"]),
        "explained_variance_ratio_sum": float(meta["explained_variance_ratio_sum"]),
        "split_fingerprint": meta["split_fingerprint"],
        "elapsed_seconds": float(elapsed),
        "cache_path": _display_path(cache_path) if cache_path else "",
    }


def _read_cache(stem: str, expected: dict, cache_label: str, n_components: int) -> tuple[pd.DataFrame, dict]:
    csv_path = _cache_csv(stem)
    meta_path = _cache_meta(stem)
    if not csv_path.exists() or not meta_path.exists():
        available = sorted(path.name for path in CACHE_DIR.glob(f"{cache_label}_k*.meta.json")) if CACHE_DIR.exists() else []
        available_text = ", ".join(available) if available else "none"
        raise WorkshopError(
            f"No saved {cache_label} text features were found for {n_components} components. "
            f"Expected file: {_display_path(csv_path)}. "
            f"Saved metadata files for this text source: {available_text}. "
            "LOAD continues only after a successful SAVE. "
            'Set TEXT_MODE to "SAVE" to compute the features and store them, '
            'or "RUN" to compute them for this session without storing them.'
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(csv_path, dtype={ID_COLUMN: str})
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise WorkshopError(
            f"The saved {cache_label} text features could not be read. "
            'Set TEXT_MODE to "SAVE" to create them again.'
        ) from exc
    problems = _cache_problems(meta, expected)
    if problems:
        raise WorkshopError(
            "The saved text features do not match this workshop run. "
            + " ".join(problems)
            + ' Set TEXT_MODE to "SAVE" to recompute and replace the saved file, '
            'or "RUN" to recompute for this session only.'
        )
    text_columns = [column for column in frame.columns if str(column).startswith("text_")]
    if frame[ID_COLUMN].duplicated().any() or len(text_columns) != int(meta["n_components"]):
        raise WorkshopError(
            "The saved text features are internally inconsistent. "
            'Set TEXT_MODE to "SAVE" to create them again.'
        )
    return frame, meta


def _cache_problems(meta: dict, expected: dict) -> list[str]:
    labels = {
        "cache_version": "The saved file was produced by a different version of the workshop code.",
        "cache_label": "The saved file belongs to a different text source.",
        "n_components": "The number of text columns does not match the current settings.",
        "random_state": "The random seed does not match the current settings.",
        "test_size": "The test-set size does not match the current settings.",
        "split_fingerprint": "The training and test rows do not match the current split.",
        "id_column": "The identifier column does not match the current settings.",
        "target_column": "The target column does not match the current settings.",
        "positive_label": "The positive label does not match the current settings.",
        "numeric_features": "The numeric feature list does not match the current settings.",
        "categorical_features": "The categorical feature list does not match the current settings.",
        "corpus_fingerprint": "The text file does not match the file used when those features were saved.",
        "aggregation": "The way messages are combined does not match the current workshop code.",
        "tfidf": "The TF-IDF settings do not match the current workshop code.",
    }
    problems = []
    for key, sentence in labels.items():
        if meta.get(key) != expected.get(key):
            problems.append(sentence)
    for key in ("n_train_documents", "n_test_documents_in_fit", "vocabulary_size", "explained_variance_ratio_sum"):
        if key not in meta:
            problems.append("The saved metadata is incomplete.")
            break
    return problems


def _write_cache(frame: pd.DataFrame, meta: dict, stem: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _cache_csv(stem)
    meta_path = _cache_meta(stem)
    partial_csv = csv_path.with_name(csv_path.name + ".partial")
    partial_meta = meta_path.with_name(meta_path.name + ".partial")
    frame.to_csv(partial_csv, index=False, float_format="%.12g")
    partial_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    partial_csv.replace(csv_path)
    partial_meta.replace(meta_path)
    return csv_path


def _cache_csv(stem: str) -> Path:
    return CACHE_DIR / f"{stem}.csv"


def _cache_meta(stem: str) -> Path:
    return CACHE_DIR / f"{stem}.meta.json"


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:20]


def _positive_array(values: pd.Series) -> np.ndarray:
    labels = values.astype(str).str.strip()
    positive = str(POSITIVE_LABEL).strip()
    unknown = sorted(set(labels.unique()) - {positive})
    if positive not in set(labels.unique()) or len(unknown) != 1:
        raise WorkshopError(
            f"{TARGET_COLUMN} must contain the positive label {positive!r} and exactly one other label. "
            f"Found: {sorted(labels.unique())}."
        )
    return (labels == positive).astype(int).to_numpy()


def _banned_predictors() -> set[str]:
    banned = {
        ID_COLUMN,
        TARGET_COLUMN,
        TEXT_COLUMN,
        TEXT_ORDER_COLUMN,
        "document",
        "source_row",
        "n_texts",
    }
    if TEXT_TYPE_COLUMN:
        banned.add(TEXT_TYPE_COLUMN)
    if TABLE_TEXT_COLUMN:
        banned.add(TABLE_TEXT_COLUMN)
    return banned


def _reject_banned_predictors(columns: list[str]) -> None:
    found = sorted(set(columns) & _banned_predictors())
    if found:
        raise WorkshopError(
            "The model was stopped because these fields cannot be used as ordinary predictors: "
            + ", ".join(found)
            + ". The identifier names a row and the target is the answer."
        )


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise WorkshopError(
            f"The {label} was not found at {_display_path(path)}. "
            "Open a terminal in the project folder, the folder that contains _quarto.yml, and run the workshop from there."
        )
    return path


def _require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise WorkshopError(
            f"{_display_path(path)} is missing columns: {', '.join(missing)}."
        )


def _require_frame_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise WorkshopError(f"The modelling table is missing columns: {', '.join(missing)}.")


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _require_roles() -> None:
    missing = [
        name
        for name, value in (
            ("ID_COLUMN", ID_COLUMN),
            ("TARGET_COLUMN", TARGET_COLUMN),
            ("POSITIVE_LABEL", POSITIVE_LABEL),
            ("TEXT_COLUMN", TEXT_COLUMN),
            ("TEXT_ORDER_COLUMN", TEXT_ORDER_COLUMN),
        )
        if value is None or str(value).strip() == ""
    ]
    if missing:
        raise WorkshopError("Set these before loading or fitting: " + ", ".join(missing) + ".")
    if not list(NUMERIC_FEATURES) and not list(CATEGORICAL_FEATURES):
        raise WorkshopError("Set NUMERIC_FEATURES, CATEGORICAL_FEATURES, or both, before loading or fitting.")
    overlap = sorted(set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES))
    if overlap:
        raise WorkshopError("A column is listed as both numeric and categorical: " + ", ".join(overlap) + ".")
    reserved = {ID_COLUMN, TARGET_COLUMN} & (set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES))
    if reserved:
        raise WorkshopError(
            "The identifier and the target cannot be predictors: " + ", ".join(sorted(reserved)) + "."
        )


def _validate_settings() -> None:
    _text_mode()
    if not isinstance(RANDOM_STATE, int):
        raise WorkshopError("RANDOM_STATE must be a whole number. The workshop uses 42.")
    if not 0 < float(TEST_SIZE) < 0.5:
        raise WorkshopError(
            "TEST_SIZE must be greater than 0 and below 0.5. "
            "The workshop uses 0.25 so most rows are available for learning and a large group remains for testing."
        )
    components = int(N_TEXT_COMPONENTS)
    if components < 2 or components > 300:
        raise WorkshopError("N_TEXT_COMPONENTS must be between 2 and 300. The workshop uses 64.")
    if not 0 < float(DECISION_THRESHOLD) < 1:
        raise WorkshopError("DECISION_THRESHOLD must be between 0 and 1. The workshop uses 0.5.")


def _text_mode() -> str:
    mode = str(TEXT_MODE).strip().upper()
    if mode not in {"RUN", "SAVE", "LOAD"}:
        raise WorkshopError(
            f'TEXT_MODE is {TEXT_MODE!r}. Use "RUN", "SAVE" or "LOAD".'
        )
    return mode


def _as_markdown(text: str):
    try:
        from IPython.display import Markdown
    except ImportError:
        return text
    return Markdown(text)

"""Tabular plus text classification.

Call ``run`` with the table file, the text file, the column roles and the
feature lists. This file does not name a particular dataset.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ID_COLUMN = "id"
TARGET_COLUMN = "target"
POSITIVE_LABEL = "1"
TEXT_COLUMN = "text"
ORDER_COLUMN = "text_index"
TYPE_COLUMN = ""
NUMERIC_FEATURES: list[str] = []
CATEGORICAL_FEATURES: list[str] = []
RESULTS_DIR = Path("data/processed/text_model_results")
RANDOM_STATE = 42
TEST_SIZE = 0.25
SVD_DIM = int(os.environ.get("SVD_DIM", "64"))


def configure(
    id_column: str,
    target_column: str,
    positive_label: str,
    numeric_features: list[str],
    categorical_features: list[str],
    text_column: str = "text",
    order_column: str = "text_index",
    type_column: str = "",
    results_dir: str | Path = "data/processed/text_model_results",
    random_state: int = 42,
    test_size: float = 0.25,
    svd_dim: int | None = None,
) -> None:
    """Store the column roles used by every function in this file."""
    global ID_COLUMN, TARGET_COLUMN, POSITIVE_LABEL, TEXT_COLUMN, ORDER_COLUMN, TYPE_COLUMN
    global NUMERIC_FEATURES, CATEGORICAL_FEATURES, RESULTS_DIR, RANDOM_STATE, TEST_SIZE, SVD_DIM
    ID_COLUMN = id_column
    TARGET_COLUMN = target_column
    POSITIVE_LABEL = str(positive_label)
    TEXT_COLUMN = text_column
    ORDER_COLUMN = order_column
    TYPE_COLUMN = type_column or ""
    NUMERIC_FEATURES = list(numeric_features)
    CATEGORICAL_FEATURES = list(categorical_features)
    RESULTS_DIR = Path(results_dir)
    RANDOM_STATE = int(random_state)
    TEST_SIZE = float(test_size)
    if svd_dim is not None:
        SVD_DIM = int(svd_dim)


def build_docs(table: pd.DataFrame, messages: pd.DataFrame) -> pd.Series:
    grouped = (
        messages.sort_values([ID_COLUMN, ORDER_COLUMN])
        .groupby(ID_COLUMN)[TEXT_COLUMN]
        .apply(lambda s: "\n\n".join(str(t) for t in s))
    )
    wanted = pd.Index(table[ID_COLUMN].astype(str).tolist(), name=ID_COLUMN)
    return grouped.reindex(wanted).fillna("")


def make_tabular_preprocessor() -> ColumnTransformer:
    transformers = []
    if NUMERIC_FEATURES:
        transformers.append(("num", StandardScaler(), list(NUMERIC_FEATURES)))
    if CATEGORICAL_FEATURES:
        transformers.append((
            "cat",
            OneHotEncoder(drop="first", sparse_output=False),
            list(CATEGORICAL_FEATURES),
        ))
    if not transformers:
        raise ValueError("Pass numeric features, categorical features, or both.")
    return ColumnTransformer(transformers=transformers)


def encode_texts(texts, idx_train, idx_test):
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, stop_words="english")),
            ("svd", TruncatedSVD(n_components=min(SVD_DIM, max(2, len(idx_train) - 1)), random_state=RANDOM_STATE)),
        ]
    )
    Xtr = pipe.fit_transform([texts[i] for i in idx_train])
    Xte = pipe.transform([texts[i] for i in idx_test])
    names = [f"text_svd_{i}" for i in range(Xtr.shape[1])]
    return np.asarray(Xtr, float), np.asarray(Xte, float), names, pipe


def evaluate(y_true, y_prob, y_pred, label: str) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "model": label,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    print(f"\n=== {label} ===")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))
    print("[[TN FP],[FN TP]]\n", cm)
    print("ROC-AUC", round(metrics["roc_auc"], 4))
    return metrics


def text_type_xai(table, messages, y, idx_train, idx_test, y_train, y_test, X_tab_train, X_tab_test):
    """Quantify which text types carry predictive signal, when a type column is set."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    types = []
    if TYPE_COLUMN and TYPE_COLUMN in messages.columns:
        types = sorted(messages[TYPE_COLUMN].dropna().unique())
    rows = []

    clf0 = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    clf0.fit(X_tab_train, y_train)
    base_auc = roc_auc_score(y_test, clf0.predict_proba(X_tab_test)[:, 1])

    for t in types:
        sub = messages[messages[TYPE_COLUMN] == t]
        docs = (
            sub.sort_values([ID_COLUMN, ORDER_COLUMN])
            .groupby(ID_COLUMN)[TEXT_COLUMN]
            .apply(lambda s: "\n\n".join(map(str, s)))
            .reindex(pd.Index(table[ID_COLUMN].astype(str).tolist(), name=ID_COLUMN))
            .fillna("")
        )
        texts = docs.tolist()
        n_nonempty = sum(1 for i in idx_train if texts[i].strip())
        if n_nonempty < 10:
            continue
        try:
            Xtr, Xte, _, _ = encode_texts(texts, idx_train, idx_test)
        except Exception:
            continue
        Xtr_b = np.hstack([X_tab_train, Xtr])
        Xte_b = np.hstack([X_tab_test, Xte])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        clf.fit(Xtr_b, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(Xte_b)[:, 1])
        rows.append({"text_type": t, "auc_with_type": auc, "auc_lift_vs_tabular": auc - base_auc, "n_texts": int(len(sub))})

    type_df = pd.DataFrame(rows)
    if not type_df.empty:
        type_df = type_df.sort_values("auc_lift_vs_tabular", ascending=False)
        type_df.to_csv(RESULTS_DIR / "text_type_predictive_strength.csv", index=False)
        plt.figure(figsize=(8, 4))
        plt.barh(type_df["text_type"], type_df["auc_lift_vs_tabular"])
        plt.axvline(0, color="k", lw=1)
        plt.xlabel("AUC lift vs tabular-only when adding this type")
        plt.title("Predictive strength of text types")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "text_type_auc_lift.png", dpi=150)
        plt.close()

    full_docs = build_docs(table, messages).tolist()
    Xtr_full, Xte_full, _, _ = encode_texts(full_docs, idx_train, idx_test)
    clf_full = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    clf_full.fit(np.hstack([X_tab_train, Xtr_full]), y_train)
    auc_full = roc_auc_score(y_test, clf_full.predict_proba(np.hstack([X_tab_test, Xte_full]))[:, 1])

    loo_rows = []
    for t in types:
        reduced = messages[messages[TYPE_COLUMN] != t]
        docs = build_docs(table, reduced).tolist()
        try:
            Xtr, Xte, _, _ = encode_texts(docs, idx_train, idx_test)
        except Exception:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        clf.fit(np.hstack([X_tab_train, Xtr]), y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(np.hstack([X_tab_test, Xte]))[:, 1])
        loo_rows.append({"removed_type": t, "auc_without_type": auc, "auc_drop": auc_full - auc})
    loo_df = pd.DataFrame(loo_rows)
    if not loo_df.empty:
        loo_df = loo_df.sort_values("auc_drop", ascending=False)
        loo_df.to_csv(RESULTS_DIR / "text_type_leave_one_out.csv", index=False)
        plt.figure(figsize=(8, 4))
        plt.barh(loo_df["removed_type"], loo_df["auc_drop"])
        plt.xlabel("AUC drop when removing this type from the corpus")
        plt.title("Leave-one-text-type-out")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "text_type_leave_one_out.png", dpi=150)
        plt.close()

    msg_rows = []
    train_docs = [full_docs[i] for i in idx_train]
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2, stop_words="english")),
            ("svd", TruncatedSVD(n_components=min(32, max(2, len(idx_train) - 1)), random_state=RANDOM_STATE)),
        ]
    )
    Xtr_txt = pipe.fit_transform(train_docs)
    clf_txt = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    clf_txt.fit(Xtr_txt, y_train)

    sample = messages.sample(n=min(400, len(messages)), random_state=RANDOM_STATE)
    for _, row in sample.iterrows():
        score = float(clf_txt.predict_proba(pipe.transform([str(row[TEXT_COLUMN])]))[:, 1])
        msg_rows.append(
            {
                ID_COLUMN: row[ID_COLUMN],
                "text_type": row[TYPE_COLUMN] if TYPE_COLUMN and TYPE_COLUMN in messages.columns else "",
                "positive_score": score,
                "text_excerpt": str(row[TEXT_COLUMN])[:280].replace("\n", " "),
            }
        )
    msg_df = pd.DataFrame(msg_rows).sort_values("positive_score", ascending=False)
    msg_df.head(25).to_csv(RESULTS_DIR / "high_positive_signal_messages.csv", index=False)
    msg_df.tail(25).to_csv(RESULTS_DIR / "low_positive_signal_messages.csv", index=False)

    if TYPE_COLUMN and TYPE_COLUMN in messages.columns and not msg_df.empty:
        means = msg_df.groupby("text_type")["positive_score"].mean().sort_values()
        plt.figure(figsize=(8, 4))
        plt.barh(means.index.astype(str), means.values)
        plt.xlabel("Mean text-only positive score for sampled messages")
        plt.title("Average score by text type")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "text_type_mean_positive_score.png", dpi=150)
        plt.close()

    return type_df, loo_df

def explain_global_local(clf, X_train, X_test, y_test, feature_names):
    coef = np.abs(clf.coef_.ravel())
    pd.DataFrame({"feature": feature_names, "abs_coefficient": coef}).sort_values(
        "abs_coefficient", ascending=False
    ).head(40).to_csv(RESULTS_DIR / "global_coefficients_top.csv", index=False)

    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X_test), size=min(400, len(X_test)), replace=False)
    perm = permutation_importance(
        clf, X_test[sample_idx], y_test[sample_idx], n_repeats=6, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    perm_df = pd.DataFrame(
        {"feature": feature_names, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
    ).sort_values("importance_mean", ascending=False)
    perm_df.head(40).to_csv(RESULTS_DIR / "permutation_importance_top.csv", index=False)
    top = perm_df.head(15).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Permutation importance (ROC-AUC drop)")
    plt.title("Global feature importance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "permutation_importance.png", dpi=150)
    plt.close()

    # Highlight text block importance vs tabular
    text_mask = np.array([n.startswith("text_svd_") for n in feature_names])
    tab_imp = float(perm_df.loc[~np.isin(perm_df["feature"], np.array(feature_names)[text_mask]), "importance_mean"].clip(lower=0).sum()) if (~text_mask).any() else 0.0
    # simpler:
    tab_sum = float(np.sum(np.maximum(perm.importances_mean[~text_mask], 0))) if text_mask.any() else float(np.sum(np.maximum(perm.importances_mean, 0)))
    txt_sum = float(np.sum(np.maximum(perm.importances_mean[text_mask], 0))) if text_mask.any() else 0.0
    pd.DataFrame([{"block": "tabular", "perm_importance_sum": tab_sum}, {"block": "text_svd", "perm_importance_sum": txt_sum}]).to_csv(
        RESULTS_DIR / "block_importance_tabular_vs_text.csv", index=False
    )
    plt.figure(figsize=(5, 3))
    plt.bar(["tabular", "text"], [tab_sum, txt_sum])
    plt.ylabel("Sum of positive permutation importances")
    plt.title("Where does the model look? Tabular vs text")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "block_importance_tabular_vs_text.png", dpi=150)
    plt.close()

    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=["negative", "positive"],
        discretize_continuous=True, random_state=RANDOM_STATE,
    )
    idxs = []
    if (y_test == 1).any():
        idxs.append(int(np.where(y_test == 1)[0][0]))
    if (y_test == 0).any():
        idxs.append(int(np.where(y_test == 0)[0][0]))
    lime_rows = []
    for i in idxs:
        exp = explainer.explain_instance(X_test[i], clf.predict_proba, num_features=12, labels=(1,))
        exp.save_to_file(str(RESULTS_DIR / f"lime_instance_{i}.html"))
        for feat, weight in exp.as_list(label=1):
            lime_rows.append({"instance": i, "y_true": int(y_test[i]), "rule": feat, "weight": weight})
    pd.DataFrame(lime_rows).to_csv(RESULTS_DIR / "lime_examples.csv", index=False)


def run(
    tabular_file,
    text_file,
    id_column: str,
    target_column: str,
    positive_label: str,
    numeric_features: list[str],
    categorical_features: list[str],
    text_column: str = "text",
    order_column: str = "text_index",
    type_column: str = "",
    results_dir: str | Path = "data/processed/text_model_results",
    random_state: int = 42,
    test_size: float = 0.25,
    svd_dim: int | None = None,
) -> pd.DataFrame:
    """Fit a tabular model and a tabular-plus-text model on one stratified split."""
    configure(
        id_column=id_column,
        target_column=target_column,
        positive_label=positive_label,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        text_column=text_column,
        order_column=order_column,
        type_column=type_column,
        results_dir=results_dir,
        random_state=random_state,
        test_size=test_size,
        svd_dim=svd_dim,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    messages = pd.read_csv(text_file, dtype={ID_COLUMN: str})
    table = pd.read_csv(tabular_file, dtype={ID_COLUMN: str})
    for column in NUMERIC_FEATURES:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)

    ids = set(messages[ID_COLUMN].astype(str))
    table = table[table[ID_COLUMN].astype(str).isin(ids)].reset_index(drop=True)
    print(f"Rows in modeling set: {len(table)}")

    y = (table[TARGET_COLUMN].astype(str).str.strip() == str(POSITIVE_LABEL)).astype(int).to_numpy()
    docs = build_docs(table, messages).tolist()
    feature_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    X_tab = table[feature_columns].copy()

    idx = np.arange(len(table))
    idx_train, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    y_train, y_test = y[idx_train], y[idx_test]

    pre = make_tabular_preprocessor()
    X_tab_train = pre.fit_transform(X_tab.iloc[idx_train])
    X_tab_test = pre.transform(X_tab.iloc[idx_test])
    tab_names = []
    if NUMERIC_FEATURES:
        tab_names.extend(list(NUMERIC_FEATURES))
    if CATEGORICAL_FEATURES:
        tab_names.extend(list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)))

    clf_a = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    clf_a.fit(X_tab_train, y_train)
    prob_a = clf_a.predict_proba(X_tab_test)[:, 1]
    metrics_a = evaluate(y_test, prob_a, (prob_a >= 0.5).astype(int), "tabular_only")

    Xtr_txt, Xte_txt, text_names, _ = encode_texts(docs, idx_train, idx_test)
    X_train_b = np.hstack([X_tab_train, Xtr_txt])
    X_test_b = np.hstack([X_tab_test, Xte_txt])
    clf_b = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    clf_b.fit(X_train_b, y_train)
    prob_b = clf_b.predict_proba(X_test_b)[:, 1]
    metrics_b = evaluate(y_test, prob_b, (prob_b >= 0.5).astype(int), "tabular_plus_text")

    plt.figure(figsize=(6, 5))
    for prob, label in [(prob_a, "Tabular only"), (prob_b, "Tabular + text")]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_test, prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close()

    comparison = pd.DataFrame([metrics_a, metrics_b])
    comparison.to_csv(RESULTS_DIR / "metrics_comparison.csv", index=False)
    summary = {
        "n_rows": int(len(table)),
        "n_texts": int(len(messages)),
        "tabular_file": str(tabular_file),
        "text_file": str(text_file),
        "metrics": [metrics_a, metrics_b],
        "auc_lift_text": float(metrics_b["roc_auc"] - metrics_a["roc_auc"]),
        "f1_lift_text": float(metrics_b["f1"] - metrics_a["f1"]),
        "text_encoding": {"backend": "tfidf_svd", "dim": len(text_names)},
    }
    Path(RESULTS_DIR / "metrics_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nText-type analysis…")
    text_type_xai(table, messages, y, idx_train, idx_test, y_train, y_test, X_tab_train, X_tab_test)
    print("Global and local analysis…")
    explain_global_local(clf_b, X_train_b, X_test_b, y_test, tab_names + text_names)

    print("\nSummary")
    print(comparison.to_string(index=False))
    print(f"AUC lift (text): {summary['auc_lift_text']:+.4f}")
    return comparison


def main() -> None:
    """Read column roles from the environment and call run().

    Required: TABULAR_FILE, TEXT_FILE, ID_COLUMN, TARGET_COLUMN, POSITIVE_LABEL,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES.
    Feature lists are comma-separated column names.
    Optional: TEXT_COLUMN, TEXT_ORDER_COLUMN, TEXT_TYPE_COLUMN, RESULTS_DIR,
    RANDOM_STATE, TEST_SIZE, SVD_DIM.
    """
    required = [
        "TABULAR_FILE",
        "TEXT_FILE",
        "ID_COLUMN",
        "TARGET_COLUMN",
        "POSITIVE_LABEL",
        "NUMERIC_FEATURES",
        "CATEGORICAL_FEATURES",
    ]
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise SystemExit(
            "Set these environment variables before running this script: "
            + ", ".join(missing)
            + ". NUMERIC_FEATURES and CATEGORICAL_FEATURES are comma-separated column names."
        )

    def split_list(name: str) -> list[str]:
        return [item.strip() for item in os.environ[name].split(",") if item.strip()]

    run(
        tabular_file=os.environ["TABULAR_FILE"],
        text_file=os.environ["TEXT_FILE"],
        id_column=os.environ["ID_COLUMN"],
        target_column=os.environ["TARGET_COLUMN"],
        positive_label=os.environ["POSITIVE_LABEL"],
        numeric_features=split_list("NUMERIC_FEATURES"),
        categorical_features=split_list("CATEGORICAL_FEATURES"),
        text_column=os.environ.get("TEXT_COLUMN", "text").strip() or "text",
        order_column=os.environ.get("TEXT_ORDER_COLUMN", "text_index").strip() or "text_index",
        type_column=os.environ.get("TEXT_TYPE_COLUMN", "").strip(),
        results_dir=os.environ.get("RESULTS_DIR", "data/processed/text_model_results").strip(),
        random_state=int(os.environ.get("RANDOM_STATE", "42")),
        test_size=float(os.environ.get("TEST_SIZE", "0.25")),
        svd_dim=int(os.environ["SVD_DIM"]) if os.environ.get("SVD_DIM", "").strip() else None,
    )


if __name__ == "__main__":
    main()

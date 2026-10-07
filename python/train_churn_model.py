"""
Churn classification with tabular + text (early fusion).

Default text corpus: customer_communications_llm.csv if present and non-empty,
else customer_communications.csv. Restricts to customers present in the corpus.

Also writes text-focused XAI:
  - predictive strength by communication type
  - leave-one-type-out AUC drop
  - example high-influence messages (complaints/support)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PYTHON_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from telco_paths import COMMS_FILE, COMMS_LLM_FILE, PROFILES_FILE, TELCO_RESULTS  # noqa: E402
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

COMMS_LLM = COMMS_LLM_FILE
COMMS_FALLBACK = COMMS_FILE
RESULTS_DIR = TELCO_RESULTS
RANDOM_STATE = 42
TEST_SIZE = 0.25
SVD_DIM = int(os.environ.get("SVD_DIM", "64"))

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def load_comms() -> pd.DataFrame:
    """Prefer full template corpus unless USE_LLM_COMMS=1."""
    use_llm = os.environ.get("USE_LLM_COMMS", "").strip() in {"1", "true", "True", "yes"}
    if use_llm and Path(COMMS_LLM).exists() and Path(COMMS_LLM).stat().st_size > 0:
        df = pd.read_csv(COMMS_LLM)
        if len(df) > 0:
            print(f"Using LLM communications: {COMMS_LLM} ({len(df)} texts)")
            return df
    print(f"Using communications: {COMMS_FALLBACK}")
    return pd.read_csv(COMMS_FALLBACK)


def build_docs(profiles: pd.DataFrame, comms: pd.DataFrame) -> pd.Series:
    grouped = (
        comms.sort_values(["customerID", "text_index"])
        .groupby("customerID")["text"]
        .apply(lambda s: "\n\n".join(str(t) for t in s))
    )
    return grouped.reindex(profiles["customerID"]).fillna("")


def make_tabular_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


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
    print(classification_report(y_true, y_pred, target_names=["No churn", "Churn"]))
    print("[[TN FP],[FN TP]]\n", cm)
    print("ROC-AUC", round(metrics["roc_auc"], 4))
    return metrics


def text_type_xai(profiles, comms, y, idx_train, idx_test, y_train, y_test, X_tab_train, X_tab_test):
    """Quantify which communication types carry predictive signal."""
    RESULTS_DIR.mkdir(exist_ok=True)
    types = sorted(comms["communication_type"].dropna().unique())
    rows = []

    # Baseline tabular AUC
    clf0 = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    clf0.fit(X_tab_train, y_train)
    base_auc = roc_auc_score(y_test, clf0.predict_proba(X_tab_test)[:, 1])

    for t in types:
        sub = comms[comms["communication_type"] == t]
        docs = (
            sub.sort_values(["customerID", "text_index"])
            .groupby("customerID")["text"]
            .apply(lambda s: "\n\n".join(map(str, s)))
            .reindex(profiles["customerID"])
            .fillna("")
        )
        texts = docs.tolist()
        # skip if too few non-empty in train
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
        rows.append({"communication_type": t, "auc_with_type": auc, "auc_lift_vs_tabular": auc - base_auc, "n_texts": int(len(sub))})

    type_df = pd.DataFrame(rows).sort_values("auc_lift_vs_tabular", ascending=False)
    type_df.to_csv(RESULTS_DIR / "text_type_predictive_strength.csv", index=False)

    if not type_df.empty:
        plt.figure(figsize=(8, 4))
        plt.barh(type_df["communication_type"], type_df["auc_lift_vs_tabular"])
        plt.axvline(0, color="k", lw=1)
        plt.xlabel("AUC lift vs tabular-only when adding this type")
        plt.title("Predictive strength of communication types")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "text_type_auc_lift.png", dpi=150)
        plt.close()

    # Leave-one-type-out on full multi-type docs
    full_docs = build_docs(profiles, comms).tolist()
    Xtr_full, Xte_full, _, _ = encode_texts(full_docs, idx_train, idx_test)
    clf_full = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    clf_full.fit(np.hstack([X_tab_train, Xtr_full]), y_train)
    auc_full = roc_auc_score(y_test, clf_full.predict_proba(np.hstack([X_tab_test, Xte_full]))[:, 1])

    loo_rows = []
    for t in types:
        reduced = comms[comms["communication_type"] != t]
        docs = build_docs(profiles, reduced).tolist()
        try:
            Xtr, Xte, _, _ = encode_texts(docs, idx_train, idx_test)
        except Exception:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        clf.fit(np.hstack([X_tab_train, Xtr]), y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(np.hstack([X_tab_test, Xte]))[:, 1])
        loo_rows.append({"removed_type": t, "auc_without_type": auc, "auc_drop": auc_full - auc})
    loo_df = pd.DataFrame(loo_rows).sort_values("auc_drop", ascending=False)
    loo_df.to_csv(RESULTS_DIR / "text_type_leave_one_out.csv", index=False)

    if not loo_df.empty:
        plt.figure(figsize=(8, 4))
        plt.barh(loo_df["removed_type"], loo_df["auc_drop"])
        plt.xlabel("AUC drop when removing this type from the corpus")
        plt.title("Leave-one-communication-type-out")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "text_type_leave_one_out.png", dpi=150)
        plt.close()

    # Example messages: among churners, show complaints/support with highest model score contribution
    # Score each single message with a text-only classifier fit on train docs
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

    sample = comms.sample(n=min(400, len(comms)), random_state=RANDOM_STATE)
    for r in sample.itertuples(index=False):
        x = pipe.transform([str(r.text)])
        p = float(clf_txt.predict_proba(x)[:, 1])
        msg_rows.append(
            {
                "customerID": r.customerID,
                "communication_type": r.communication_type,
                "p_churn_text_only": p,
                "text_excerpt": str(r.text)[:280].replace("\n", " "),
            }
        )
    msg_df = pd.DataFrame(msg_rows).sort_values("p_churn_text_only", ascending=False)
    msg_df.head(25).to_csv(RESULTS_DIR / "high_churn_signal_messages.csv", index=False)
    msg_df.tail(25).to_csv(RESULTS_DIR / "low_churn_signal_messages.csv", index=False)

    # Scatter: type vs mean p
    means = msg_df.groupby("communication_type")["p_churn_text_only"].mean().sort_values()
    plt.figure(figsize=(8, 4))
    plt.barh(means.index.astype(str), means.values)
    plt.xlabel("Mean text-only P(churn) for sampled messages")
    plt.title("Average churn signal by communication type")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "text_type_mean_p_churn.png", dpi=150)
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
        X_train, feature_names=feature_names, class_names=["No churn", "Churn"],
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


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    comms = load_comms()
    profiles = pd.read_csv(PROFILES_FILE)
    profiles["TotalCharges"] = pd.to_numeric(profiles["TotalCharges"], errors="coerce").fillna(0.0)
    profiles["SeniorCitizen"] = profiles["SeniorCitizen"].astype(int)

    # Keep customers that have communications
    ids = set(comms["customerID"])
    profiles = profiles[profiles["customerID"].isin(ids)].reset_index(drop=True)
    print(f"Customers in modeling set: {len(profiles)}")

    y = (profiles["Churn"].astype(str).str.strip() == "Yes").astype(int).to_numpy()
    docs = build_docs(profiles, comms).tolist()
    X_tab = profiles[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    idx = np.arange(len(profiles))
    idx_train, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    y_train, y_test = y[idx_train], y[idx_test]

    pre = make_tabular_preprocessor()
    X_tab_train = pre.fit_transform(X_tab.iloc[idx_train])
    X_tab_test = pre.transform(X_tab.iloc[idx_test])
    cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    tab_names = NUMERIC_FEATURES + cat_names

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
    plt.title("ROC — churn classification")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close()

    comparison = pd.DataFrame([metrics_a, metrics_b])
    comparison.to_csv(RESULTS_DIR / "metrics_comparison.csv", index=False)
    summary = {
        "n_customers": int(len(profiles)),
        "n_texts": int(len(comms)),
        "comms_file": COMMS_LLM if Path(COMMS_LLM).exists() else COMMS_FALLBACK,
        "metrics": [metrics_a, metrics_b],
        "auc_lift_text": float(metrics_b["roc_auc"] - metrics_a["roc_auc"]),
        "f1_lift_text": float(metrics_b["f1"] - metrics_a["f1"]),
        "text_encoding": {"backend": "tfidf_svd", "dim": len(text_names)},
    }
    Path(RESULTS_DIR / "metrics_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nText-type XAI…")
    text_type_xai(profiles, comms, y, idx_train, idx_test, y_train, y_test, X_tab_train, X_tab_test)
    print("Global/local XAI…")
    explain_global_local(clf_b, X_train_b, X_test_b, y_test, tab_names + text_names)

    print("\nSummary")
    print(comparison.to_string(index=False))
    print(f"AUC lift (text): {summary['auc_lift_text']:+.4f}")


if __name__ == "__main__":
    main()

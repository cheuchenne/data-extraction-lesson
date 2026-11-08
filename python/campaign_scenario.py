"""Build the course example for the targeting workshop.

The participant document does not import this file. It reads the two CSV
files this module writes.

``test_probabilities_with_text.csv`` holds the logistic-regression scores on
the test customers. Those customers were not used to fit the model. The
model is the structured-fields-plus-text logistic regression from the
previous workshop (seed 42, test share 0.25, 64 text columns).

The customer file has no retention campaign, so it cannot identify the
probability of churn after a campaign. This module therefore builds a
synthetic campaign probability

    p1_i = p0_i * (1 - r_i)

and writes it to ``campaign_example_population.csv``.

Construction of r_i, seed 42
----------------------------
Draw u_i independently from Uniform(0, 1) with ``numpy`` Generator seed 42.
Let m_i be 1 when the customer's contract is month-to-month, and 0 otherwise.
Let s_i be 1 when the customer is recorded as a senior citizen, and 0
otherwise. Then

    r_i = clip(0.05 + 0.32 * u_i + 0.08 * m_i + 0.03 * s_i, 0.05, 0.55).

r_i does not use the observed churn label. A customer who stayed and a
customer who left can receive the same r_i. p1_i is not set to zero for
customers who did not churn. The draw is not a constant, so campaign
effectiveness differs across customers. These probabilities are an
illustrative scenario. They are not estimated causal effects.

Illustrative money values
-------------------------
The loss if the customer churns is twelve times the current monthly charge:
one year of that charge, treated as the amount at stake. It is not an
estimated lifetime value.

The contact cost starts at 30, plus 15 when the customer is a senior
citizen. A further reproducible 12 percent of customers, drawn with Generator
seed 43, are assigned a contact cost of 180. Costs differ so that the budget
problem is not a ranking by churn probability, and not automatically a
ranking by expected net benefit.

If the checks at the end of this file fail, change these scenario
parameters. Do not change the observed churn labels and do not refit the
logistic regression to force the checks to pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import python.workshop_churn as workshop
PROCESSED = ROOT / "data" / "processed" / "telco"
PROBABILITY_FILE = PROCESSED / "test_probabilities_with_text.csv"
POPULATION_FILE = PROCESSED / "campaign_example_population.csv"
INPUTS_FILE = PROCESSED / "campaign_scenario_inputs.csv"
META_FILE = PROCESSED / "test_probabilities_with_text.meta.json"

SCENARIO_SEED = 42
COST_SEED = 43

TABULAR_FILE = "data/raw/telco/telco_customers.csv"
TEXT_FILE = "data/raw/telco/customer_communications.csv"


class ScenarioError(Exception):
    """The illustrative campaign scenario failed an internal check."""


def main() -> None:
    if not INPUTS_FILE.exists() or not PROBABILITY_FILE.exists():
        fit_and_save_probabilities()
    else:
        print(f"Using the saved test scores in {PROBABILITY_FILE.name}.")
    build_population()


def fit_and_save_probabilities() -> None:
    """Fit the previous workshop's combined model once and store its test scores.

    The fit uses the training rows only. The saved scores are the test rows.
    """
    workshop.ID_COLUMN = "customerID"
    workshop.TARGET_COLUMN = "Churn"
    workshop.POSITIVE_LABEL = "Yes"
    workshop.TEXT_COLUMN = "text"
    workshop.TEXT_ORDER_COLUMN = "text_index"
    workshop.TEXT_TYPE_COLUMN = "communication_type"
    workshop.NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
    workshop.CATEGORICAL_FEATURES = [
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
    workshop.RANDOM_STATE = 42
    workshop.TEST_SIZE = 0.25
    workshop.N_TEXT_COMPONENTS = 64
    workshop.TEXT_MODE = "RUN"
    workshop.DECISION_THRESHOLD = 0.5

    table, messages = workshop.load_datasets(TABULAR_FILE, TEXT_FILE)
    split = workshop.prepare_train_test(table)
    text_features = workshop.build_text_features(split, messages)
    combined = workshop.combine_with_text(split, text_features)
    fitted = workshop.run_combined_model(combined)

    test = split["test"]
    identifiers = test[workshop.ID_COLUMN].astype(str).tolist()
    if identifiers != [str(value) for value in fitted["test_customer_ids"]]:
        raise ScenarioError("The test rows and the predicted probabilities are not in the same order.")
    probabilities = np.asarray(fitted["prob_test"], dtype=float)
    observed = np.asarray(fitted["y_test"], dtype=int)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ScenarioError("The test probabilities are not all inside 0 to 1.")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    probability_frame = pd.DataFrame({
        "customer_id": identifiers,
        "p0": probabilities,
        "observed_outcome": observed,
    })
    probability_frame.to_csv(PROBABILITY_FILE, index=False)
    inputs = pd.DataFrame({
        "customer_id": identifiers,
        "p0": probabilities,
        "observed_outcome": observed,
        "monthly_charges": pd.to_numeric(test["MonthlyCharges"], errors="raise").to_numpy(dtype=float),
        "senior_citizen": pd.to_numeric(test["SeniorCitizen"], errors="raise").to_numpy(dtype=float),
        "month_to_month": (test["Contract"].astype(str) == "Month-to-month").to_numpy(dtype=float),
    })
    inputs.to_csv(INPUTS_FILE, index=False)
    auc = float(fitted["test_metrics"]["roc_auc"])
    META_FILE.write_text(json.dumps({
        "source_model": "logistic regression on structured fields plus text columns",
        "fit_uses_training_rows_only": True,
        "random_state": 42,
        "test_size": 0.25,
        "n_text_components": 64,
        "class_weight": "balanced",
        "n_test": int(len(probability_frame)),
        "test_roc_auc": auc,
        "probability_file": PROBABILITY_FILE.name,
        "p0": "predicted probability of churn on the test customers, if there is no campaign",
        "p1": "synthetic campaign probability written by campaign_scenario.py; not an estimated causal effect",
        "observed_outcome": "historical churn indicator, 1 for the positive label; not an input to the targeting rule",
    }, indent=2))
    print(f"Saved {len(probability_frame)} test probabilities. Test ROC-AUC {auc:.3f}.")


def build_population() -> pd.DataFrame:
    """Attach a reproducible synthetic campaign probability and illustrative costs."""
    inputs = pd.read_csv(INPUTS_FILE, dtype={"customer_id": str})
    stored = pd.read_csv(PROBABILITY_FILE, dtype={"customer_id": str})
    if inputs["customer_id"].tolist() != stored["customer_id"].tolist():
        raise ScenarioError("The scenario inputs and the probability file are not the same customers.")
    if not np.allclose(inputs["p0"].to_numpy(dtype=float), stored["p0"].to_numpy(dtype=float)):
        raise ScenarioError("p0 in the scenario inputs does not match the probability file.")

    p0 = inputs["p0"].to_numpy(dtype=float)
    effectiveness = _effectiveness(
        len(inputs),
        inputs["month_to_month"].to_numpy(dtype=float),
        inputs["senior_citizen"].to_numpy(dtype=float),
    )
    p1 = p0 * (1.0 - effectiveness)
    loss, cost = _economics(
        inputs["monthly_charges"].to_numpy(dtype=float),
        inputs["senior_citizen"].to_numpy(dtype=float),
    )
    _check_scenario(p0, p1, cost, loss, effectiveness)

    population = pd.DataFrame({
        "customer_id": inputs["customer_id"],
        "p0": p0,
        "p1": p1,
        "observed_outcome": inputs["observed_outcome"].to_numpy(dtype=int),
        "example_loss": loss,
        "example_cost": cost,
    })
    population.to_csv(POPULATION_FILE, index=False)
    benefit = loss * (p0 - p1) - cost
    targeted = int(np.sum(benefit > 0))
    print(
        f"Saved {POPULATION_FILE.name}: {len(population)} customers, "
        f"{targeted} with a positive expected net benefit."
    )
    return population


def _effectiveness(n: int, month_to_month: np.ndarray, senior: np.ndarray) -> np.ndarray:
    generator = np.random.default_rng(SCENARIO_SEED)
    draw = generator.uniform(0.0, 1.0, size=n)
    rate = 0.05 + 0.32 * draw + 0.08 * month_to_month + 0.03 * senior
    return np.clip(rate, 0.05, 0.55)


def _economics(monthly_charges: np.ndarray, senior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    loss = np.asarray(monthly_charges, dtype=float) * 12.0
    generator = np.random.default_rng(COST_SEED)
    cost = np.full(len(monthly_charges), 30.0) + 15.0 * np.asarray(senior, dtype=float)
    expensive = generator.random(len(cost)) < 0.12
    cost[expensive] = 180.0
    return loss, cost


def _check_scenario(p0, p1, cost, loss, effectiveness) -> None:
    problems = []
    if not (np.all(p0 >= 0) and np.all(p0 <= 1) and np.all(p1 >= 0) and np.all(p1 <= p0 + 1e-12)):
        problems.append("The scenario must satisfy 0 <= p1 <= p0 <= 1.")
    if np.any(effectiveness < 0) or np.any(effectiveness > 1):
        problems.append("Campaign effectiveness must stay inside 0 to 1.")
    if np.unique(np.round(effectiveness, 6)).size < 2:
        problems.append("Campaign effectiveness must vary across customers.")
    benefit = loss * (p0 - p1) - cost
    n_positive = int(np.sum(benefit > 0))
    n_negative = int(np.sum(benefit < 0))
    n = len(benefit)
    if n_positive == 0 or n_negative == 0:
        problems.append("The scenario needs both positive and negative expected net benefits.")
    if n_positive < 0.05 * n or n_positive > 0.90 * n:
        problems.append(
            f"The unconstrained plan targets {n_positive} of {n} customers. "
            "Adjust the scenario parameters so the plan is not nearly everyone or nearly no one."
        )
    # A pure cutoff on p0 would put every targeted customer above every skipped one.
    targeted = benefit > 0
    if targeted.any() and (~targeted).any():
        if np.min(p0[targeted]) >= np.max(p0[~targeted]) - 1e-12:
            problems.append("The plan collapsed to a cutoff on p0. Costs, losses or effectiveness need more variation.")
    if problems:
        raise ScenarioError(" ".join(problems))


if __name__ == "__main__":
    main()

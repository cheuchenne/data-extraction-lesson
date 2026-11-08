"""Choose whom to target when each customer has a cost and a churn loss.

The decision minimises expected economic cost. A probability of churn is an
input, not something this file estimates. Paths and column names are chosen
by the caller. This file does not name a particular dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[1]

# RUN solves the budget problem. SAVE stores that solution. LOAD reads it
# back and refuses a file that was built from different inputs.
OPTIMIZATION_MODE = "RUN"
CACHE_DIR = ROOT / "data" / "processed" / "targeting_cache"

NAVY = "#1f4e79"
CLAY = "#b85c38"
GREEN = "#2f6f4e"
# A benefit this close to zero is rounding, not a reason to spend money.
BENEFIT_TOLERANCE = 1e-8


class TargetingError(Exception):
    """The inputs cannot be turned into a targeting decision."""


def allocate(p0, p1, cost, loss, ids=None) -> dict:
    """Target a customer when the expected loss avoided exceeds the contact cost.

    The customers are decided one by one. That is the exact solution when
    there is no budget, because the total expected cost is the sum of the
    customer expected costs.
    """
    prepared = _prepare(p0, p1, cost, loss, ids)
    target = _positive_benefit(prepared["net_benefit"]).astype(int)
    return _pack(prepared, target, rule="unconstrained")


def allocate_frame(frame, p0, p1, cost, loss, id_column=None) -> dict:
    """Same decision as ``allocate``, reading inputs from a table.

    ``p0``, ``p1``, ``cost`` and ``loss`` are column names, or a single
    number used for every row. ``id_column`` is optional.
    """
    columns = _columns_from_frame(frame, p0, p1, cost, loss, id_column)
    return allocate(
        columns["p0"], columns["p1"], columns["cost"], columns["loss"], ids=columns["ids"]
    )


def allocate_under_equal_cost(p0, p1, cost, loss, budget, ids=None) -> dict:
    """Target at most K customers when every contact has the same cost.

    K is the number of contacts the budget can pay for. Among customers with
    a positive expected net benefit, take the K largest. A non-positive
    benefit is never selected: the contact would raise expected cost.
    """
    prepared = _prepare(p0, p1, cost, loss, ids)
    _check_budget_value(budget)
    unit_cost = _require_equal_cost(prepared["cost"])
    target = _top_k_target(prepared["net_benefit"], unit_cost, float(budget))
    return _pack(prepared, target, rule="equal_cost_budget", budget=float(budget))


def allocate_under_budget(p0, p1, cost, loss, budget, ids=None, mode=None) -> dict:
    """Minimise expected cost when total contact spending cannot exceed the budget.

    Campaign costs may differ across customers. The solution is an exact
    binary optimisation. Ranking customers by expected net benefit and
    taking them until the money runs out is not the same problem.

    ``mode`` is ``"RUN"``, ``"SAVE"`` or ``"LOAD"``. The default is
    ``OPTIMIZATION_MODE``. ``LOAD`` requires a saved solution built from
    these same inputs and this same budget.
    """
    prepared = _prepare(p0, p1, cost, loss, ids)
    _check_budget_value(budget)
    budget = float(budget)
    chosen_mode = _mode(mode)
    fingerprint = _fingerprint(prepared, budget)
    if chosen_mode == "LOAD":
        target = _load_target(fingerprint, len(prepared["p0"]))
        print(f"Loaded the budget solution ({len(target)} customers).")
    else:
        target = _knapsack(prepared["net_benefit"], prepared["cost"], budget)
        if chosen_mode == "SAVE":
            _save_target(fingerprint, target, budget)
            print(f"Saved the budget solution ({len(target)} customers).")
    return _pack(prepared, target, rule="budget", budget=budget)


def allocate_by_benefit_rank(p0, p1, cost, loss, budget, ids=None) -> dict:
    """Spend the budget on the largest net benefits that still fit.

    This ranking is the solution when every contact costs the same. It can
    leave a better affordable set unused when contact costs differ.
    """
    prepared = _prepare(p0, p1, cost, loss, ids)
    _check_budget_value(budget)
    target = _greedy_benefit(prepared["net_benefit"], prepared["cost"], float(budget))
    return _pack(prepared, target, rule="benefit_rank", budget=float(budget))


def campaign_probabilities(p0, effectiveness):
    """Campaign churn probabilities when every customer has the same effectiveness.

    ``effectiveness`` is a number r between 0 and 1. The campaign probability
    is ``p0 * (1 - r)``. This is an assumption the caller supplies. It is not
    an estimate.
    """
    baseline = _as_p0(p0)
    _check_probability(baseline, "p0")
    rate = _require_effectiveness(effectiveness)
    return baseline * (1.0 - rate)


def probability_threshold(cost, loss, effectiveness) -> float:
    """Cutoff on the no-campaign probability when cost, loss and effectiveness are common.

    Under those three restrictions the individual rule becomes: target
    customer i when ``p0_i`` is above ``cost / (effectiveness * loss)``.
    """
    contact = _require_number(cost, "cost")
    churn_loss = _require_number(loss, "loss")
    rate = _require_effectiveness(effectiveness)
    if contact < 0 or churn_loss < 0:
        raise TargetingError("cost and loss must be zero or positive.")
    if rate == 0 or churn_loss == 0:
        raise TargetingError(
            "A probability cutoff needs a positive campaign effectiveness and a positive loss. "
            "With either one equal to zero, a contact that costs money is never worth making."
        )
    return contact / (rate * churn_loss)


def effectiveness_sensitivity(p0, cost, loss, effectiveness, budget=None, equal_cost=None) -> pd.DataFrame:
    """How the allocation changes when the common campaign effectiveness changes.

    Each value is applied as ``p1 = p0 * (1 - r)``, with the same r for every
    customer. ``budget`` adds the capped allocation on the same rows. When
    every contact cost is equal, the cap uses the K-customer rule.
    """
    baseline = _as_p0(p0)
    rates = [float(value) for value in effectiveness]
    if not rates:
        raise TargetingError("Provide at least one campaign effectiveness.")
    rows = []
    previous = None
    for rate in rates:
        campaign = campaign_probabilities(baseline, rate)
        free = allocate(baseline, campaign, cost, loss)
        row = {
            "effectiveness": rate,
            "customers_targeted": free["n_targeted"],
            "share_targeted": free["share_targeted"],
            "campaign_spend": free["campaign_spend"],
            "expected_cost": free["expected_cost"],
            "benefit_vs_no_campaign": free["benefit"],
        }
        contact = np.asarray(free["cost"], dtype=float)
        churn_loss = np.asarray(free["loss"], dtype=float)
        if np.unique(contact).size == 1 and np.unique(churn_loss).size == 1 and rate > 0 and churn_loss[0] > 0:
            row["threshold"] = probability_threshold(float(contact[0]), float(churn_loss[0]), rate)
        else:
            row["threshold"] = np.nan
        if budget is not None:
            use_equal = _costs_are_equal(contact) if equal_cost is None else bool(equal_cost)
            if use_equal:
                capped = allocate_under_equal_cost(baseline, campaign, cost, loss, budget)
            else:
                capped = allocate_under_budget(baseline, campaign, cost, loss, budget, mode="RUN")
            row["budget_customers_targeted"] = capped["n_targeted"]
            row["budget_campaign_spend"] = capped["campaign_spend"]
            row["budget_expected_cost"] = capped["expected_cost"]
            row["budget_benefit_vs_no_campaign"] = capped["benefit"]
        changed = 0 if previous is None else int(np.sum(free["target"] != previous))
        row["decisions_changed"] = changed
        previous = free["target"]
        rows.append(row)
    return pd.DataFrame(rows)


def customer_table(result: dict) -> pd.DataFrame:
    """One row per customer, with the quantities that enter the decision."""
    frame = pd.DataFrame({
        "p0": result["p0"],
        "p1": result["p1"],
        "cost": result["cost"],
        "loss": result["loss"],
        "expected_cost_without": result["loss"] * result["p0"],
        "expected_cost_with_campaign": result["cost"] + result["loss"] * result["p1"],
        "net_benefit": result["net_benefit"],
        "target": result["target"].astype(int),
    })
    frame.insert(0, "row", np.arange(1, len(frame) + 1))
    if result["ids"] is not None:
        frame.insert(1, "id", result["ids"])
    return frame


def selected_customers(result: dict, limit: int = 8) -> pd.DataFrame:
    """Customers the plan targets, highest expected net benefit first."""
    table = customer_table(result)
    chosen = table.loc[table["target"] == 1].sort_values(
        ["net_benefit", "row"], ascending=[False, True]
    )
    if limit is not None:
        chosen = chosen.head(int(limit))
    return chosen.reset_index(drop=True)


def compare(results: dict) -> pd.DataFrame:
    """Put several plans side by side. Keys are the plan names."""
    rows = []
    for name, result in results.items():
        within = ""
        budget_text = ""
        if result.get("budget") is not None:
            budget_text = result["budget"]
            within = "Yes" if result["campaign_spend"] <= result["budget"] + 1e-6 else "No"
        row = {
            "Plan": name,
            "Customers targeted": result["n_targeted"],
            "Share targeted": result["share_targeted"],
            "Campaign spend": result["campaign_spend"],
            "Expected cost": result["expected_cost"],
            "Benefit vs no campaign": result["benefit"],
            "Budget": budget_text,
            "Within budget": within,
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame["Budget"].eq("").all():
        frame = frame.drop(columns=["Budget", "Within budget"])
    return frame


def threshold_counterexample(result: dict) -> pd.DataFrame:
    """Two customers for whom a cutoff on p0 would make the other choice.

    Returns an empty table when the plan happens to match some cutoff on p0.
    """
    p0 = np.asarray(result["p0"], dtype=float)
    target = np.asarray(result["target"], dtype=int)
    chosen = np.flatnonzero(target == 1)
    skipped = np.flatnonzero(target == 0)
    if len(chosen) == 0 or len(skipped) == 0:
        return customer_table(result).iloc[0:0]
    # The targeted customer with the smallest p0, if some skipped customer is higher.
    order = np.argsort(p0[chosen], kind="mergesort")
    for position in order:
        i = int(chosen[position])
        higher = skipped[p0[skipped] > p0[i] + 1e-12]
        if len(higher):
            j = int(higher[np.argmax(p0[higher])])
            table = customer_table(result)
            return table.iloc[[i, j]].reset_index(drop=True)
    return customer_table(result).iloc[0:0]


def read_scores(path) -> pd.DataFrame:
    """Load scores written by ``save_scores`` in the previous workshop.

    The file must contain ``customer_id``, ``p0`` and ``observed_outcome``.
    ``p0`` is the score. ``observed_outcome`` is 1 or 0. The path is the
    caller's: point it at the folder where those scores were saved.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise TargetingError(
            f"The score file {file_path} does not exist. "
            "Write it from the previous workshop with save_scores, "
            "using this same path."
        )
    frame = pd.read_csv(file_path, dtype={"customer_id": str})
    required = ["customer_id", "p0", "observed_outcome"]
    missing = [name for name in required if name not in frame.columns]
    if missing or frame.empty:
        missing_text = ", ".join(missing) if missing else "rows"
        raise TargetingError(
            f"{file_path} has no {missing_text}. "
            "save_scores writes customer_id, p0 and observed_outcome."
        )
    scores = frame["p0"].to_numpy(dtype=float)
    observed = frame["observed_outcome"].to_numpy(dtype=float)
    _check_missing(scores, "p0")
    _check_missing(observed, "observed outcome")
    _check_finite(scores, "p0")
    _check_probability(scores, "p0")
    if not np.isin(observed, [0.0, 1.0]).all():
        raise TargetingError("observed_outcome must be 0 or 1 for every customer.")
    return frame


def read_population(path) -> pd.DataFrame:
    """Read a customer table from a CSV path supplied by the caller."""
    file_path = Path(path)
    if not file_path.exists():
        raise TargetingError(
            f"The file {file_path} does not exist. "
            "The course example is built by python/campaign_scenario.py."
        )
    frame = pd.read_csv(file_path)
    if "customer_id" in frame.columns:
        frame["customer_id"] = frame["customer_id"].astype(str)
    if frame.empty:
        raise TargetingError(f"{file_path} has no rows.")
    return frame


def plot_expected_costs(result: dict):
    """Total expected cost with no campaign, and with the plan."""
    labels = ["No campaign", "Optimal targeting"]
    values = [result["expected_cost_without"], result["expected_cost"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, values, color=[CLAY, NAVY], width=0.62)
    ceiling = max(values) if max(values) > 0 else 1.0
    ax.set_ylim(0, ceiling * 1.18)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel("Total expected cost")
    ax.set_title("Expected cost with and without the campaign")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_net_benefits(result: dict):
    """Expected net benefit of every customer, ordered from low to high."""
    benefit = np.asarray(result["net_benefit"], dtype=float)
    target = np.asarray(result["target"], dtype=int)
    order = np.argsort(benefit, kind="mergesort")
    ordered_benefit = benefit[order]
    ordered_target = target[order]
    positions = np.arange(len(benefit))
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    kept = ordered_target == 0
    taken = ordered_target == 1
    ax.scatter(positions[kept], ordered_benefit[kept], s=10, color=CLAY, label="Do not target", linewidths=0)
    ax.scatter(positions[taken], ordered_benefit[taken], s=10, color=NAVY, label="Target", linewidths=0)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Customers, ordered by expected net benefit")
    ax.set_ylabel("Expected net benefit")
    ax.set_title("Target a customer when the expected net benefit is positive")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_allocations(unconstrained: dict, constrained: dict):
    """Benefit, spending, and which unconstrained targets the budget keeps."""
    _require_same_customers(unconstrained, constrained)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.3))

    names = ["No budget", "With budget"]
    benefits = [unconstrained["benefit"], constrained["benefit"]]
    spending = [unconstrained["campaign_spend"], constrained["campaign_spend"]]
    x = np.arange(2)
    width = 0.36
    axes[0].bar(x - width / 2, benefits, width=width, color=NAVY, label="Expected net benefit")
    axes[0].bar(x + width / 2, spending, width=width, color=GREEN, label="Campaign spend")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Amount")
    axes[0].set_title("The budget changes the total")
    axes[0].legend(frameon=False, fontsize=8)

    benefit = np.asarray(unconstrained["net_benefit"], dtype=float)
    free_target = np.asarray(unconstrained["target"], dtype=int) == 1
    capped_target = np.asarray(constrained["target"], dtype=int) == 1
    shown = free_target | capped_target
    positions = np.arange(int(shown.sum()))
    ordered = np.argsort(benefit[shown], kind="mergesort")
    shown_benefit = benefit[shown][ordered]
    both = capped_target[shown][ordered] & free_target[shown][ordered]
    dropped = free_target[shown][ordered] & ~capped_target[shown][ordered]
    added = capped_target[shown][ordered] & ~free_target[shown][ordered]
    axes[1].scatter(positions[dropped], shown_benefit[dropped], s=12, color=CLAY, label="Dropped by the budget", linewidths=0)
    axes[1].scatter(positions[both], shown_benefit[both], s=12, color=NAVY, label="Kept under the budget", linewidths=0)
    if added.any():
        axes[1].scatter(positions[added], shown_benefit[added], s=12, color=GREEN, label="Added under the budget", linewidths=0)
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_xlabel("Selected customers, ordered by expected net benefit")
    axes[1].set_ylabel("Expected net benefit")
    axes[1].set_title("Who the budget keeps")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_effectiveness(sensitivity: pd.DataFrame):
    """Allocation and expected benefit as campaign effectiveness changes."""
    frame = sensitivity.sort_values("effectiveness")
    has_budget = "budget_customers_targeted" in frame.columns
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.3))
    axes[0].plot(
        frame["effectiveness"],
        frame["customers_targeted"],
        color=NAVY,
        marker="o",
        linewidth=2,
        label="No budget",
    )
    axes[1].plot(
        frame["effectiveness"],
        frame["benefit_vs_no_campaign"],
        color=NAVY,
        marker="o",
        linewidth=2,
        label="No budget",
    )
    if has_budget:
        axes[0].plot(
            frame["effectiveness"],
            frame["budget_customers_targeted"],
            color=CLAY,
            marker="o",
            linewidth=2,
            label="With budget",
        )
        axes[1].plot(
            frame["effectiveness"],
            frame["budget_benefit_vs_no_campaign"],
            color=CLAY,
            marker="o",
            linewidth=2,
            label="With budget",
        )
    axes[0].set_xlabel("Campaign effectiveness")
    axes[0].set_ylabel("Customers targeted")
    axes[0].set_title("Who is targeted")
    axes[1].set_xlabel("Campaign effectiveness")
    axes[1].set_ylabel("Expected net benefit")
    axes[1].set_title("Value of the plan")
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _prepare(p0, p1, cost, loss, ids) -> dict:
    baseline = _as_p0(p0)
    n = len(baseline)
    campaign = _as_vector(p1, n, "p1")
    contact = _as_vector(cost, n, "cost")
    churn_loss = _as_vector(loss, n, "loss")
    _check_missing(baseline, "p0")
    _check_missing(campaign, "p1")
    _check_missing(contact, "cost")
    _check_missing(churn_loss, "loss")
    _check_finite(baseline, "p0")
    _check_finite(campaign, "p1")
    _check_finite(contact, "cost")
    _check_finite(churn_loss, "loss")
    _check_probability(baseline, "p0")
    _check_probability(campaign, "p1")
    _check_nonnegative(contact, "cost")
    _check_nonnegative(churn_loss, "loss")
    benefit = churn_loss * (baseline - campaign) - contact
    return {
        "p0": baseline,
        "p1": campaign,
        "cost": contact,
        "loss": churn_loss,
        "net_benefit": benefit,
        "ids": _ids(ids, n),
    }


def _pack(prepared: dict, target: np.ndarray, rule: str, budget: float | None = None) -> dict:
    target = np.asarray(target, dtype=int)
    if target.shape != prepared["p0"].shape or not np.isin(target, [0, 1]).all():
        raise TargetingError("The targeting decision must be 0 or 1 for every customer.")
    without = prepared["loss"] * prepared["p0"]
    chosen_cost = (
        prepared["cost"] * target
        + prepared["loss"] * ((1 - target) * prepared["p0"] + target * prepared["p1"])
    )
    expected_without = float(np.sum(without))
    expected_cost = float(np.sum(chosen_cost))
    benefit = float(np.sum(prepared["net_benefit"] * target))
    if abs((expected_without - expected_cost) - benefit) > 1e-4 * max(1.0, abs(benefit)):
        raise TargetingError("The expected-benefit identity failed for this plan.")
    spend = float(np.sum(prepared["cost"] * target))
    if budget is not None and spend > float(budget) + 1e-6:
        raise TargetingError(
            f"The plan spends {spend:.4f}, which is above the budget of {float(budget):.4f}."
        )
    n = int(len(target))
    n_targeted = int(target.sum())
    return {
        "target": target,
        "p0": prepared["p0"],
        "p1": prepared["p1"],
        "cost": prepared["cost"],
        "loss": prepared["loss"],
        "net_benefit": prepared["net_benefit"],
        "ids": prepared["ids"],
        "expected_cost_without": expected_without,
        "expected_cost": expected_cost,
        "campaign_spend": spend,
        "benefit": benefit,
        "n_targeted": n_targeted,
        "share_targeted": n_targeted / n,
        "n_customers": n,
        "rule": rule,
        "budget": None if budget is None else float(budget),
    }


def _columns_from_frame(frame, p0, p1, cost, loss, id_column) -> dict:
    if not isinstance(frame, pd.DataFrame):
        raise TargetingError("Pass a table with one row per customer.")
    if frame.empty:
        raise TargetingError("The table has no customers.")

    def _values(spec, name):
        if isinstance(spec, str):
            if spec not in frame.columns:
                raise TargetingError(f"The table has no column {spec!r} for {name}.")
            return frame[spec].to_numpy()
        return spec

    ids = None
    if id_column is not None:
        if id_column not in frame.columns:
            raise TargetingError(f"The table has no identifier column {id_column!r}.")
        ids = frame[id_column].tolist()
    return {
        "p0": _values(p0, "p0"),
        "p1": _values(p1, "p1"),
        "cost": _values(cost, "cost"),
        "loss": _values(loss, "loss"),
        "ids": ids,
    }


def _as_p0(p0) -> np.ndarray:
    if isinstance(p0, (str, bytes)) or np.isscalar(p0):
        raise TargetingError("p0 must list one probability per customer.")
    array = np.asarray(p0, dtype=float)
    if array.ndim == 0:
        raise TargetingError("p0 must list one probability per customer.")
    array = array.reshape(-1)
    if len(array) == 0:
        raise TargetingError("Provide at least one customer.")
    return array


def _as_vector(value, n: int, name: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise TargetingError(f"{name} must be a number or one number per customer.")
    if np.isscalar(value):
        number = float(value)
        if not np.isfinite(number):
            raise TargetingError(f"{name} is not a finite number.")
        return np.full(n, number, dtype=float)
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        number = float(array)
        if not np.isfinite(number):
            raise TargetingError(f"{name} is not a finite number.")
        return np.full(n, number, dtype=float)
    array = array.reshape(-1)
    if array.shape != (n,):
        raise TargetingError(
            f"{name} has length {len(array)} and p0 has length {n}. "
            "Use one value for every customer, or one value per customer."
        )
    return array


def _ids(ids, n: int):
    if ids is None:
        return None
    values = [str(value) for value in list(ids)]
    if len(values) != n:
        raise TargetingError(
            f"The identifier has length {len(values)} and p0 has length {n}."
        )
    return values


def _check_missing(array: np.ndarray, name: str) -> None:
    missing = int(np.isnan(array).sum())
    if missing:
        noun = "value" if missing == 1 else "values"
        raise TargetingError(
            f"{name} has {missing} missing {noun}. The calculation does not fill them in."
        )


def _check_finite(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        raise TargetingError(f"{name} contains a value that is not finite.")


def _check_probability(array: np.ndarray, name: str) -> None:
    bad = (array < 0) | (array > 1)
    if bad.any():
        raise TargetingError(
            f"{name} must be between 0 and 1. Found {float(array[bad][0]):.4g}."
        )


def _check_nonnegative(array: np.ndarray, name: str) -> None:
    bad = array < 0
    if bad.any():
        raise TargetingError(
            f"{name} must be zero or positive. Found {float(array[bad][0]):.4g}."
        )


def _check_budget_value(budget) -> None:
    number = _require_number(budget, "budget")
    if number < 0:
        raise TargetingError("The budget must be zero or positive.")


def _require_number(value, name: str) -> float:
    if isinstance(value, (str, bytes)):
        raise TargetingError(f"{name} must be a single number for this calculation.")
    array = np.asarray(value, dtype=float)
    if array.ndim > 0:
        array = array.reshape(-1)
        if len(array) != 1:
            raise TargetingError(f"{name} must be a single number for this calculation.")
        value = array[0]
    number = float(value)
    if not np.isfinite(number):
        raise TargetingError(f"{name} is not a finite number.")
    return number


def _require_effectiveness(effectiveness) -> float:
    rate = _require_number(effectiveness, "effectiveness")
    if rate < 0 or rate > 1:
        raise TargetingError("Campaign effectiveness must be between 0 and 1.")
    return rate


def _require_equal_cost(cost: np.ndarray) -> float:
    if not _costs_are_equal(cost):
        raise TargetingError(
            "This rule requires the same campaign cost for every customer. "
            "Use allocate_under_budget when costs differ."
        )
    return float(cost[0])


def _costs_are_equal(cost: np.ndarray) -> bool:
    return bool(np.max(cost) - np.min(cost) <= 1e-8)


def _top_k_target(benefit: np.ndarray, unit_cost: float, budget: float) -> np.ndarray:
    target = np.zeros(len(benefit), dtype=int)
    positive = np.flatnonzero(_positive_benefit(benefit))
    if len(positive) == 0:
        return target
    if unit_cost == 0:
        target[positive] = 1
        return target
    capacity = int(np.floor((budget + 1e-9) / unit_cost))
    order = positive[np.argsort(-benefit[positive], kind="mergesort")]
    target[order[:capacity]] = 1
    return target


def _greedy_benefit(benefit: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    target = np.zeros(len(benefit), dtype=int)
    spend = 0.0
    for index in np.argsort(-benefit, kind="mergesort"):
        if benefit[index] <= BENEFIT_TOLERANCE:
            continue
        next_spend = spend + float(cost[index])
        if next_spend <= budget + 1e-9:
            target[index] = 1
            spend = next_spend
    return target


def _knapsack(benefit: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    """Exact solution of max benefit, subject to the budget and binary choices.

    A customer with a non-positive expected net benefit is left out. Selecting
    that customer cannot reduce expected cost when the contact cost is
    nonnegative.
    """
    target = np.zeros(len(benefit), dtype=int)
    worth = np.flatnonzero(_positive_benefit(benefit))
    if len(worth) == 0:
        return target
    worth_cost = cost[worth]
    free = worth[worth_cost <= 1e-12]
    target[free] = 1
    paid = worth[worth_cost > 1e-12]
    if len(paid) == 0:
        return target
    paid_cost = cost[paid]
    if float(np.sum(paid_cost)) <= budget + 1e-8:
        target[paid] = 1
        return target
    outcome = milp(
        c=-benefit[paid],
        integrality=np.ones(len(paid)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(paid_cost.reshape(1, -1), -np.inf, budget),
        options={"time_limit": 60, "disp": False},
    )
    if not outcome.success or outcome.x is None:
        message = outcome.message if outcome.message else "no optimum was returned"
        raise TargetingError(
            "The budget solver did not return a proven optimum "
            f"({message}). For a large file, save a finished run and load it."
        )
    if np.max(np.minimum(np.abs(outcome.x), np.abs(outcome.x - 1))) > 1e-6:
        raise TargetingError("The budget solver did not return a binary decision.")
    target[paid] = np.rint(outcome.x).astype(int)
    return target


def _positive_benefit(benefit: np.ndarray) -> np.ndarray:
    return np.asarray(benefit, dtype=float) > BENEFIT_TOLERANCE


def _require_same_customers(left: dict, right: dict) -> None:
    if len(left["target"]) != len(right["target"]):
        raise TargetingError("The two plans are for different numbers of customers.")
    if left["ids"] is not None and right["ids"] is not None and left["ids"] != right["ids"]:
        raise TargetingError("The two plans are not in the same customer order.")


def _mode(mode) -> str:
    chosen = OPTIMIZATION_MODE if mode is None else mode
    chosen = str(chosen).strip().upper()
    if chosen not in {"RUN", "SAVE", "LOAD"}:
        raise TargetingError('OPTIMIZATION_MODE must be "RUN", "SAVE" or "LOAD".')
    return chosen


def _fingerprint(prepared: dict, budget: float) -> str:
    payload = {
        "p0": np.round(prepared["p0"], 8).tolist(),
        "p1": np.round(prepared["p1"], 8).tolist(),
        "cost": np.round(prepared["cost"], 8).tolist(),
        "loss": np.round(prepared["loss"], 8).tolist(),
        "budget": round(float(budget), 8),
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cache_meta_path() -> Path:
    return CACHE_DIR / "budget_allocation.meta.json"


def _cache_target_path() -> Path:
    return CACHE_DIR / "budget_allocation.csv"


def _save_target(fingerprint: str, target: np.ndarray, budget: float) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"target": target.astype(int)}).to_csv(_cache_target_path(), index=False)
    _cache_meta_path().write_text(json.dumps({
        "fingerprint": fingerprint,
        "n_customers": int(len(target)),
        "budget": float(budget),
        "n_targeted": int(target.sum()),
    }, indent=2))


def _load_target(fingerprint: str, n_customers: int) -> np.ndarray:
    meta_path = _cache_meta_path()
    target_path = _cache_target_path()
    if not meta_path.exists() or not target_path.exists():
        raise TargetingError(
            "No saved budget solution is available. "
            'Set OPTIMIZATION_MODE to "SAVE" to solve the problem and store it, '
            'or to "RUN" to solve it now.'
        )
    meta = json.loads(meta_path.read_text())
    if meta.get("fingerprint") != fingerprint or int(meta.get("n_customers", -1)) != n_customers:
        raise TargetingError(
            "The saved budget solution was built from different inputs or a different budget. "
            'Set OPTIMIZATION_MODE to "SAVE" to replace it, or to "RUN" to solve this problem now.'
        )
    loaded = pd.read_csv(target_path)
    if "target" not in loaded.columns or len(loaded) != n_customers:
        raise TargetingError(
            "The saved budget solution cannot be read. "
            'Set OPTIMIZATION_MODE to "SAVE" to create it again.'
        )
    return loaded["target"].to_numpy(dtype=int)

"""One-customer profile against the HarborLink panel.

The invoice table stays invoice-level. For the profile we first build one
row per customer (means, sums, or the constant attributes), then mark that
customer's value on a small summary plot of every customer.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from python.eda_visualisation import load_invoice_table
from python.harborlink_dictionary import VARIABLES

ROOT = Path(__file__).resolve().parents[1]

CANCELLED = {
    "invoice_id",
    "order_id",
    "invoice_date",
    "due_date",
    "status",
    "period",
    "payment_id",
    "payment_date",
    "payment_status",
    "tenure_months",
}

# Constant (or almost constant) attributes: keep the customer's own value.
KEEP_AS_IS = [
    "customer_id",
    "customer_name",
    "company_name",
    "customer_segment",
    "country",
    "signup_date",
    "subscription_type",
    "start_date",
]

# Invoice-level numbers reduced to one value per customer.
MEAN_COLUMNS = [
    "amount",
    "monthly_amount",
    "days_to_due",
    "paid_after_due",
    "is_outstanding",
]
SUM_COLUMNS = [
    "n_interactions",
    "n_reminders",
    "n_complaints",
    "n_payment_issues",
    "n_emails",
]
MEAN_NONEMPTY = ["days_to_pay"]

MAX_BAR_LEVELS = 12
PAID_COLOR = "#2C6EBC"
UNPAID_COLOR = "#C0392B"
MARKER_COLOR = "#1A1A1A"


def _prepare_invoices(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for column in ["invoice_date", "due_date", "payment_date", "signup_date", "start_date"]:
        if column in work.columns:
            work[column] = pd.to_datetime(work[column], errors="coerce")
    for column in ["paid_after_due", "is_outstanding"]:
        if column in work.columns:
            work[column] = work[column].astype(float)
    return work


def customer_rows(frame: pd.DataFrame, customer) -> pd.DataFrame:
    """Rows of one customer, selected by customer_id or customer_name."""
    if customer is None or (isinstance(customer, str) and not str(customer).strip()):
        raise ValueError("Pass a customer_id or a customer_name.")

    if isinstance(customer, (int, np.integer)) or (
        isinstance(customer, str) and str(customer).strip().isdigit()
    ):
        customer_id = int(customer)
        rows = frame.loc[frame["customer_id"] == customer_id]
        if rows.empty:
            raise KeyError(f"No customer with customer_id = {customer_id}.")
        return rows.sort_values("invoice_date")

    name = str(customer).strip()
    names = frame["customer_name"].astype(str)
    exact = names.str.casefold() == name.casefold()
    rows = frame.loc[exact]
    if rows.empty:
        contains = names.str.contains(name, case=False, regex=False, na=False)
        matched = names.loc[contains].drop_duplicates().tolist()
        if len(matched) == 1:
            rows = frame.loc[names == matched[0]]
        elif len(matched) > 1:
            raise KeyError(
                f"Several customers match {name!r}: {matched[:8]}"
                + ("…" if len(matched) > 8 else "")
            )
        else:
            raise KeyError(f"No customer named {name!r}.")
    return rows.sort_values("invoice_date")


def customer_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per customer, with the reductions used in the profile."""
    work = _prepare_invoices(frame)
    grouped = work.groupby("customer_id", dropna=False)
    parts: dict[str, pd.Series] = {}
    for column in KEEP_AS_IS:
        if column == "customer_id":
            continue
        if column in work.columns:
            parts[column] = grouped[column].first()
    for column in MEAN_COLUMNS:
        if column in work.columns:
            parts[column] = grouped[column].mean()
    for column in SUM_COLUMNS:
        if column in work.columns:
            parts[column] = grouped[column].sum()
    for column in MEAN_NONEMPTY:
        if column in work.columns:
            parts[column] = grouped[column].mean()
    summary = pd.DataFrame(parts)
    summary.index.name = "customer_id"
    return summary.reset_index()


def _format_value(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "missing"
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.2f}"
    text = str(value)
    return text if text.lower() != "nan" else "missing"


def _kind(name: str) -> str:
    return VARIABLES.get(name, {}).get("kind", "unknown")


def _plot_kind(name: str, series: pd.Series) -> str:
    """How to draw the panel of all customers for this variable."""
    kind = _kind(name)
    nunique = series.dropna().nunique()
    if name in KEEP_AS_IS and kind in {"identifier"} and nunique > MAX_BAR_LEVELS:
        return "label"
    if kind == "date" or name in {"signup_date", "start_date"}:
        return "date"
    if kind == "numeric" or name in MEAN_COLUMNS + SUM_COLUMNS + MEAN_NONEMPTY:
        return "hist"
    if nunique > MAX_BAR_LEVELS:
        return "label"
    return "bar"


def _draw_timeline(ax, rows: pd.DataFrame, title: str) -> None:
    dates = pd.to_datetime(rows["invoice_date"], errors="coerce")
    paid = rows["payment_status"].astype(str).str.strip().eq("paid")
    ax.axhline(0.0, color="#B0B0B0", lw=1.2, zorder=1)
    ax.scatter(
        dates.loc[paid],
        np.zeros(int(paid.sum())),
        s=70,
        c=PAID_COLOR,
        zorder=3,
        label="paid",
        edgecolors="white",
        linewidths=0.6,
    )
    ax.scatter(
        dates.loc[~paid],
        np.zeros(int((~paid).sum())),
        s=70,
        c=UNPAID_COLOR,
        zorder=3,
        label="unpaid or overdue",
        edgecolors="white",
        linewidths=0.6,
    )
    ax.set_yticks([])
    ax.set_ylim(-1.15, 1.35)
    ax.set_xlabel("Invoice date")
    ax.set_title(title, loc="left", fontsize=12, pad=8)
    ax.legend(loc="upper left", frameon=True, fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_label_panel(ax, name: str, value) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(name, fontsize=8, pad=2)
    ax.text(
        0.5,
        0.5,
        _format_value(value),
        ha="center",
        va="center",
        fontsize=8,
        color=MARKER_COLOR,
        wrap=True,
        transform=ax.transAxes,
    )


def _mark_on_hist(ax, values: pd.Series, customer_value, bins: int = 12) -> None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    ax.hist(numeric, bins=min(bins, max(5, int(numeric.nunique()))), color=PAID_COLOR, edgecolor="white")
    if customer_value is None or pd.isna(customer_value):
        ax.text(0.5, 0.9, "no value", ha="center", va="top", transform=ax.transAxes, fontsize=7, color=UNPAID_COLOR)
        return
    customer_value = float(customer_value)
    counts, edges = np.histogram(numeric, bins=min(bins, max(5, int(numeric.nunique()))))
    index = int(np.clip(np.searchsorted(edges, customer_value, side="right") - 1, 0, len(counts) - 1))
    height = counts[index] if len(counts) else 1
    ax.scatter(
        [customer_value],
        [max(height, 1) * 0.92],
        s=42,
        c=UNPAID_COLOR,
        zorder=5,
        edgecolors=MARKER_COLOR,
        linewidths=0.6,
    )
    ax.axvline(customer_value, color=UNPAID_COLOR, lw=1.1, ls="--", zorder=4)
    ax.text(
        0.5,
        -0.38,
        _format_value(customer_value),
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=7,
        color=UNPAID_COLOR,
        clip_on=False,
    )


def _mark_on_bars(ax, values: pd.Series, customer_value) -> None:
    labels = values.fillna("(missing)").astype(str)
    counts = labels.value_counts().sort_index()
    this = "(missing)" if customer_value is None or pd.isna(customer_value) else str(customer_value)
    colors = [UNPAID_COLOR if index == this else PAID_COLOR for index in counts.index]
    xpos = np.arange(len(counts))
    ax.bar(xpos, counts.values, color=colors, edgecolor="white")
    ax.set_xticks(xpos)
    ax.set_xticklabels(list(counts.index), rotation=45, ha="right", fontsize=6)
    if this in set(counts.index):
        position = list(counts.index).index(this)
        height = counts.iloc[position]
        ax.scatter(
            [position],
            [height],
            s=36,
            c=MARKER_COLOR,
            zorder=5,
            marker="D",
        )
    ax.text(
        0.5,
        -0.42,
        this,
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=7,
        color=UNPAID_COLOR,
        clip_on=False,
    )


def _mark_on_dates(ax, values: pd.Series, customer_value) -> None:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    numbers = mdates.date2num(dates)
    ax.hist(numbers, bins=min(12, max(5, int(dates.nunique()))), color=PAID_COLOR, edgecolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=6)
    if customer_value is None or pd.isna(customer_value):
        ax.text(0.5, 0.9, "no value", ha="center", va="top", transform=ax.transAxes, fontsize=7, color=UNPAID_COLOR)
        return
    this_date = pd.to_datetime(customer_value)
    this_num = float(mdates.date2num(this_date))
    counts, edges = np.histogram(numbers, bins=min(12, max(5, int(dates.nunique()))))
    index = int(np.clip(np.searchsorted(edges, this_num, side="right") - 1, 0, len(counts) - 1))
    height = counts[index] if len(counts) else 1
    ax.scatter(
        [this_num],
        [max(height, 1) * 0.92],
        s=42,
        c=UNPAID_COLOR,
        zorder=5,
        edgecolors=MARKER_COLOR,
        linewidths=0.6,
    )
    ax.axvline(this_num, color=UNPAID_COLOR, lw=1.1, ls="--", zorder=4)
    ax.text(
        0.5,
        -0.42,
        _format_value(this_date),
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=7,
        color=UNPAID_COLOR,
        clip_on=False,
    )


def _draw_variable_panel(ax, name: str, all_values: pd.Series, customer_value) -> None:
    ax.set_title(name, fontsize=8, pad=3)
    ax.tick_params(axis="y", labelsize=6)
    kind = _plot_kind(name, all_values)
    if kind == "label":
        _draw_label_panel(ax, name, customer_value)
        return
    if kind == "hist":
        _mark_on_hist(ax, all_values, customer_value)
    elif kind == "date":
        _mark_on_dates(ax, all_values, customer_value)
    else:
        _mark_on_bars(ax, all_values, customer_value)
    ax.set_ylabel("")
    ax.set_xlabel("")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_customer_profile(
    customer,
    invoices: pd.DataFrame | None = None,
):
    """Draw the invoice timeline and customer-level panels for one customer.

    `customer` may be a `customer_id` (1827) or a `customer_name`
    ("Elena Vargas"). Each small plot below the timeline is the distribution
    of that variable **across customers**, with this customer marked.
    """
    frame = _prepare_invoices(invoices if invoices is not None else load_invoice_table())
    rows = customer_rows(frame, customer)
    summaries = customer_summary_table(frame)
    customer_id = int(rows["customer_id"].iloc[0])
    profile = summaries.loc[summaries["customer_id"] == customer_id]
    if profile.empty:
        raise KeyError(f"Could not build a summary row for customer_id = {customer_id}.")
    profile = profile.iloc[0]
    display_name = str(profile.get("customer_name", customer_id))

    panel_names = [
        name
        for name in KEEP_AS_IS + MEAN_COLUMNS + SUM_COLUMNS + MEAN_NONEMPTY
        if name in summaries.columns
    ]

    n_panels = len(panel_names)
    n_cols = 7
    n_rows = int(np.ceil(n_panels / n_cols))
    fig = plt.figure(figsize=(16.8, 3.1 + 2.35 * n_rows))
    grid = GridSpec(
        n_rows + 1,
        n_cols,
        figure=fig,
        height_ratios=[1.45] + [1.0] * n_rows,
        hspace=0.72,
        wspace=0.38,
    )
    ax_time = fig.add_subplot(grid[0, :])
    _draw_timeline(
        ax_time,
        rows,
        f"{display_name}  ·  customer_id {customer_id}  ·  {len(rows)} invoices",
    )

    for index, name in enumerate(panel_names):
        row = 1 + index // n_cols
        col = index % n_cols
        ax = fig.add_subplot(grid[row, col])
        _draw_variable_panel(ax, name, summaries[name], profile[name])

    fig.suptitle(
        "Customer profile against the HarborLink panel",
        fontsize=13,
        y=0.995,
    )
    plt.show()
    return fig

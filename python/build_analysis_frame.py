"""Assemble the HarborLink invoice-level analysis table.

The Quarto lesson shows the same joins with pandas directly. This module
is the reproducible version of that pipeline.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "data" / "raw" / "analysis"
PROCESSED_DIR = ROOT / "data" / "processed"
AS_OF = pd.Timestamp("2026-09-12")


def load_customers(path: Path | None = None) -> pd.DataFrame:
    source = path or ANALYSIS_DIR / "customers.csv"
    return pd.read_csv(source, parse_dates=["signup_date"])


def load_invoices(path: Path | None = None) -> pd.DataFrame:
    source = path or ANALYSIS_DIR / "invoices.xlsx"
    return pd.read_excel(
        source,
        sheet_name="invoices",
        parse_dates=["invoice_date", "due_date"],
    )


def load_subscriptions_and_payments(path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = path or ANALYSIS_DIR / "company.sqlite"
    connection = sqlite3.connect(source)
    try:
        subscriptions = pd.read_sql_query("SELECT * FROM subscriptions", connection)
        payments = pd.read_sql_query(
            "SELECT payment_id, invoice_id, payment_date, status AS payment_status FROM payments",
            connection,
            parse_dates=["payment_date"],
        )
    finally:
        connection.close()
    return subscriptions, payments


def load_interactions(path: Path | None = None) -> pd.DataFrame:
    source = path or ANALYSIS_DIR / "interactions.json"
    with source.open(encoding="utf-8") as file:
        documents = json.load(file)
    contacts = pd.json_normalize(
        documents,
        record_path="contacts",
        meta=["customer_id"],
    )
    contacts["date"] = pd.to_datetime(contacts["date"])
    contacts["period"] = contacts["date"].dt.strftime("%Y-%m")
    return contacts


def summarise_interactions(contacts: pd.DataFrame) -> pd.DataFrame:
    """One row per customer and billing period."""
    grouped = contacts.groupby(["customer_id", "period"], as_index=False)
    return grouped.agg(
        n_interactions=("interaction_type", "size"),
        n_reminders=("interaction_type", lambda s: int((s == "reminder").sum())),
        n_complaints=("interaction_type", lambda s: int((s == "complaint").sum())),
        n_payment_issues=(
            "interaction_type",
            lambda s: int((s == "payment_issue").sum()),
        ),
        n_emails=("channel", lambda s: int((s == "email").sum())),
    )


def load_api_snapshot(path: Path | None = None) -> pd.DataFrame:
    source = path or ANALYSIS_DIR / "api_invoices.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    snapshot = pd.DataFrame(payload["invoices"])
    return snapshot[["invoice_id", "status"]].rename(columns={"status": "status_api"})


def add_analysis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["days_to_due"] = (result["due_date"] - result["invoice_date"]).dt.days
    result["days_to_pay"] = (result["payment_date"] - result["invoice_date"]).dt.days
    result["paid_after_due"] = (
        result["payment_date"].notna() & (result["payment_date"] > result["due_date"])
    )
    result["is_outstanding"] = result["payment_status"].isin(["unpaid", "overdue"])
    result["tenure_months"] = (
        (result["invoice_date"].dt.year - result["signup_date"].dt.year) * 12
        + (result["invoice_date"].dt.month - result["signup_date"].dt.month)
    )
    for column in [
        "n_interactions",
        "n_reminders",
        "n_complaints",
        "n_payment_issues",
        "n_emails",
    ]:
        result[column] = result[column].fillna(0).astype(int)
    return result


def build_invoice_analysis_frame() -> pd.DataFrame:
    customers = load_customers()
    invoices = load_invoices()
    subscriptions, payments = load_subscriptions_and_payments()
    contacts = summarise_interactions(load_interactions())
    api_snapshot = load_api_snapshot()

    frame = (
        invoices.merge(customers, on="customer_id", how="left", validate="m:1")
        .merge(
            subscriptions[
                ["subscription_id", "subscription_type", "monthly_amount", "start_date"]
            ],
            on="subscription_id",
            how="left",
            validate="m:1",
        )
        .merge(payments, on="invoice_id", how="left", validate="1:1")
        .merge(contacts, on=["customer_id", "period"], how="left", validate="m:1")
        .merge(api_snapshot, on="invoice_id", how="left", validate="m:1")
    )
    frame = add_analysis_columns(frame)
    return frame.sort_values(["invoice_date", "customer_id"]).reset_index(drop=True)


def write_invoice_analysis_frame(path: Path | None = None) -> Path:
    destination = path or PROCESSED_DIR / "harborlink_invoices.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = build_invoice_analysis_frame()
    frame.to_csv(destination, index=False)
    try:
        from python.harborlink_catalog import forward_invoice_descriptions
    except ImportError:
        from harborlink_catalog import forward_invoice_descriptions

    catalog = forward_invoice_descriptions(ANALYSIS_DIR)
    catalog.to_csv(PROCESSED_DIR / "harborlink_invoices_variables.csv", index=False)
    try:
        from python.data_dictionary import write_variable_workbooks
    except ImportError:
        from data_dictionary import write_variable_workbooks

    write_variable_workbooks(
        frame,
        catalog,
        PROCESSED_DIR / "harborlink_invoices_dictionary.xlsx",
        PROCESSED_DIR / "harborlink_invoices_summary.xlsx",
    )
    return destination


def main() -> None:
    path = write_invoice_analysis_frame()
    frame = pd.read_csv(path)
    print(f"Wrote {path.relative_to(ROOT)} with shape {frame.shape}")


if __name__ == "__main__":
    main()

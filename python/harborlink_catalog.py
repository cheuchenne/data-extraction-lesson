"""Variable descriptions that sit next to each HarborLink extract.

Each source file in data/raw/analysis/ has a companion `*_variables.csv`
written at generation time. The analysis columns created later (CRM counts,
days_to_pay, …) have their own rows, attached at the moment those columns
are computed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CUSTOMERS = [
    {"variable": "customer_id", "type": "identifier", "definition": "Identifier of the customer billed on this invoice."},
    {"variable": "customer_name", "type": "identifier", "definition": "Name of the person who is the HarborLink customer."},
    {"variable": "company_name", "type": "identifier", "definition": "Name of the customer's company."},
    {
        "variable": "customer_segment",
        "type": "factor",
        "definition": (
            "Commercial segment of the customer (smb, mid-market, enterprise). "
            "Empty when the customer file has no segment for that customer."
        ),
    },
    {"variable": "country", "type": "factor", "definition": "Country of the customer, as an ISO two-letter code."},
    {"variable": "signup_date", "type": "date", "definition": "Date the customer first subscribed to HarborLink."},
]

INVOICES = [
    {"variable": "invoice_id", "type": "identifier", "definition": "Unique identifier of one invoice (one billing cycle for one customer)."},
    {"variable": "customer_id", "type": "identifier", "definition": "Identifier of the customer billed on this invoice."},
    {"variable": "subscription_id", "type": "identifier", "definition": "Identifier of that customer's HarborLink subscription."},
    {"variable": "order_id", "type": "identifier", "definition": "Identifier of the billing order behind this invoice."},
    {"variable": "invoice_date", "type": "date", "definition": "Date the invoice was issued."},
    {"variable": "due_date", "type": "date", "definition": "Date by which the invoice should be paid."},
    {"variable": "amount", "type": "numeric", "definition": "Amount billed on this invoice, in euros."},
    {
        "variable": "status",
        "type": "factor",
        "definition": (
            "Payment status stored on the invoice workbook. "
            "It should agree with payment_status; both are kept so a mismatch can be checked."
        ),
    },
    {"variable": "period", "type": "date", "definition": "Billing month of the invoice, written YYYY-MM."},
]

SUBSCRIPTIONS = [
    {"variable": "subscription_id", "type": "identifier", "definition": "Identifier of that customer's HarborLink subscription."},
    {"variable": "customer_id", "type": "identifier", "definition": "Identifier of the customer billed on this invoice."},
    {"variable": "subscription_type", "type": "factor", "definition": "Plan of the subscription: starter, standard or premium."},
    {
        "variable": "monthly_amount",
        "type": "numeric",
        "definition": "Monthly price of the plan, in euros. It should match amount on these invoices.",
    },
    {"variable": "start_date", "type": "date", "definition": "Date this subscription started."},
    {"variable": "status", "type": "factor", "definition": "Whether the subscription is active in the operations table."},
]

PAYMENTS = [
    {"variable": "payment_id", "type": "identifier", "definition": "Identifier of the payment row attached to this invoice."},
    {"variable": "invoice_id", "type": "identifier", "definition": "Unique identifier of one invoice (one billing cycle for one customer)."},
    {"variable": "customer_id", "type": "identifier", "definition": "Identifier of the customer billed on this invoice."},
    {"variable": "amount", "type": "numeric", "definition": "Amount recorded on the payment row, in euros."},
    {
        "variable": "status",
        "type": "factor",
        "definition": (
            "Finance snapshot on 12 September 2026. "
            "paid = a payment_date is recorded (including paid after the due date). "
            "unpaid = no payment yet, and the due date has not passed. "
            "overdue = no payment, and the due date has already passed. "
            "overdue is not 'paid late'."
        ),
    },
    {
        "variable": "payment_date",
        "type": "date",
        "definition": (
            "Date a payment was recorded. Empty when no payment has been recorded "
            "(unpaid or overdue invoices)."
        ),
    },
]

INTERACTIONS = [
    {"variable": "customer_id", "type": "identifier", "definition": "Identifier of the customer billed on this invoice."},
    {"variable": "date", "type": "date", "definition": "Date of one CRM contact."},
    {"variable": "channel", "type": "factor", "definition": "Channel of the CRM contact, for example email."},
    {
        "variable": "interaction_type",
        "type": "factor",
        "definition": "Kind of CRM contact: reminder, complaint, payment_issue, and so on.",
    },
    {"variable": "text", "type": "identifier", "definition": "Free-text body of the CRM contact."},
]

API_INVOICES = [
    {"variable": "invoice_id", "type": "identifier", "definition": "Unique identifier of one invoice (one billing cycle for one customer)."},
    {"variable": "customer_id", "type": "identifier", "definition": "Identifier of the customer billed on this invoice."},
    {"variable": "invoice_date", "type": "date", "definition": "Date the invoice was issued."},
    {"variable": "due_date", "type": "date", "definition": "Date by which the invoice should be paid."},
    {"variable": "amount", "type": "numeric", "definition": "Amount billed on this invoice, in euros."},
    {"variable": "currency", "type": "factor", "definition": "Currency of the API snapshot."},
    {
        "variable": "status",
        "type": "factor",
        "definition": (
            "Payment status returned by the September API snapshot for this invoice_id. "
            "Empty for invoices that are not in that snapshot (every month except September)."
        ),
    },
    {"variable": "order_id", "type": "identifier", "definition": "Identifier of the billing order behind this invoice."},
    {"variable": "period", "type": "date", "definition": "Billing month of the invoice, written YYYY-MM."},
]

CRM_COUNTS = [
    {
        "variable": "n_interactions",
        "type": "numeric",
        "definition": (
            "Number of CRM contacts for this customer during the invoice month. "
            "Zero means no contact that month, not a missing value."
        ),
    },
    {"variable": "n_reminders", "type": "numeric", "definition": "Number of those CRM contacts that were payment reminders."},
    {"variable": "n_complaints", "type": "numeric", "definition": "Number of those CRM contacts that were complaints."},
    {
        "variable": "n_payment_issues",
        "type": "numeric",
        "definition": "Number of those CRM contacts that reported a payment issue.",
    },
    {"variable": "n_emails", "type": "numeric", "definition": "Number of those CRM contacts that went through email."},
]

ANALYSIS_COLUMNS = [
    {
        "variable": "days_to_due",
        "type": "numeric",
        "definition": "Number of days between invoice_date and due_date (the payment window).",
    },
    {
        "variable": "days_to_pay",
        "type": "numeric",
        "definition": (
            "Number of days between invoice_date and payment_date. "
            "Empty when there is no payment_date."
        ),
    },
    {
        "variable": "paid_after_due",
        "type": "factor",
        "definition": (
            "True when a payment_date exists and is after due_date. "
            "False if the invoice was paid on time, or if it was never paid."
        ),
    },
    {
        "variable": "is_outstanding",
        "type": "factor",
        "definition": (
            "True when payment_status is unpaid or overdue. "
            "The mean of this column is the outstanding rate."
        ),
    },
    {
        "variable": "tenure_months",
        "type": "numeric",
        "definition": (
            "Whole months between signup_date and this invoice_date. "
            "Zero in the customer's first billing month."
        ),
    },
]

SOURCE_FILES = {
    "customers_variables.csv": CUSTOMERS,
    "invoices_variables.csv": INVOICES,
    "subscriptions_variables.csv": SUBSCRIPTIONS,
    "payments_variables.csv": PAYMENTS,
    "interactions_variables.csv": INTERACTIONS,
    "api_invoices_variables.csv": API_INVOICES,
}


def _frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["variable", "type", "definition"])


def write_source_description_files(directory: Path) -> list[Path]:
    """Write one `*_variables.csv` next to each generated HarborLink extract."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, rows in SOURCE_FILES.items():
        path = directory / name
        _frame(rows).to_csv(path, index=False)
        written.append(path)
    return written


def crm_count_descriptions() -> pd.DataFrame:
    return _frame(CRM_COUNTS)


def analysis_column_descriptions() -> pd.DataFrame:
    return _frame(ANALYSIS_COLUMNS)


def forward_invoice_descriptions(analysis_dir: Path) -> pd.DataFrame:
    """Copy source descriptions onto the invoice table, with the same renames as the joins."""
    try:
        from python.data_dictionary import attach_descriptions, read_descriptions
    except ImportError:
        from data_dictionary import attach_descriptions, read_descriptions

    catalog = read_descriptions(analysis_dir / "invoices_variables.csv")
    catalog = attach_descriptions(
        catalog,
        analysis_dir / "customers_variables.csv",
        skip=["customer_id"],
    )
    catalog = attach_descriptions(
        catalog,
        analysis_dir / "subscriptions_variables.csv",
        keep=["subscription_type", "monthly_amount", "start_date"],
    )
    catalog = attach_descriptions(
        catalog,
        analysis_dir / "payments_variables.csv",
        rename={"status": "payment_status"},
        keep=["payment_id", "payment_date", "payment_status"],
    )
    catalog = attach_descriptions(catalog, crm_count_descriptions())
    catalog = attach_descriptions(
        catalog,
        analysis_dir / "api_invoices_variables.csv",
        rename={"status": "status_api"},
        keep=["status_api"],
    )
    catalog = attach_descriptions(catalog, analysis_column_descriptions())
    return catalog


def variables_dict() -> dict[str, dict[str, str]]:
    """Flat map used by the customer-profile plots."""
    mapping: dict[str, dict[str, str]] = {}
    for rows in list(SOURCE_FILES.values()) + [CRM_COUNTS, ANALYSIS_COLUMNS]:
        for row in rows:
            mapping[row["variable"]] = {
                "kind": row["type"],
                "definition": row["definition"],
            }
    mapping["payment_status"] = {
        "kind": "factor",
        "definition": next(r["definition"] for r in PAYMENTS if r["variable"] == "status"),
    }
    mapping["status_api"] = {
        "kind": "factor",
        "definition": next(r["definition"] for r in API_INVOICES if r["variable"] == "status"),
    }
    return mapping

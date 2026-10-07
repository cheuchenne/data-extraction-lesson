"""Small loading helpers for the HarborLink lesson.

The Quarto document and the notebook use pandas directly so that students
see the actual commands. These functions are a compact reminder of the
same patterns.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def load_customers_csv(path: Path | str | None = None) -> pd.DataFrame:
    """Load the denormalised customer export."""
    source = Path(path) if path else RAW_DIR / "customers.csv"
    return pd.read_csv(source, parse_dates=["signup_date"])


def load_customers_excel(
    path: Path | str | None = None,
    sheet_name: str = "customers",
) -> pd.DataFrame:
    """Load one worksheet from the HarborLink workbook."""
    source = Path(path) if path else RAW_DIR / "customers.xlsx"
    return pd.read_excel(source, sheet_name=sheet_name)


def load_interactions(path: Path | str | None = None) -> list[dict]:
    """Load nested interaction documents with the standard-library json module."""
    source = Path(path) if path else RAW_DIR / "interactions.json"
    with source.open(encoding="utf-8") as file:
        return json.load(file)


def interactions_to_frame(
    documents: list[dict] | None = None,
    unit: str = "interaction",
) -> pd.DataFrame:
    """Flatten nested JSON according to the chosen analytical unit."""
    payload = documents if documents is not None else load_interactions()
    if unit == "customer":
        return pd.json_normalize(payload)
    if unit == "interaction":
        return pd.json_normalize(
            payload,
            record_path="contacts",
            meta=["customer_id", "company_name", ["subscription", "status"]],
            sep="_",
        )
    raise ValueError("unit must be 'customer' or 'interaction'")


def load_unpaid_payments(path: Path | str | None = None) -> pd.DataFrame:
    """Return unpaid or overdue payments joined with customer names."""
    source = Path(path) if path else RAW_DIR / "company.sqlite"
    query = """
        SELECT
            c.customer_id,
            c.customer_name,
            p.invoice_id,
            p.amount,
            p.status
        FROM payments AS p
        JOIN customers AS c
            ON c.customer_id = p.customer_id
        WHERE p.status IN ('unpaid', 'overdue')
        ORDER BY p.status DESC, c.customer_id
    """
    connection = sqlite3.connect(source)
    try:
        return pd.read_sql_query(query, connection)
    finally:
        connection.close()

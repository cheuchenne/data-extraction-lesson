"""Write the larger HarborLink source files used in Section 2.c.

Run from the project root:

    python python/generate_analysis_data.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PYTHON_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from harborlink_panel import (  # noqa: E402
    api_invoices_payload,
    build_panel,
    customers_export_rows,
    interactions_json,
)
from harborlink_catalog import write_source_description_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "analysis"


def write_sqlite(path: Path, panel: dict) -> None:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute(
            """
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                customer_name TEXT NOT NULL,
                company_name TEXT NOT NULL,
                customer_segment TEXT,
                country TEXT NOT NULL,
                signup_date TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE subscriptions (
                subscription_id TEXT PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                subscription_type TEXT NOT NULL,
                monthly_amount REAL NOT NULL,
                start_date TEXT NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE invoices (
                invoice_id TEXT PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                subscription_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                invoice_date TEXT NOT NULL,
                due_date TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT NOT NULL,
                period TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE payments (
                payment_id TEXT PRIMARY KEY,
                invoice_id TEXT NOT NULL,
                customer_id INTEGER NOT NULL,
                amount REAL NOT NULL,
                status TEXT NOT NULL,
                payment_date TEXT,
                FOREIGN KEY (invoice_id) REFERENCES invoices (invoice_id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE interactions (
                interaction_id TEXT PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                channel TEXT NOT NULL,
                interaction_type TEXT NOT NULL,
                text TEXT NOT NULL,
                period TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
            """
        )
        cursor.executemany(
            """
            INSERT INTO customers VALUES (
                :customer_id, :customer_name, :company_name,
                :customer_segment, :country, :signup_date
            )
            """,
            customers_export_rows(panel["customers"]),
        )
        cursor.executemany(
            """
            INSERT INTO subscriptions VALUES (
                :subscription_id, :customer_id, :subscription_type,
                :monthly_amount, :start_date, :status
            )
            """,
            panel["subscriptions"],
        )
        cursor.executemany(
            """
            INSERT INTO invoices VALUES (
                :invoice_id, :customer_id, :subscription_id, :order_id,
                :invoice_date, :due_date, :amount, :status, :period
            )
            """,
            panel["invoices"],
        )
        cursor.executemany(
            """
            INSERT INTO payments VALUES (
                :payment_id, :invoice_id, :customer_id,
                :amount, :status, :payment_date
            )
            """,
            panel["payments"],
        )
        cursor.executemany(
            """
            INSERT INTO interactions VALUES (
                :interaction_id, :customer_id, :date,
                :channel, :interaction_type, :text, :period
            )
            """,
            panel["interactions"],
        )
        connection.commit()
    finally:
        connection.close()


def generate_all() -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    written = {
        "customers.csv": OUT_DIR / "customers.csv",
        "invoices.xlsx": OUT_DIR / "invoices.xlsx",
        "interactions.json": OUT_DIR / "interactions.json",
        "api_invoices.json": OUT_DIR / "api_invoices.json",
        "company.sqlite": OUT_DIR / "company.sqlite",
    }
    pd.DataFrame(customers_export_rows(panel["customers"])).to_csv(
        written["customers.csv"], index=False
    )
    invoices = pd.DataFrame(panel["invoices"])
    customers = pd.DataFrame(customers_export_rows(panel["customers"]))
    with pd.ExcelWriter(written["invoices.xlsx"], engine="openpyxl") as writer:
        invoices.to_excel(writer, sheet_name="invoices", index=False)
        customers.to_excel(writer, sheet_name="customers", index=False)
    written["interactions.json"].write_text(
        json.dumps(interactions_json(panel), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    written["api_invoices.json"].write_text(
        json.dumps(api_invoices_payload(panel), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_sqlite(written["company.sqlite"], panel)
    # One *_variables.csv per extract, written with the data (not after the join).
    for path in write_source_description_files(OUT_DIR):
        written[path.name] = path
    return written


def main() -> None:
    written = generate_all()
    print("Generated HarborLink analysis sources.")
    for name, path in written.items():
        print(f"  {name:22} {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

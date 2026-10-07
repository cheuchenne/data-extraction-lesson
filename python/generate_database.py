"""Create the HarborLink SQLite database from the canonical tables."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from harborlink import CUSTOMERS, INTERACTIONS, INVOICES, PAYMENTS, SUBSCRIPTIONS


CREATE_STATEMENTS = [
    """
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        customer_name TEXT NOT NULL,
        company_name TEXT NOT NULL,
        customer_segment TEXT NOT NULL,
        country TEXT NOT NULL,
        signup_date TEXT NOT NULL
    )
    """,
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
    """,
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
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
        FOREIGN KEY (subscription_id) REFERENCES subscriptions (subscription_id)
    )
    """,
    """
    CREATE TABLE payments (
        payment_id TEXT PRIMARY KEY,
        invoice_id TEXT NOT NULL,
        customer_id INTEGER NOT NULL,
        amount REAL NOT NULL,
        status TEXT NOT NULL,
        payment_date TEXT,
        FOREIGN KEY (invoice_id) REFERENCES invoices (invoice_id),
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
    )
    """,
    """
    CREATE TABLE interactions (
        interaction_id TEXT PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        channel TEXT NOT NULL,
        interaction_type TEXT NOT NULL,
        text TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
    )
    """,
]


def write_sqlite(path: Path) -> Path:
    """Replace the SQLite file at *path* with the canonical HarborLink tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    connection = sqlite3.connect(path)
    try:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        for statement in CREATE_STATEMENTS:
            cursor.execute(statement)

        cursor.executemany(
            """
            INSERT INTO customers (
                customer_id, customer_name, company_name,
                customer_segment, country, signup_date
            )
            VALUES (
                :customer_id, :customer_name, :company_name,
                :customer_segment, :country, :signup_date
            )
            """,
            CUSTOMERS,
        )
        cursor.executemany(
            """
            INSERT INTO subscriptions (
                subscription_id, customer_id, subscription_type,
                monthly_amount, start_date, status
            )
            VALUES (
                :subscription_id, :customer_id, :subscription_type,
                :monthly_amount, :start_date, :status
            )
            """,
            SUBSCRIPTIONS,
        )
        cursor.executemany(
            """
            INSERT INTO invoices (
                invoice_id, customer_id, subscription_id, order_id,
                invoice_date, due_date, amount, status, period
            )
            VALUES (
                :invoice_id, :customer_id, :subscription_id, :order_id,
                :invoice_date, :due_date, :amount, :status, :period
            )
            """,
            INVOICES,
        )
        cursor.executemany(
            """
            INSERT INTO payments (
                payment_id, invoice_id, customer_id,
                amount, status, payment_date
            )
            VALUES (
                :payment_id, :invoice_id, :customer_id,
                :amount, :status, :payment_date
            )
            """,
            PAYMENTS,
        )
        cursor.executemany(
            """
            INSERT INTO interactions (
                interaction_id, customer_id, date,
                channel, interaction_type, text
            )
            VALUES (
                :interaction_id, :customer_id, :date,
                :channel, :interaction_type, :text
            )
            """,
            INTERACTIONS,
        )
        connection.commit()
    finally:
        connection.close()
    return path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "data" / "raw" / "company.sqlite"
    write_sqlite(destination)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()

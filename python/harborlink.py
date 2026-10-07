"""Canonical HarborLink business data used by every generated file.

All identifiers, amounts, dates and statuses are defined here so that the
CSV export, Excel workbook, JSON interactions, SQLite database, mock API
payloads and synthetic invoices remain consistent.
"""

from __future__ import annotations

COMPANY = {
    "name": "HarborLink",
    "legal_name": "HarborLink BV",
    "tagline": "Customer-communication platform for small and mid-sized businesses",
    "address": "Rue de la Science 14, 1040 Brussels, Belgium",
    "currency": "EUR",
}

# Current snapshot used in the lesson (Saturday 12 September 2026).
AS_OF_DATE = "2026-09-12"

SUBSCRIPTION_PRICES = {
    "starter": 19.90,
    "standard": 29.90,
    "premium": 49.90,
}

CUSTOMERS = [
    {
        "customer_id": 1827,
        "customer_name": "Elena Vargas",
        "company_name": "Atelier Vega",
        "customer_segment": "mid-market",
        "country": "BE",
        "signup_date": "2025-03-12",
    },
    {
        "customer_id": 1841,
        "customer_name": "Marc Dubois",
        "company_name": "Dubois & Fils",
        "customer_segment": "smb",
        "country": "FR",
        "signup_date": "2025-05-04",
    },
    {
        "customer_id": 1903,
        "customer_name": "Priya Shah",
        "company_name": "Shah Consulting",
        "customer_segment": "smb",
        "country": "NL",
        "signup_date": "2025-06-18",
    },
    {
        "customer_id": 1918,
        "customer_name": "Thomas Berg",
        "company_name": "Berg Logistics",
        "customer_segment": "mid-market",
        "country": "DE",
        "signup_date": "2025-01-20",
    },
    {
        "customer_id": 2044,
        "customer_name": "Amelie Laurent",
        "company_name": "Laurent Design",
        "customer_segment": "smb",
        "country": "BE",
        "signup_date": "2025-09-01",
    },
    {
        "customer_id": 2110,
        "customer_name": "Jonas Keller",
        "company_name": "Keller Studio",
        "customer_segment": "smb",
        "country": "DE",
        "signup_date": "2026-01-15",
    },
    {
        "customer_id": 2235,
        "customer_name": "Sofia Rossi",
        "company_name": "Rossi Media",
        "customer_segment": "mid-market",
        "country": "IT",
        "signup_date": "2025-11-08",
    },
    {
        "customer_id": 2301,
        "customer_name": "Liam OConnor",
        "company_name": "Northwind Retail",
        "customer_segment": "smb",
        "country": "IE",
        "signup_date": "2026-02-02",
    },
]

SUBSCRIPTIONS = [
    {
        "subscription_id": "SUB-1827",
        "customer_id": 1827,
        "subscription_type": "premium",
        "monthly_amount": 49.90,
        "start_date": "2025-03-12",
        "status": "active",
    },
    {
        "subscription_id": "SUB-1841",
        "customer_id": 1841,
        "subscription_type": "standard",
        "monthly_amount": 29.90,
        "start_date": "2025-05-04",
        "status": "active",
    },
    {
        "subscription_id": "SUB-1903",
        "customer_id": 1903,
        "subscription_type": "starter",
        "monthly_amount": 19.90,
        "start_date": "2025-06-18",
        "status": "active",
    },
    {
        "subscription_id": "SUB-1918",
        "customer_id": 1918,
        "subscription_type": "premium",
        "monthly_amount": 49.90,
        "start_date": "2025-01-20",
        "status": "active",
    },
    {
        "subscription_id": "SUB-2044",
        "customer_id": 2044,
        "subscription_type": "standard",
        "monthly_amount": 29.90,
        "start_date": "2025-09-01",
        "status": "active",
    },
    {
        "subscription_id": "SUB-2110",
        "customer_id": 2110,
        "subscription_type": "starter",
        "monthly_amount": 19.90,
        "start_date": "2026-01-15",
        "status": "active",
    },
    {
        "subscription_id": "SUB-2235",
        "customer_id": 2235,
        "subscription_type": "premium",
        "monthly_amount": 49.90,
        "start_date": "2025-11-08",
        "status": "active",
    },
    {
        "subscription_id": "SUB-2301",
        "customer_id": 2301,
        "subscription_type": "standard",
        "monthly_amount": 29.90,
        "start_date": "2026-02-02",
        "status": "active",
    },
]

# One August invoice (all paid) and one September invoice per customer.
INVOICES = [
    {
        "invoice_id": "INV-2026-0801",
        "customer_id": 1827,
        "subscription_id": "SUB-1827",
        "order_id": "ORD-2026-1827-08",
        "invoice_date": "2026-08-01",
        "due_date": "2026-08-15",
        "amount": 49.90,
        "status": "paid",
        "period": "August 2026",
    },
    {
        "invoice_id": "INV-2026-0802",
        "customer_id": 1841,
        "subscription_id": "SUB-1841",
        "order_id": "ORD-2026-1841-08",
        "invoice_date": "2026-08-01",
        "due_date": "2026-08-15",
        "amount": 29.90,
        "status": "paid",
        "period": "August 2026",
    },
    {
        "invoice_id": "INV-2026-0803",
        "customer_id": 1903,
        "subscription_id": "SUB-1903",
        "order_id": "ORD-2026-1903-08",
        "invoice_date": "2026-08-01",
        "due_date": "2026-08-15",
        "amount": 19.90,
        "status": "paid",
        "period": "August 2026",
    },
    {
        "invoice_id": "INV-2026-0804",
        "customer_id": 1918,
        "subscription_id": "SUB-1918",
        "order_id": "ORD-2026-1918-08",
        "invoice_date": "2026-08-01",
        "due_date": "2026-08-15",
        "amount": 49.90,
        "status": "paid",
        "period": "August 2026",
    },
    {
        "invoice_id": "INV-2026-0805",
        "customer_id": 2044,
        "subscription_id": "SUB-2044",
        "order_id": "ORD-2026-2044-08",
        "invoice_date": "2026-08-01",
        "due_date": "2026-08-15",
        "amount": 29.90,
        "status": "paid",
        "period": "August 2026",
    },
    {
        "invoice_id": "INV-2026-0806",
        "customer_id": 2110,
        "subscription_id": "SUB-2110",
        "order_id": "ORD-2026-2110-08",
        "invoice_date": "2026-08-01",
        "due_date": "2026-08-15",
        "amount": 19.90,
        "status": "paid",
        "period": "August 2026",
    },
    {
        "invoice_id": "INV-2026-0807",
        "customer_id": 2235,
        "subscription_id": "SUB-2235",
        "order_id": "ORD-2026-2235-08",
        "invoice_date": "2026-08-01",
        "due_date": "2026-08-15",
        "amount": 49.90,
        "status": "paid",
        "period": "August 2026",
    },
    {
        "invoice_id": "INV-2026-0808",
        "customer_id": 2301,
        "subscription_id": "SUB-2301",
        "order_id": "ORD-2026-2301-08",
        "invoice_date": "2026-08-01",
        "due_date": "2026-08-15",
        "amount": 29.90,
        "status": "paid",
        "period": "August 2026",
    },
    {
        "invoice_id": "INV-2026-0901",
        "customer_id": 1827,
        "subscription_id": "SUB-1827",
        "order_id": "ORD-2026-1827-09",
        "invoice_date": "2026-08-28",
        "due_date": "2026-09-15",
        "amount": 49.90,
        "status": "unpaid",
        "period": "September 2026",
    },
    {
        "invoice_id": "INV-2026-0902",
        "customer_id": 1841,
        "subscription_id": "SUB-1841",
        "order_id": "ORD-2026-1841-09",
        "invoice_date": "2026-08-28",
        "due_date": "2026-09-15",
        "amount": 29.90,
        "status": "paid",
        "period": "September 2026",
    },
    {
        "invoice_id": "INV-2026-0903",
        "customer_id": 1903,
        "subscription_id": "SUB-1903",
        "order_id": "ORD-2026-1903-09",
        "invoice_date": "2026-08-28",
        "due_date": "2026-09-15",
        "amount": 19.90,
        "status": "paid",
        "period": "September 2026",
    },
    {
        "invoice_id": "INV-2026-0904",
        "customer_id": 1918,
        "subscription_id": "SUB-1918",
        "order_id": "ORD-2026-1918-09",
        "invoice_date": "2026-08-28",
        "due_date": "2026-09-01",
        "amount": 49.90,
        "status": "overdue",
        "period": "September 2026",
    },
    {
        "invoice_id": "INV-2026-0905",
        "customer_id": 2044,
        "subscription_id": "SUB-2044",
        "order_id": "ORD-2026-2044-09",
        "invoice_date": "2026-08-28",
        "due_date": "2026-09-15",
        "amount": 29.90,
        "status": "paid",
        "period": "September 2026",
    },
    {
        "invoice_id": "INV-2026-0906",
        "customer_id": 2110,
        "subscription_id": "SUB-2110",
        "order_id": "ORD-2026-2110-09",
        "invoice_date": "2026-08-28",
        "due_date": "2026-09-15",
        "amount": 19.90,
        "status": "unpaid",
        "period": "September 2026",
    },
    {
        "invoice_id": "INV-2026-0907",
        "customer_id": 2235,
        "subscription_id": "SUB-2235",
        "order_id": "ORD-2026-2235-09",
        "invoice_date": "2026-08-28",
        "due_date": "2026-09-15",
        "amount": 49.90,
        "status": "paid",
        "period": "September 2026",
    },
    {
        "invoice_id": "INV-2026-0908",
        "customer_id": 2301,
        "subscription_id": "SUB-2301",
        "order_id": "ORD-2026-2301-09",
        "invoice_date": "2026-08-28",
        "due_date": "2026-09-15",
        "amount": 29.90,
        "status": "paid",
        "period": "September 2026",
    },
]

PAYMENTS = [
    {
        "payment_id": "PAY-2026-0801",
        "invoice_id": "INV-2026-0801",
        "customer_id": 1827,
        "amount": 49.90,
        "status": "paid",
        "payment_date": "2026-08-10",
    },
    {
        "payment_id": "PAY-2026-0802",
        "invoice_id": "INV-2026-0802",
        "customer_id": 1841,
        "amount": 29.90,
        "status": "paid",
        "payment_date": "2026-08-08",
    },
    {
        "payment_id": "PAY-2026-0803",
        "invoice_id": "INV-2026-0803",
        "customer_id": 1903,
        "amount": 19.90,
        "status": "paid",
        "payment_date": "2026-08-06",
    },
    {
        "payment_id": "PAY-2026-0804",
        "invoice_id": "INV-2026-0804",
        "customer_id": 1918,
        "amount": 49.90,
        "status": "paid",
        "payment_date": "2026-08-12",
    },
    {
        "payment_id": "PAY-2026-0805",
        "invoice_id": "INV-2026-0805",
        "customer_id": 2044,
        "amount": 29.90,
        "status": "paid",
        "payment_date": "2026-08-09",
    },
    {
        "payment_id": "PAY-2026-0806",
        "invoice_id": "INV-2026-0806",
        "customer_id": 2110,
        "amount": 19.90,
        "status": "paid",
        "payment_date": "2026-08-14",
    },
    {
        "payment_id": "PAY-2026-0807",
        "invoice_id": "INV-2026-0807",
        "customer_id": 2235,
        "amount": 49.90,
        "status": "paid",
        "payment_date": "2026-08-07",
    },
    {
        "payment_id": "PAY-2026-0808",
        "invoice_id": "INV-2026-0808",
        "customer_id": 2301,
        "amount": 29.90,
        "status": "paid",
        "payment_date": "2026-08-11",
    },
    {
        "payment_id": "PAY-2026-0901",
        "invoice_id": "INV-2026-0901",
        "customer_id": 1827,
        "amount": 49.90,
        "status": "unpaid",
        "payment_date": None,
    },
    {
        "payment_id": "PAY-2026-0902",
        "invoice_id": "INV-2026-0902",
        "customer_id": 1841,
        "amount": 29.90,
        "status": "paid",
        "payment_date": "2026-09-03",
    },
    {
        "payment_id": "PAY-2026-0903",
        "invoice_id": "INV-2026-0903",
        "customer_id": 1903,
        "amount": 19.90,
        "status": "paid",
        "payment_date": "2026-09-02",
    },
    {
        "payment_id": "PAY-2026-0904",
        "invoice_id": "INV-2026-0904",
        "customer_id": 1918,
        "amount": 49.90,
        "status": "overdue",
        "payment_date": None,
    },
    {
        "payment_id": "PAY-2026-0905",
        "invoice_id": "INV-2026-0905",
        "customer_id": 2044,
        "amount": 29.90,
        "status": "paid",
        "payment_date": "2026-09-04",
    },
    {
        "payment_id": "PAY-2026-0906",
        "invoice_id": "INV-2026-0906",
        "customer_id": 2110,
        "amount": 19.90,
        "status": "unpaid",
        "payment_date": None,
    },
    {
        "payment_id": "PAY-2026-0907",
        "invoice_id": "INV-2026-0907",
        "customer_id": 2235,
        "amount": 49.90,
        "status": "paid",
        "payment_date": "2026-09-01",
    },
    {
        "payment_id": "PAY-2026-0908",
        "invoice_id": "INV-2026-0908",
        "customer_id": 2301,
        "amount": 29.90,
        "status": "paid",
        "payment_date": "2026-09-05",
    },
]

INTERACTIONS = [
    {
        "interaction_id": "INT-1001",
        "customer_id": 1827,
        "date": "2026-09-05",
        "channel": "email",
        "interaction_type": "payment_issue",
        "text": (
            "Hello, I am contacting you because my September subscription "
            "has still not been paid. Could you please check?"
        ),
    },
    {
        "interaction_id": "INT-1002",
        "customer_id": 1827,
        "date": "2026-09-06",
        "channel": "crm_note",
        "interaction_type": "note",
        "text": (
            "Customer emailed about unpaid September premium subscription. "
            "Invoice INV-2026-0901 still open. Finance to confirm bank matching."
        ),
    },
    {
        "interaction_id": "INT-1003",
        "customer_id": 1827,
        "date": "2026-09-08",
        "channel": "reminder",
        "interaction_type": "reminder",
        "text": (
            "Friendly reminder: invoice INV-2026-0901 for EUR 49.90 is due "
            "on 15 September 2026."
        ),
    },
    {
        "interaction_id": "INT-1004",
        "customer_id": 1918,
        "date": "2026-09-03",
        "channel": "ticket",
        "interaction_type": "complaint",
        "text": (
            "We received a reminder although our August invoice was paid. "
            "Please stop sending reminders until this is clarified."
        ),
    },
    {
        "interaction_id": "INT-1005",
        "customer_id": 1918,
        "date": "2026-09-04",
        "channel": "crm_note",
        "interaction_type": "note",
        "text": (
            "Complaint about reminder. August invoice is paid; September "
            "invoice INV-2026-0904 is overdue since 1 September."
        ),
    },
    {
        "interaction_id": "INT-1006",
        "customer_id": 2110,
        "date": "2026-09-07",
        "channel": "ticket",
        "interaction_type": "payment_issue",
        "text": (
            "I think the starter plan was charged twice. Can someone check "
            "my last two invoices?"
        ),
    },
    {
        "interaction_id": "INT-1007",
        "customer_id": 2044,
        "date": "2026-09-02",
        "channel": "email",
        "interaction_type": "question",
        "text": "Could you send a copy of the September invoice for our accountant?",
    },
    {
        "interaction_id": "INT-1008",
        "customer_id": 2235,
        "date": "2026-08-20",
        "channel": "email",
        "interaction_type": "question",
        "text": "We would like to add two extra user seats next month. What is the process?",
    },
    {
        "interaction_id": "INT-1009",
        "customer_id": 1841,
        "date": "2026-09-01",
        "channel": "crm_note",
        "interaction_type": "note",
        "text": "Payment for September received. No follow-up needed.",
    },
    {
        "interaction_id": "INT-1010",
        "customer_id": 2301,
        "date": "2026-09-09",
        "channel": "ticket",
        "interaction_type": "question",
        "text": "Please confirm that our standard plan renews automatically in October.",
    },
]


def subscription_by_customer(customer_id: int) -> dict:
    """Return the unique active subscription for a customer."""
    matches = [row for row in SUBSCRIPTIONS if row["customer_id"] == customer_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one subscription for customer {customer_id}")
    return matches[0]


def latest_invoice_status(customer_id: int) -> str:
    """Return the payment status of the customer's most recent invoice."""
    invoices = [row for row in INVOICES if row["customer_id"] == customer_id]
    latest = max(invoices, key=lambda row: row["invoice_date"])
    return latest["status"]


def customers_export_rows() -> list[dict]:
    """Denormalised customer export, as a finance team might send a CSV."""
    rows = []
    for customer in CUSTOMERS:
        subscription = subscription_by_customer(customer["customer_id"])
        rows.append(
            {
                "customer_id": customer["customer_id"],
                "customer_name": customer["customer_name"],
                "company_name": customer["company_name"],
                "customer_segment": customer["customer_segment"],
                "subscription_type": subscription["subscription_type"],
                "monthly_amount": subscription["monthly_amount"],
                "signup_date": customer["signup_date"],
                "country": customer["country"],
                "payment_status": latest_invoice_status(customer["customer_id"]),
            }
        )
    return rows


def interactions_json() -> list[dict]:
    """Nested JSON documents: one object per customer, with a list of contacts."""
    documents = []
    for customer in CUSTOMERS:
        subscription = subscription_by_customer(customer["customer_id"])
        contacts = [
            {
                "date": row["date"],
                "channel": row["channel"],
                "interaction_type": row["interaction_type"],
                "text": row["text"],
            }
            for row in INTERACTIONS
            if row["customer_id"] == customer["customer_id"]
        ]
        documents.append(
            {
                "customer_id": customer["customer_id"],
                "company_name": customer["company_name"],
                "subscription": {
                    "type": subscription["subscription_type"],
                    "monthly_amount": subscription["monthly_amount"],
                    "status": latest_invoice_status(customer["customer_id"]),
                },
                "contacts": contacts,
            }
        )
    return documents


def api_invoices_payload() -> dict:
    """JSON payload served by the local HarborLink mock API."""
    return {
        "company": COMPANY["name"],
        "as_of": AS_OF_DATE,
        "invoices": [
            {
                "invoice_id": row["invoice_id"],
                "customer_id": row["customer_id"],
                "invoice_date": row["invoice_date"],
                "due_date": row["due_date"],
                "amount": row["amount"],
                "currency": COMPANY["currency"],
                "status": row["status"],
                "order_id": row["order_id"],
            }
            for row in INVOICES
            if row["invoice_id"].startswith("INV-2026-09")
        ],
    }


def example_invoice() -> dict:
    """Invoice printed in the synthetic PDF (Elena Vargas / Atelier Vega)."""
    invoice = next(row for row in INVOICES if row["invoice_id"] == "INV-2026-0901")
    customer = next(
        row for row in CUSTOMERS if row["customer_id"] == invoice["customer_id"]
    )
    subscription = subscription_by_customer(invoice["customer_id"])
    return {
        "invoice": invoice,
        "customer": customer,
        "subscription": subscription,
    }

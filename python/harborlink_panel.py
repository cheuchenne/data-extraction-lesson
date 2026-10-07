"""Larger HarborLink panel used to assemble the Section 2.c analysis table.

The eight customers from the Section 2.b extract are kept with the same
identifiers, amounts and September 2026 statuses. Additional synthetic
customers extend the history so that joins produce a table large enough
for live exploratory analysis, while remaining small enough to handle
in class.
"""

from __future__ import annotations

import random
import sys
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from harborlink import (  # noqa: E402
    AS_OF_DATE,
    COMPANY,
    CUSTOMERS,
    INTERACTIONS,
    INVOICES,
    PAYMENTS,
    SUBSCRIPTION_PRICES,
    SUBSCRIPTIONS,
)

SEED = 42
N_NEW_CUSTOMERS = 232
AS_OF = date.fromisoformat(AS_OF_DATE)
PANEL_END = date(2026, 9, 1)
FIRST_NEW_ID = 3001

FIRST_NAMES = [
    "Anna", "Ben", "Clara", "David", "Eva", "Felix", "Greta", "Hugo",
    "Ines", "Jules", "Karim", "Lea", "Marta", "Noah", "Olga", "Paul",
    "Quinn", "Rosa", "Samir", "Tina", "Ulrich", "Vera", "Willem", "Yuki",
]
LAST_NAMES = [
    "Bernard", "Costa", "Declercq", "Engel", "Fabre", "Gatti", "Hansen",
    "Ibrahim", "Janssens", "Kowalski", "Lefevre", "Moreau", "Nguyen",
    "Ortiz", "Peeters", "Rossi", "Silva", "Traore", "Urban", "Vermeulen",
]
COMPANY_WORDS = [
    "Atelier", "Studio", "Partners", "Consulting", "Logistics", "Digital",
    "Retail", "Media", "Design", "Works", "Labs", "Group", "Agency", "Hub",
]
COUNTRIES = ["BE", "FR", "DE", "NL", "IT", "ES", "IE", "LU"]
SEGMENTS = ["smb", "mid-market", "enterprise"]
PLANS = ["starter", "standard", "premium"]
BEHAVIOURS = ["reliable", "slow", "risky"]

KNOWN_BEHAVIOUR = {
    1827: "slow",
    1841: "reliable",
    1903: "reliable",
    1918: "risky",
    2044: "reliable",
    2110: "slow",
    2235: "reliable",
    2301: "reliable",
}

KNOWN_INVOICE_IDS = {row["invoice_id"] for row in INVOICES}


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + delta
    return total // 12, total % 12 + 1


def _months_between(start: date, end: date) -> list[tuple[int, int]]:
    year, month = start.year, start.month
    months = []
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        year, month = _add_months(year, month, 1)
    return months


def _period_label(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _invoice_dates(year: int, month: int) -> tuple[date, date]:
    if (year, month) == (2026, 9):
        return date(2026, 8, 28), date(2026, 9, 15)
    invoice_date = date(year, month, 1)
    due_date = date(year, month, 15)
    return invoice_date, due_date


def _first_billing_month(signup: date) -> date:
    if signup.day <= 15:
        return date(signup.year, signup.month, 1)
    year, month = _add_months(signup.year, signup.month, 1)
    return date(year, month, 1)


def _choose_status(behaviour: str, invoice_date: date, due_date: date, rng: random.Random) -> str:
    if due_date >= AS_OF:
        if behaviour == "risky" and rng.random() < 0.55:
            return "unpaid"
        if behaviour == "slow" and rng.random() < 0.35:
            return "unpaid"
        if rng.random() < 0.08:
            return "unpaid"
        return "paid"
    if behaviour == "risky" and rng.random() < 0.22:
        return "overdue"
    if behaviour == "slow" and rng.random() < 0.06:
        return "overdue"
    if rng.random() < 0.02:
        return "overdue"
    return "paid"


def _payment_date_for(status: str, invoice_date: date, due_date: date, behaviour: str, rng: random.Random):
    if status != "paid":
        return None
    if behaviour == "slow":
        delay = rng.randint(12, 24)
    elif behaviour == "risky":
        delay = rng.randint(8, 20)
    else:
        delay = rng.randint(2, 12)
    paid = invoice_date + timedelta(days=delay)
    last_day = monthrange(invoice_date.year, invoice_date.month)[1]
    latest = date(invoice_date.year, invoice_date.month, last_day)
    if paid > AS_OF:
        paid = min(AS_OF, due_date)
    if paid > latest and invoice_date.month != 8:
        paid = min(paid, latest + timedelta(days=5))
    return paid


def _new_customers(rng: random.Random) -> list[dict]:
    rows = []
    used_names: set[tuple[str, str]] = set()
    for offset in range(N_NEW_CUSTOMERS):
        customer_id = FIRST_NEW_ID + offset
        while True:
            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            if (first, last) not in used_names:
                used_names.add((first, last))
                break
        signup = date(2024, 11, 1) + timedelta(days=rng.randint(0, 560))
        if signup >= date(2026, 7, 1):
            signup = date(2026, 6, 15)
        segment = rng.choices(SEGMENTS, weights=[0.62, 0.28, 0.10], k=1)[0]
        if rng.random() < 0.03:
            segment = None
        plan = rng.choices(PLANS, weights=[0.40, 0.40, 0.20], k=1)[0]
        behaviour = rng.choices(BEHAVIOURS, weights=[0.72, 0.18, 0.10], k=1)[0]
        rows.append(
            {
                "customer_id": customer_id,
                "customer_name": f"{first} {last}",
                "company_name": f"{last} {rng.choice(COMPANY_WORDS)}",
                "customer_segment": segment,
                "country": rng.choice(COUNTRIES),
                "signup_date": signup.isoformat(),
                "subscription_type": plan,
                "monthly_amount": SUBSCRIPTION_PRICES[plan],
                "behaviour": behaviour,
            }
        )
    return rows


def _known_customer_rows() -> list[dict]:
    by_id = {row["customer_id"]: row for row in CUSTOMERS}
    rows = []
    for subscription in SUBSCRIPTIONS:
        customer = by_id[subscription["customer_id"]]
        rows.append(
            {
                **customer,
                "subscription_type": subscription["subscription_type"],
                "monthly_amount": subscription["monthly_amount"],
                "behaviour": KNOWN_BEHAVIOUR[customer["customer_id"]],
            }
        )
    return rows


def _generate_history(customer: dict, rng: random.Random) -> tuple[list[dict], list[dict]]:
    signup = date.fromisoformat(customer["signup_date"])
    start = _first_billing_month(signup)
    invoices = []
    payments = []
    for year, month in _months_between(start, PANEL_END):
        period = _period_label(year, month)
        if customer["customer_id"] in KNOWN_BEHAVIOUR and period in {"2026-08", "2026-09"}:
            continue
        invoice_date, due_date = _invoice_dates(year, month)
        if customer["customer_id"] == 1918 and rng.random() < 0.25:
            due_date = date(year, month, 1)
        status = _choose_status(customer["behaviour"], invoice_date, due_date, rng)
        invoice_id = f"INV-{year:04d}{month:02d}-{customer['customer_id']}"
        invoices.append(
            {
                "invoice_id": invoice_id,
                "customer_id": customer["customer_id"],
                "subscription_id": f"SUB-{customer['customer_id']}",
                "order_id": f"ORD-{year:04d}{month:02d}-{customer['customer_id']}",
                "invoice_date": invoice_date.isoformat(),
                "due_date": due_date.isoformat(),
                "amount": customer["monthly_amount"],
                "status": status,
                "period": period,
            }
        )
        payments.append(
            {
                "payment_id": f"PAY-{year:04d}{month:02d}-{customer['customer_id']}",
                "invoice_id": invoice_id,
                "customer_id": customer["customer_id"],
                "amount": customer["monthly_amount"],
                "status": status,
                "payment_date": (
                    None
                    if status != "paid"
                    else _payment_date_for(
                        status, invoice_date, due_date, customer["behaviour"], rng
                    ).isoformat()
                ),
            }
        )
    return invoices, payments


def _known_period(period_name: str) -> str:
    mapping = {"August 2026": "2026-08", "September 2026": "2026-09"}
    return mapping[period_name]


def _normalise_known_invoices() -> list[dict]:
    rows = []
    for invoice in INVOICES:
        rows.append({**invoice, "period": _known_period(invoice["period"])})
    return rows


def _interaction_text(kind: str, period: str, amount: float) -> str:
    if kind == "payment_issue":
        return f"Could you check the {period} subscription invoice of EUR {amount:.2f}?"
    if kind == "reminder":
        return f"Reminder: the {period} invoice of EUR {amount:.2f} is still open."
    if kind == "complaint":
        return f"We received a reminder for {period} and would like this clarified."
    if kind == "question":
        return f"Please send a copy of the {period} invoice for our accountant."
    return f"CRM note about the {period} billing cycle."


def _generate_interactions(invoices: list[dict], rng: random.Random) -> list[dict]:
    rows = []
    next_id = 2000
    for invoice in invoices:
        if invoice["invoice_id"] in KNOWN_INVOICE_IDS:
            continue
        status = invoice["status"]
        period = invoice["period"]
        year, month = int(period[:4]), int(period[5:])
        base = date(year, month, min(8, monthrange(year, month)[1]))
        planned: list[tuple[str, str]] = []
        if status in {"unpaid", "overdue"}:
            planned.append(("email", "payment_issue"))
            if rng.random() < 0.75:
                planned.append(("reminder", "reminder"))
            if status == "overdue" and rng.random() < 0.55:
                planned.append(("ticket", "complaint"))
            if rng.random() < 0.40:
                planned.append(("crm_note", "note"))
        elif rng.random() < 0.12:
            planned.append(("email", "question"))
        elif rng.random() < 0.06:
            planned.append(("ticket", "question"))
        for channel, kind in planned:
            offset = rng.randint(0, 10)
            event_date = min(base + timedelta(days=offset), AS_OF)
            next_id += 1
            rows.append(
                {
                    "interaction_id": f"INT-{next_id}",
                    "customer_id": invoice["customer_id"],
                    "date": event_date.isoformat(),
                    "channel": channel,
                    "interaction_type": kind,
                    "text": _interaction_text(kind, period, invoice["amount"]),
                    "period": period,
                }
            )
    known = []
    for row in INTERACTIONS:
        event = date.fromisoformat(row["date"])
        known.append({**row, "period": _period_label(event.year, event.month)})
    return known + rows


def build_panel() -> dict[str, list[dict]]:
    """Return all panel tables as lists of dictionaries."""
    rng = random.Random(SEED)
    customers = _known_customer_rows() + _new_customers(rng)

    subscriptions = list(SUBSCRIPTIONS)
    for customer in customers:
        if customer["customer_id"] in KNOWN_BEHAVIOUR:
            continue
        subscriptions.append(
            {
                "subscription_id": f"SUB-{customer['customer_id']}",
                "customer_id": customer["customer_id"],
                "subscription_type": customer["subscription_type"],
                "monthly_amount": customer["monthly_amount"],
                "start_date": customer["signup_date"],
                "status": "active",
            }
        )

    invoices = _normalise_known_invoices()
    payments = [dict(row) for row in PAYMENTS]
    for customer in customers:
        extra_invoices, extra_payments = _generate_history(customer, rng)
        invoices.extend(extra_invoices)
        payments.extend(extra_payments)

    interactions = _generate_interactions(invoices, rng)
    return {
        "company": COMPANY,
        "customers": customers,
        "subscriptions": subscriptions,
        "invoices": invoices,
        "payments": payments,
        "interactions": interactions,
    }


def customers_export_rows(customers: list[dict]) -> list[dict]:
    """Customer attributes only. Plan and amount live in the subscription table."""
    rows = []
    for customer in customers:
        rows.append(
            {
                "customer_id": customer["customer_id"],
                "customer_name": customer["customer_name"],
                "company_name": customer["company_name"],
                "customer_segment": customer["customer_segment"],
                "country": customer["country"],
                "signup_date": customer["signup_date"],
            }
        )
    return rows


def interactions_json(panel: dict[str, list[dict]]) -> list[dict]:
    by_customer: dict[int, list[dict]] = {}
    for row in panel["interactions"]:
        by_customer.setdefault(row["customer_id"], []).append(
            {
                "date": row["date"],
                "channel": row["channel"],
                "interaction_type": row["interaction_type"],
                "text": row["text"],
            }
        )
    subscriptions = {row["customer_id"]: row for row in panel["subscriptions"]}
    latest_status = {}
    for invoice in panel["invoices"]:
        latest_status[invoice["customer_id"]] = invoice["status"]
    documents = []
    for customer in panel["customers"]:
        subscription = subscriptions[customer["customer_id"]]
        documents.append(
            {
                "customer_id": customer["customer_id"],
                "company_name": customer["company_name"],
                "subscription": {
                    "type": subscription["subscription_type"],
                    "monthly_amount": subscription["monthly_amount"],
                    "status": latest_status[customer["customer_id"]],
                },
                "contacts": by_customer.get(customer["customer_id"], []),
            }
        )
    return documents


def api_invoices_payload(panel: dict[str, list[dict]]) -> dict:
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
                "period": row["period"],
            }
            for row in panel["invoices"]
            if row["period"] == "2026-09"
        ],
    }

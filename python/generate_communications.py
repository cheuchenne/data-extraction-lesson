"""Generate short customer communications from telco_customers.csv.

Run from the project root:

    python python/generate_communications.py

For each selected customer:
  - Account facts are written the same way for both outcomes
  - Churn changes only how often one message mentions a service problem
  - The same service sentences are used for both outcomes
  - The word "churn" is never written into a message
"""

from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

PYTHON_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from generate_profiles import customer_profile  # noqa: E402
from telco_paths import COMMS_FILE, PROFILES_FILE  # noqa: E402

OUTPUT_FILE = COMMS_FILE

# Process all customers by default. Override for pilots, e.g.:
#   N_CUSTOMERS=5 python3 generate_communications.py
_N = os.environ.get("N_CUSTOMERS", "").strip()
N_CUSTOMERS = int(_N) if _N else None  # None = all rows
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))

COMMUNICATION_TYPES = [
    "customer_service",
    "billing_renewal",
    "promotion",
    "complaint",
    "support",
    "crm_note",
]


# ---------------------------------------------------------------------------
# Profile parsing (key facts only)
# ---------------------------------------------------------------------------

def parse_profile(profile: str) -> dict:
    """Extract a few important facts from the CustomerProfile text."""
    text = profile if isinstance(profile, str) else ""

    def find(*patterns, default=""):
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return default

    gender = "female" if re.search(r"\bfemale\b", text, re.I) else (
        "male" if re.search(r"\bmale\b", text, re.I) else "customer"
    )

    if re.search(r"fiber-optic Internet", text, re.I):
        internet = "fiber-optic Internet"
    elif re.search(r"DSL Internet", text, re.I):
        internet = "DSL Internet"
    elif re.search(r"no Internet service", text, re.I):
        internet = "no Internet service"
    else:
        internet = "Internet service"

    has_phone = bool(re.search(r"has phone service\.", text, re.I))
    no_phone = bool(
        re.search(r"does not have phone service|has no phone service", text, re.I)
    )
    phone = "no phone service" if no_phone and not has_phone else (
        "phone service" if has_phone else "phone service status unclear"
    )

    if re.search(r"month-to-month contract", text, re.I):
        contract = "month-to-month"
    elif re.search(r"one-year contract", text, re.I):
        contract = "one-year"
    elif re.search(r"two-year contract", text, re.I):
        contract = "two-year"
    else:
        contract = "current"

    monthly = find(r"monthly charge is approximately \$([0-9]+(?:\.[0-9]+)?)")
    total = find(r"charged approximately \$([0-9]+(?:\.[0-9]+)?) in total")
    tenure = find(r"with the company for (\d+) months?")

    if re.search(r"electronic check", text, re.I):
        payment = "electronic check"
    elif re.search(r"mailed check", text, re.I):
        payment = "mailed check"
    elif re.search(r"automatic bank transfer", text, re.I):
        payment = "automatic bank transfer"
    elif re.search(r"automatic credit card", text, re.I):
        payment = "automatic credit card"
    else:
        payment = "their usual payment method"

    paperless = bool(re.search(r"uses paperless billing", text, re.I))

    addons_on = []
    addons_off = []
    addon_map = {
        "online security": r"online security service",
        "online backup": r"online backup service",
        "device protection": r"device protection service",
        "tech support": r"technical support service",
        "streaming TV": r"streaming TV service",
        "streaming movies": r"streaming movie service",
    }
    for label, pattern in addon_map.items():
        if re.search(rf"subscribes to the company's {pattern}", text, re.I):
            addons_on.append(label)
        elif re.search(rf"does not subscribe to the company's {pattern}", text, re.I):
            addons_off.append(label)

    return {
        "gender": gender,
        "internet": internet,
        "phone": phone,
        "contract": contract,
        "monthly": monthly,
        "total": total,
        "tenure": tenure,
        "payment": payment,
        "paperless": paperless,
        "addons_on": addons_on,
        "addons_off": addons_off,
    }


def money(value: str) -> str:
    """Format a parsed amount for billing-style precision."""
    if not value:
        return "$0.00"
    try:
        return f"${float(value):.2f}"
    except ValueError:
        return f"${value}"


def service_summary(facts: dict) -> str:
    parts = [facts["internet"]]
    if facts["phone"] == "no phone service":
        parts.append("no phone")
    if facts["addons_on"]:
        parts.append(facts["addons_on"][0] + " on")
    return "; ".join(parts)


def missing_addon(facts: dict) -> str | None:
    if facts["addons_off"]:
        return facts["addons_off"][0]
    return None


# ---------------------------------------------------------------------------
# Text generation
#
# Account facts are written the same way for both outcomes. Churn is used
# only as a probability: one message, when the customer has a complaint,
# a support contact or a CRM note, may include a service sentence. The same
# sentences are available for both outcomes. A negative sentence is more
# common for customers who leave and still occurs for customers who stay.
# ---------------------------------------------------------------------------

# Shared wording. None of these sentences is reserved for one outcome.
NEGATIVE_NOTES = (
    "the line dropped during the week",
    "the latest bill was higher than expected",
    "support was slow to call back",
    "there was a short outage on the service",
)
ROUTINE_NOTES = (
    "there was no issue to report",
    "the service has been fine",
    "the last bill matched the plan",
    "this was a routine question",
)

# Chance that the single service sentence is a negative note.
# The gap is large enough to add information, and small enough that many
# customers who leave have a routine note and many who stay have a negative one.
P_NEGATIVE_NOTE = {"Yes": 0.75, "No": 0.18}
CARRIER_TYPES = ("complaint", "support", "crm_note")


def experience_clause(churn: str, rng: random.Random) -> str:
    """Draw one service sentence. The lists do not depend on the outcome."""
    p_negative = P_NEGATIVE_NOTE.get(churn, P_NEGATIVE_NOTE["No"])
    pool = NEGATIVE_NOTES if rng.random() < p_negative else ROUTINE_NOTES
    return rng.choice(pool)


_AGENTS = (
    "Alex", "Jordan", "Sam", "Riley", "Casey", "Morgan", "Taylor", "Drew",
)
_CLOSINGS = (
    "Thanks again.",
    "We can pick this up if you write back.",
    "Call us if something else comes up.",
    "I will leave the account as it is unless you say otherwise.",
)


def _choice(style: random.Random, options: tuple[str, ...]) -> str:
    return options[style.randrange(len(options))]


def _trailer(style: random.Random) -> str:
    agent = _choice(style, _AGENTS)
    closing = _choice(style, _CLOSINGS)
    ref = style.randrange(100000, 999999)
    aside = _choice(style, (
        f"{agent} took this on the phone queue.",
        f"{agent} handled this by email.",
        f"{agent} copied the chat into the account.",
        f"A colleague of {agent} reviewed the wording before it was stored.",
        f"The contact was tagged for the morning desk.",
        f"The contact was tagged for the afternoon desk.",
        f"Callback window offered: later the same day.",
        f"No callback was requested.",
    ))
    return f"\n\n— {agent}, ref {ref}. {aside} {closing}"


def generate_customer_service(facts: dict, note: str | None, style: random.Random) -> str:
    body = (
        "**Agent:** Hi, thanks for contacting support. How can I help?\n"
        "**Customer:** I want to confirm the services on my account.\n"
        "**Agent:** I have the account open. Tell me what you would like to check.\n"
        "**Customer:** Thanks, that was the only question.\n"
        "**Agent:** You are welcome."
    )
    return body + _trailer(style)


def generate_billing_renewal(facts: dict, note: str | None, style: random.Random) -> str:
    monthly = money(facts["monthly"])
    total = money(facts["total"])
    billing = "paperless billing" if facts["paperless"] else "standard billing"
    body = (
        f"**Subject:** Your monthly bill — {monthly}\n\n"
        f"Hello,\n\n"
        f"Your current plan is billed at **{monthly}** ({billing}). "
        f"Total charged to date: **{total}**.\n\n"
        f"Reply if you have any questions about this bill.\n\n"
        f"Best regards,\nBilling Team"
    )
    return body + _trailer(style)


def generate_promotion(facts: dict, note: str | None, style: random.Random) -> str:
    offer = missing_addon(facts) or "a plan review"
    body = (
        f"**Subject:** An optional add-on\n\n"
        f"Hello,\n\n"
        f"Some customers add {offer}. "
        f"If you are interested, we can show the price impact on your "
        f"{money(facts['monthly'])} monthly charge.\n\n"
        f"Best regards,\nOffers Team"
    )
    return body + _trailer(style)


def generate_complaint(facts: dict, note: str | None, style: random.Random) -> str:
    detail = note or "a question about the account"
    body = (
        f"**Customer:** I am calling about the service. What I wanted noted: {detail}.\n"
        f"**Agent:** Thanks for calling. I have logged that and can review the latest charge with you.\n"
        f"**Customer:** Please keep the note on the account.\n"
        f"**Agent:** Done. We can follow up if you contact us again."
    )
    return body + _trailer(style)


def generate_support(facts: dict, note: str | None, style: random.Random) -> str:
    detail = note or "a general check of the service"
    body = (
        f"**Agent:** Support here. I can help with the service on your account.\n"
        f"**Customer:** The reason for the call: {detail}.\n"
        f"**Agent:** I have opened a note and recorded that."
    )
    return body + _trailer(style)


def generate_crm_note(facts: dict, note: str | None, style: random.Random) -> str:
    detail = note or "no extra service comment"
    body = (
        f"Customer contact logged. Latest contact: {detail}. "
        f"Monthly charge on the account is {money(facts['monthly'])}."
    )
    return body + _trailer(style)


GENERATORS = {
    "customer_service": generate_customer_service,
    "billing_renewal": generate_billing_renewal,
    "promotion": generate_promotion,
    "complaint": generate_complaint,
    "support": generate_support,
    "crm_note": generate_crm_note,
}


def generate_texts_for_customer(profile: str, churn: str, rng: random.Random) -> list[dict]:
    """Generate 2–5 communications for one customer.

    At most one message carries a service sentence. Billing, promotion and
    ordinary service confirmation never do. The sentence is not a label:
    both outcomes can receive a negative note or a routine note.
    """
    facts = parse_profile(profile)
    n_texts = rng.randint(2, 5)
    types = rng.sample(COMMUNICATION_TYPES, k=n_texts)
    if not any(comm_type in CARRIER_TYPES for comm_type in types):
        types[0] = rng.choice(CARRIER_TYPES)
    carrier_indexes = [index for index, comm_type in enumerate(types) if comm_type in CARRIER_TYPES]
    carrier = rng.choice(carrier_indexes) if carrier_indexes else None
    note = experience_clause(churn, rng) if carrier is not None else None
    # Wording is chosen after the service sentence, so the sentence itself
    # stays on the same customers when the surrounding lines change.
    style = random.Random(rng.randrange(2**31 - 1))

    texts = []
    for index, comm_type in enumerate(types):
        body = GENERATORS[comm_type](facts, note if index == carrier else None, style)
        assert "churn" not in body.lower()
        texts.append({"communication_type": comm_type, "text": body})
    return texts


# ---------------------------------------------------------------------------
# Main: process customers one at a time (sample of N_CUSTOMERS)
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(RANDOM_SEED)

    profiles_path = Path(PROFILES_FILE)
    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Missing {PROFILES_FILE}. Run generate_profiles.py first."
        )

    df = pd.read_csv(profiles_path)
    if "Churn" not in df.columns:
        raise ValueError("Expected column Churn.")

    if N_CUSTOMERS is None:
        selected_rows = list(range(len(df)))
    else:
        selected_rows = rng.sample(range(len(df)), k=min(N_CUSTOMERS, len(df)))

    print(
        f"Generating communications for {len(selected_rows)} / {len(df)} customers "
        f"(seed={RANDOM_SEED})…"
    )

    output_rows = []

    for i, source_row in enumerate(selected_rows, start=1):
        row = df.iloc[source_row]
        profile = customer_profile(row)
        churn = str(row["Churn"]).strip()

        # Independent RNG stream per customer so full-file order is stable
        # even if we later subsample with the same seed.
        cust_rng = random.Random(RANDOM_SEED + source_row)
        generated = generate_texts_for_customer(profile, churn, cust_rng)

        for text_index, item in enumerate(generated, start=1):
            output_rows.append(
                {
                    "source_row": source_row,
                    "customerID": row.get("customerID", ""),
                    "n_texts": len(generated),
                    "text_index": text_index,
                    "communication_type": item["communication_type"],
                    "text": item["text"],
                }
            )

        if i % 500 == 0 or i == len(selected_rows):
            print(f"  … {i}/{len(selected_rows)} customers", flush=True)

    out = pd.DataFrame(output_rows)
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"\nWrote {len(out)} texts for {len(selected_rows)} customers.")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

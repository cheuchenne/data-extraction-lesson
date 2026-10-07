"""Generate short customer communications from telco_with_profiles.csv.

Run from the project root:

    python python/generate_communications.py

For each selected customer:
  - CustomerProfile is the principal textual input
  - Churn is private context (never mentioned in the texts)
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
# Text generation (Churn = hidden tone only)
# ---------------------------------------------------------------------------

def generate_customer_service(facts: dict, churn: str) -> str:
    summary = service_summary(facts)
    price = money(facts["monthly"])
    if churn == "Yes":
        return (
            f"**Agent:** Hi, thanks for contacting support. How can I help?\n"
            f"**Customer:** I want to double-check what I’m paying for — "
            f"my setup should be {summary}.\n"
            f"**Agent:** Confirmed: {summary}. You’re on a {facts['contract']} "
            f"plan at {price}, paid by {facts['payment']}.\n"
            f"**Customer:** Okay, thanks for clarifying.\n"
            f"**Agent:** Glad to help."
        )
    return (
        f"**Agent:** Hi, thanks for contacting support. How can I help?\n"
        f"**Customer:** Just confirming my setup — {summary}.\n"
        f"**Agent:** Confirmed. You’re on a {facts['contract']} plan at {price}, "
        f"paid by {facts['payment']}.\n"
        f"**Customer:** Perfect, thanks.\n"
        f"**Agent:** You’re welcome. Have a good day."
    )


def generate_billing_renewal(facts: dict, churn: str) -> str:
    monthly = money(facts["monthly"])
    total = money(facts["total"])
    billing = "paperless billing" if facts["paperless"] else "standard billing"
    if churn == "Yes":
        return (
            f"**Subject:** Your bill — {monthly}\n\n"
            f"Hello,\n\n"
            f"Your {facts['contract']} plan ({service_summary(facts)}) is billed at "
            f"**{monthly}**. Payment method: {facts['payment']} ({billing}). "
            f"Total charged to date: **{total}**.\n\n"
            f"If anything on this bill looks incorrect, reply and we will review it "
            f"with you.\n\n"
            f"Best regards,\nBilling Team"
        )
    return (
        f"**Subject:** Your monthly bill — {monthly}\n\n"
        f"Hello,\n\n"
        f"Your {facts['contract']} plan ({service_summary(facts)}) is billed at "
        f"**{monthly}**. Payment is by {facts['payment']} via {billing}. "
        f"Total charged to date: **{total}**.\n\n"
        f"Reply if you have any questions about this bill.\n\n"
        f"Best regards,\nBilling Team"
    )


def generate_promotion(facts: dict, churn: str) -> str:
    offer = missing_addon(facts) or "a simpler plan review"
    if churn == "Yes":
        return (
            f"**Subject:** A quick option for your account\n\n"
            f"Hello,\n\n"
            f"Based on your {facts['contract']} {facts['internet']} service, "
            f"you may want to look at {offer}. No obligation — reply if you’d like "
            f"a short overview of costs before your next bill.\n\n"
            f"Best regards,\nOffers Team"
        )
    return (
        f"**Subject:** Optional add-on for your {facts['internet']}\n\n"
        f"Hello,\n\n"
        f"Customers on {facts['contract']} plans sometimes add {offer}. "
        f"If you’re interested, we can show the exact price impact on your "
        f"{money(facts['monthly'])} monthly charge.\n\n"
        f"Best regards,\nOffers Team"
    )


def generate_complaint(facts: dict, churn: str) -> str:
    monthly = money(facts["monthly"])
    if churn == "Yes":
        return (
            f"**Customer:** I’m frustrated — for {monthly} a month on "
            f"{facts['internet']}, I expected fewer issues / clearer billing.\n"
            f"**Agent:** I’m sorry about that. I see you’re on a "
            f"{facts['contract']} plan ({service_summary(facts)}). "
            f"I can review the latest charge and service details with you now.\n"
            f"**Customer:** Please do — I need this sorted.\n"
            f"**Agent:** Understood. I’ll walk through the account step by step."
        )
    return (
        f"**Customer:** I had a minor issue with my {facts['internet']} service "
        f"and wanted it noted.\n"
        f"**Agent:** Thanks for flagging it. You’re on {service_summary(facts)} "
        f"at {monthly}. I’ve logged the report and we can follow up if it continues.\n"
        f"**Customer:** Okay, appreciated.\n"
        f"**Agent:** You’re welcome."
    )


def generate_support(facts: dict, churn: str) -> str:
    if facts["internet"] == "no Internet service":
        topic = "phone service settings"
    else:
        topic = f"{facts['internet']} connectivity"
    if churn == "Yes":
        return (
            f"**Agent:** Support here — I can help with {topic}.\n"
            f"**Customer:** Yes, please. It’s been unreliable and I’m on "
            f"{facts['contract']} at {money(facts['monthly'])}.\n"
            f"**Agent:** Thanks — I’ve opened a support case for "
            f"{service_summary(facts)}. I’ll confirm the next check and update you."
        )
    return (
        f"**Agent:** Support here — happy to help with {topic}.\n"
        f"**Customer:** Quick check on my connection, please.\n"
        f"**Agent:** Looking at your account now ({service_summary(facts)}). "
        f"Basic diagnostics look fine; tell me if anything still seems off."
    )


def generate_crm_note(facts: dict, churn: str) -> str:
    tenure = facts["tenure"] or "?"
    base = (
        f"{facts['gender'].capitalize()} customer (~{tenure} months). "
        f"{service_summary(facts)}. {facts['contract'].capitalize()} contract, "
        f"{facts['payment']}, {money(facts['monthly'])}/mo."
    )
    if churn == "Yes":
        return (
            f"{base} Recent tone suggests friction (billing clarity or service "
            f"expectations). Keep replies concrete; offer a calm review of the "
            f"current plan before suggesting changes."
        )
    return (
        f"{base} Recent contact was routine, not a complaint. "
        f"Good candidate for gentle education on optional add-ons later; "
        f"no pressure—keep onboarding tone light and check satisfaction "
        f"after another billing cycle."
    )


GENERATORS = {
    "customer_service": generate_customer_service,
    "billing_renewal": generate_billing_renewal,
    "promotion": generate_promotion,
    "complaint": generate_complaint,
    "support": generate_support,
    "crm_note": generate_crm_note,
}


def generate_texts_for_customer(profile: str, churn: str, rng: random.Random) -> list[dict]:
    """Generate 2–5 varied communications for one customer."""
    facts = parse_profile(profile)
    n_texts = rng.randint(2, 5)
    types = rng.sample(COMMUNICATION_TYPES, k=n_texts)

    texts = []
    for comm_type in types:
        body = GENERATORS[comm_type](facts, churn)
        # Safety: never leak churn wording into outputs.
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
    if "CustomerProfile" not in df.columns or "Churn" not in df.columns:
        raise ValueError("Expected columns CustomerProfile and Churn.")

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
        profile = row["CustomerProfile"]
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

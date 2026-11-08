"""Rebuild telco_customers.csv from the public IBM Telco churn table.

    python python/generate_profiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PYTHON_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from telco_paths import IBM_TELCO_FILE, PROFILES_FILE  # noqa: E402

INPUT_FILE = IBM_TELCO_FILE
OUTPUT_FILE = PROFILES_FILE

translation = {
    "gender": {
        "Female": "The customer is female.",
        "Male": "The customer is male.",
    },
    "SeniorCitizen": {
        0: "The customer is not a senior citizen.",
        1: "The customer is a senior citizen.",
    },
    "Partner": {
        "Yes": "The customer has a partner.",
        "No": "The customer does not have a partner.",
    },
    "Dependents": {
        "Yes": "The customer has dependents.",
        "No": "The customer has no dependents.",
    },
    "PhoneService": {
        "Yes": "The customer has phone service.",
        "No": "The customer does not have phone service.",
    },
    "MultipleLines": {
        "Yes": "The customer has multiple phone lines.",
        "No": "The customer has a single phone line.",
        "No phone service": "The customer has no phone service.",
    },
    "InternetService": {
        "DSL": "The customer uses DSL Internet.",
        "Fiber optic": "The customer uses fiber-optic Internet.",
        "No": "The customer has no Internet service.",
    },
    "OnlineSecurity": {
        "Yes": "The customer subscribes to the company's online security service.",
        "No": "The customer does not subscribe to the company's online security service.",
        "No internet service": (
            "The online security service is not applicable because "
            "the customer has no Internet service."
        ),
    },
    "OnlineBackup": {
        "Yes": "The customer subscribes to the company's online backup service.",
        "No": "The customer does not subscribe to the company's online backup service.",
        "No internet service": (
            "The online backup service is not applicable because "
            "the customer has no Internet service."
        ),
    },
    "DeviceProtection": {
        "Yes": "The customer subscribes to the company's device protection service.",
        "No": "The customer does not subscribe to the company's device protection service.",
        "No internet service": (
            "The device protection service is not applicable because "
            "the customer has no Internet service."
        ),
    },
    "TechSupport": {
        "Yes": "The customer subscribes to the company's technical support service.",
        "No": "The customer does not subscribe to the company's technical support service.",
        "No internet service": (
            "Technical support is not applicable because "
            "the customer has no Internet service."
        ),
    },
    "StreamingTV": {
        "Yes": "The customer subscribes to the company's streaming TV service.",
        "No": "The customer does not subscribe to the company's streaming TV service.",
        "No internet service": (
            "Streaming TV is not applicable because "
            "the customer has no Internet service."
        ),
    },
    "StreamingMovies": {
        "Yes": "The customer subscribes to the company's streaming movie service.",
        "No": "The customer does not subscribe to the company's streaming movie service.",
        "No internet service": (
            "Streaming movies are not applicable because "
            "the customer has no Internet service."
        ),
    },
    "Contract": {
        "Month-to-month": "The customer has a month-to-month contract.",
        "One year": "The customer has a one-year contract.",
        "Two year": "The customer has a two-year contract.",
    },
    "PaperlessBilling": {
        "Yes": "The customer uses paperless billing.",
        "No": "The customer does not use paperless billing.",
    },
    "PaymentMethod": {
        "Electronic check": "The customer pays by electronic check.",
        "Mailed check": "The customer pays by mailed check.",
        "Bank transfer (automatic)": "The customer pays by automatic bank transfer.",
        "Credit card (automatic)": "The customer pays by automatic credit card payment.",
    },
}


def customer_profile(row):
    sentences = []
    categorical_variables = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]
    for variable in categorical_variables:
        sentences.append(translation[variable][row[variable]])
    sentences.append(
        f"The customer has been with the company for {row['tenure']} months."
    )
    sentences.append(
        f"The customer's monthly charge is approximately ${row['MonthlyCharges']:.2f}."
    )
    sentences.append(
        f"The customer has been charged approximately "
        f"${row['TotalCharges']:.2f} in total since joining the company."
    )
    return " ".join(sentences)


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    print("Number of customers:", len(df))
    print("Number of variables:", len(df.columns))
    print("\nExample profile:\n")
    print(customer_profile(df.iloc[0]))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print("\nProfiles generated:", len(df))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

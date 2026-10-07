"""Load the HarborLink invoice table used in the EDA lesson."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "harborlink_invoices.csv"

DATE_COLUMNS = ["invoice_date", "due_date", "payment_date", "signup_date"]


def load_invoice_table(path: Path | str | None = None) -> pd.DataFrame:
    """Load the merged invoice-level analysis file produced in the join session."""
    source = Path(path) if path else PROCESSED
    return pd.read_csv(source, parse_dates=DATE_COLUMNS)

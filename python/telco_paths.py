"""Project-root paths for the telco modelling files.

The course-facing tables live under data/raw/telco/.
Training artefacts live under data/processed/telco/.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TELCO_RAW = ROOT / "data" / "raw" / "telco"
TELCO_RESULTS = ROOT / "data" / "processed" / "telco"

IBM_TELCO_FILE = TELCO_RAW / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROFILES_FILE = TELCO_RAW / "telco_with_profiles.csv"
COMMS_FILE = TELCO_RAW / "customer_communications.csv"
COMMS_FILE_2 = TELCO_RAW / "customer_communications_2.csv"
COMMS_FILE_3 = TELCO_RAW / "customer_communications_3.csv"
COMMS_LLM_FILE = TELCO_RAW / "customer_communications_llm.csv"

"""HarborLink convenience around the general data-dictionary helpers."""

from __future__ import annotations

try:
    from python.harborlink_catalog import variables_dict
except ImportError:
    from harborlink_catalog import variables_dict

VARIABLES = variables_dict()


def definition_frame(columns: list[str] | None = None):
    """One row per variable: name, type, definition (HarborLink analysis columns)."""
    import pandas as pd

    names = columns or list(VARIABLES)
    rows = []
    for name in names:
        meta = VARIABLES.get(name, {"kind": "unknown", "definition": ""})
        rows.append(
            {
                "variable": name,
                "type": meta["kind"],
                "definition": meta["definition"],
            }
        )
    return pd.DataFrame(rows)

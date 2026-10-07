"""Attach a data table to a description table and write dictionary / summary workbooks.

The two inputs are general:

- `data`: a DataFrame, or a path to a CSV / Excel file
- `descriptions`: a DataFrame, or a path to a CSV / Excel / JSON file
  with columns `variable`, `type`, `definition`

The two outputs are Excel workbooks (glossary + column portraits).
"""

from __future__ import annotations

import json
import tempfile
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.datavalidation import DataValidation

DESCRIPTION_COLUMNS = ["variable", "type", "definition"]
_DIGIT_PATTERN = r"-?\d+(\.\d+)?"


def read_table(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Load a rectangular data set from memory or from a file."""
    if isinstance(source, pd.DataFrame):
        return source
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("invoices", "customers", "data", "rows"):
                if key in payload and isinstance(payload[key], list):
                    return pd.DataFrame(payload[key])
        return pd.json_normalize(payload)
    return pd.read_csv(path)


def read_descriptions(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Load variable, type and definition. Accepts CSV, Excel or JSON."""
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        elif suffix == ".json":
            frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        else:
            frame = pd.read_csv(path)
    missing = [column for column in DESCRIPTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "A description table needs columns variable, type, definition. "
            f"Missing: {missing}."
        )
    return frame[DESCRIPTION_COLUMNS].copy()


def attach_descriptions(
    base: pd.DataFrame | str | Path,
    incoming: pd.DataFrame | str | Path,
    *,
    rename: dict[str, str] | None = None,
    keep: list[str] | None = None,
    skip: list[str] | None = None,
) -> pd.DataFrame:
    """Forward descriptions the way a merge forwards columns.

    Start from `base`. Add rows from `incoming`. `rename` uses the same
    mapping as the data (for example status → payment_status). `keep`
    restricts which incoming names are copied. Names already in `base`,
    and names in `skip`, are not duplicated.
    """
    catalog = read_descriptions(base)
    added = read_descriptions(incoming)
    if rename:
        added = added.copy()
        added["variable"] = added["variable"].replace(rename)
    if keep is not None:
        added = added.loc[added["variable"].isin(keep)]
    blocked = set(catalog["variable"]) | set(skip or [])
    added = added.loc[~added["variable"].isin(blocked)]
    return pd.concat([catalog, added], ignore_index=True)


def glossary_for_frame(frame: pd.DataFrame, descriptions: pd.DataFrame | str | Path) -> pd.DataFrame:
    """One glossary row per column of `frame`, using the forwarded descriptions."""
    catalog = read_descriptions(descriptions).drop_duplicates("variable", keep="first")
    lookup = catalog.set_index("variable")
    rows = []
    for name in frame.columns:
        if name in lookup.index:
            rows.append(
                {
                    "variable": name,
                    "type": lookup.loc[name, "type"],
                    "definition": lookup.loc[name, "definition"],
                }
            )
        else:
            rows.append({"variable": name, "type": "unknown", "definition": ""})
    return pd.DataFrame(rows)


def _as_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if pd.isna(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.2f}"
    return str(value)


def _column_stats(series: pd.Series, kind: str) -> dict[str, str]:
    stats = {"min": "", "max": "", "mean": "", "median": "", "std": ""}
    if kind == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        stats["min"] = _fmt(numeric.min())
        stats["max"] = _fmt(numeric.max())
        stats["mean"] = _fmt(numeric.mean())
        stats["median"] = _fmt(numeric.median())
        stats["std"] = _fmt(numeric.std(ddof=1))
    elif kind == "date":
        dates = _as_datetime(series)
        stats["min"] = _fmt(dates.min())
        stats["max"] = _fmt(dates.max())
    return stats


def _looks_like_digits(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    text = series.dropna().astype(str).str.strip()
    if text.empty:
        return False
    return bool(text.str.fullmatch(_DIGIT_PATTERN).mean() >= 0.9)


def _draw_histogram(series: pd.Series, kind: str, title: str) -> BytesIO | None:
    clean = series.dropna()
    if clean.empty:
        return None
    numeric_like = kind == "numeric" or (kind != "identifier" and _looks_like_digits(clean))
    if kind == "identifier":
        return None

    fig, ax = plt.subplots(figsize=(2.15, 1.35))
    if numeric_like:
        numeric = pd.to_numeric(clean, errors="coerce").dropna()
        if numeric.empty:
            plt.close(fig)
            return None
        nunique = int(numeric.nunique())
        if nunique <= 8:
            counts = numeric.value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values, color="#2C6EBC")
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        else:
            ax.hist(
                numeric,
                bins=min(12, max(5, nunique)),
                color="#2C6EBC",
                edgecolor="white",
            )
    elif kind == "date":
        values = _as_datetime(clean).dropna()
        if values.empty:
            plt.close(fig)
            return None
        nunique = int(values.nunique())
        ax.hist(
            values,
            bins=min(12, max(5, nunique)),
            color="#2C6EBC",
            edgecolor="white",
        )
        ax.tick_params(axis="x", labelrotation=45, labelsize=6)
    else:
        nunique = int(clean.nunique())
        if nunique > 20:
            plt.close(fig)
            return None
        counts = clean.astype(str).value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color="#2C6EBC")
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    buf.name = f"{title}.png"
    return buf


def write_variable_dictionary(
    data: pd.DataFrame | str | Path,
    descriptions: pd.DataFrame | str | Path,
    dictionary_path: str | Path,
) -> Path:
    """Write the glossary Excel file (one row per variable)."""
    frame = read_table(data)
    glossary = glossary_for_frame(frame, descriptions)
    destination = Path(dictionary_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "dictionary"
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2C3E50")
    wrap = Alignment(wrap_text=True, vertical="top")
    thin = Border(
        left=Side(style="thin", color="CCCCCC"),
        right=Side(style="thin", color="CCCCCC"),
        top=Side(style="thin", color="CCCCCC"),
        bottom=Side(style="thin", color="CCCCCC"),
    )
    for row in dataframe_to_rows(glossary, index=False, header=True):
        sheet.append(row)
    for cell in sheet[1]:
        cell.font = header_font
        cell.fill = header_fill
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, max_col=3):
        for cell in row:
            cell.alignment = wrap
            cell.border = thin
        row[2].alignment = Alignment(wrap_text=True, vertical="top")
        sheet.row_dimensions[row[0].row].height = 48
    sheet.column_dimensions["A"].width = 22
    sheet.column_dimensions["B"].width = 14
    sheet.column_dimensions["C"].width = 88
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:C{sheet.max_row}"
    note = workbook.create_sheet("how_to_read")
    note["A1"] = (
        "type = identifier: a key or a name; do not compute a mean. "
        "numeric: min, max, mean, median and standard deviation are meaningful. "
        "factor: a category (including True/False). "
        "date: min and max are the earliest and latest values."
    )
    note["A1"].alignment = Alignment(wrap_text=True, vertical="top")
    note.column_dimensions["A"].width = 100
    note.row_dimensions[1].height = 60
    workbook.save(destination)
    workbook.close()
    return destination


def write_column_summary_workbook(
    data: pd.DataFrame | str | Path,
    descriptions: pd.DataFrame | str | Path,
    summary_path: str | Path,
) -> Path:
    """Write the hover-summary workbook (notes, stats, histograms)."""
    frame = read_table(data)
    glossary = glossary_for_frame(frame, descriptions)
    destination = Path(summary_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    definitions = dict(zip(glossary["variable"], glossary["definition"]))
    kinds = dict(zip(glossary["variable"], glossary["type"]))

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "column_summary"
    header_font = Font(bold=True, color="FFFFFF", size=10)
    header_fill = PatternFill("solid", fgColor="2C3E50")
    label_fill = PatternFill("solid", fgColor="EEF2F5")
    label_font = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    labels = ["min", "max", "mean", "median", "std. deviation", "histogram"]

    sheet.cell(1, 1, value="")
    for col_idx, name in enumerate(frame.columns, start=2):
        cell = sheet.cell(1, col_idx, value=name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        text = str(definitions.get(name, "") or "")
        if text:
            note = Comment(text, "dictionary")
            note.width = 280
            note.height = min(140, 28 + 14 * text.count(".") + 8 * (len(text) // 50))
            cell.comment = note
            prompt = DataValidation(
                type="textLength",
                operator="lessThan",
                formula1="10000",
                allow_blank=True,
            )
            prompt.promptTitle = name
            prompt.prompt = text[:255]
            prompt.showInputMessage = True
            prompt.showErrorMessage = False
            prompt.add(cell.coordinate)
            sheet.add_data_validation(prompt)
        sheet.column_dimensions[get_column_letter(col_idx)].width = 16

    for row_idx, label in enumerate(labels, start=2):
        lab = sheet.cell(row_idx, 1, value=label)
        lab.fill = label_fill
        lab.font = label_font
        lab.alignment = Alignment(vertical="center")

    sheet.column_dimensions["A"].width = 16
    sheet.row_dimensions[1].height = 22
    sheet.freeze_panes = "B2"

    tmp_dir = Path(tempfile.mkdtemp(prefix="dict_hist_"))
    image_files: list[Path] = []
    for col_idx, name in enumerate(frame.columns, start=2):
        kind = str(kinds.get(name, "unknown"))
        stats = _column_stats(frame[name], kind)
        sheet.cell(2, col_idx, value=stats["min"]).alignment = center
        sheet.cell(3, col_idx, value=stats["max"]).alignment = center
        sheet.cell(4, col_idx, value=stats["mean"]).alignment = center
        sheet.cell(5, col_idx, value=stats["median"]).alignment = center
        sheet.cell(6, col_idx, value=stats["std"]).alignment = center
        buf = _draw_histogram(frame[name], kind, name)
        hist_cell = sheet.cell(7, col_idx)
        hist_cell.alignment = center
        if buf is None:
            nunique = frame[name].nunique(dropna=True)
            hist_cell.value = f"{nunique} distinct" if kind == "identifier" else "too many levels"
            continue
        image_path = tmp_dir / f"{col_idx}_{name}.png"
        image_path.write_bytes(buf.getvalue())
        image_files.append(image_path)
        picture = XLImage(str(image_path))
        picture.width = 118
        picture.height = 74
        picture.anchor = hist_cell.coordinate
        sheet.add_image(picture)

    sheet.row_dimensions[7].height = 62
    glossary_sheet = workbook.create_sheet("dictionary")
    for row in dataframe_to_rows(glossary, index=False, header=True):
        glossary_sheet.append(row)
    glossary_sheet.column_dimensions["A"].width = 22
    glossary_sheet.column_dimensions["B"].width = 14
    glossary_sheet.column_dimensions["C"].width = 88
    workbook.save(destination)
    workbook.close()
    for image_path in image_files:
        image_path.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    return destination


def write_variable_workbooks(
    data: pd.DataFrame | str | Path,
    descriptions: pd.DataFrame | str | Path,
    dictionary_path: str | Path,
    summary_path: str | Path,
) -> tuple[Path, Path]:
    """From a data table and a description table, write dictionary and summary Excel files.

    `data` and `descriptions` may each be a DataFrame or a path. The function
    does not assume HarborLink names: any pair of files with the right
    description columns can be passed in.
    """
    dictionary = write_variable_dictionary(data, descriptions, dictionary_path)
    summary = write_column_summary_workbook(data, descriptions, summary_path)
    return dictionary, summary

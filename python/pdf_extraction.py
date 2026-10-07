"""Helpers that inspect the synthetic HarborLink invoices with PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def describe_pdf(path: Path | str) -> dict:
    """Summarise pages, extracted text length and embedded images."""
    source = Path(path)
    document = pymupdf.open(source)
    try:
        pages = []
        for index, page in enumerate(document, start=1):
            text = page.get_text() or ""
            pages.append(
                {
                    "page": index,
                    "characters": len(text),
                    "images": len(page.get_images()),
                    "preview": text[:180].replace("\n", " "),
                }
            )
        return {"file": source.name, "n_pages": document.page_count, "pages": pages}
    finally:
        document.close()


def extract_text(path: Path | str) -> str:
    """Concatenate extracted text from every page."""
    document = pymupdf.open(path)
    try:
        parts = [page.get_text() or "" for page in document]
    finally:
        document.close()
    return "\n\n".join(parts).strip()


def extract_tables(path: Path | str) -> list[pd.DataFrame]:
    """Return detected tables as DataFrames. The list may be empty."""
    frames = []
    document = pymupdf.open(path)
    try:
        for page in document:
            found = page.find_tables()
            for table in found.tables:
                data = table.extract()
                if not data:
                    continue
                header, *rows = data
                frames.append(pd.DataFrame(rows, columns=header))
    finally:
        document.close()
    return frames

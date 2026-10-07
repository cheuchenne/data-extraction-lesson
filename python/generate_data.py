"""Generate every synthetic HarborLink file used in Section 2.b.

Run from the project root or from this folder:

    python python/generate_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from generate_database import write_sqlite  # noqa: E402
from harborlink import (  # noqa: E402
    AS_OF_DATE,
    COMPANY,
    CUSTOMERS,
    INVOICES,
    api_invoices_payload,
    customers_export_rows,
    example_invoice,
    interactions_json,
)


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def write_csv(path: Path) -> None:
    pd.DataFrame(customers_export_rows()).to_csv(
        path, index=False, float_format="%.2f"
    )


def write_excel(path: Path) -> None:
    customers = pd.DataFrame(customers_export_rows())
    invoices = pd.DataFrame(INVOICES)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        customers.to_excel(writer, sheet_name="customers", index=False)
        invoices.to_excel(writer, sheet_name="invoices", index=False)

    workbook = load_workbook(path)
    notes = workbook.create_sheet("Monthly_notes")
    notes["A1"] = "HarborLink — analyst notes (September 2026)"
    notes["A1"].font = Font(bold=True, size=14)
    notes.merge_cells("A1:D1")
    notes["A3"] = (
        "These notes were typed in a spreadsheet. They are not a clean table."
    )
    notes.merge_cells("A3:D3")
    notes["A5"] = "Follow-up"
    notes["B5"] = "Customer 1827 emailed about an unpaid September invoice."
    notes["A6"] = "Follow-up"
    notes["B6"] = "Customer 1918 complained after a reminder; September is overdue."
    notes["A8"] = "Do not treat this sheet as analysis-ready data."
    notes["A8"].font = Font(italic=True, color="C00000")
    notes.column_dimensions["A"].width = 14
    notes.column_dimensions["B"].width = 78
    for cell in ("A3", "B5", "B6"):
        notes[cell].alignment = Alignment(wrap_text=True)
    workbook.save(path)


def write_json_files(raw_dir: Path) -> None:
    (raw_dir / "interactions.json").write_text(
        json.dumps(interactions_json(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (raw_dir / "api_invoices.json").write_text(
        json.dumps(api_invoices_payload(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _invoice_lines() -> dict:
    bundle = example_invoice()
    invoice = bundle["invoice"]
    customer = bundle["customer"]
    subscription = bundle["subscription"]
    return {
        "title": COMPANY["name"],
        "subtitle": COMPANY["tagline"],
        "invoice_id": invoice["invoice_id"],
        "bill_to": [
            customer["customer_name"],
            customer["company_name"],
            f"Customer ID: {customer['customer_id']}",
            customer["country"],
        ],
        "details": [
            ("Invoice date", invoice["invoice_date"]),
            ("Due date", invoice["due_date"]),
            ("Order", invoice["order_id"]),
            ("Currency", COMPANY["currency"]),
            ("Status", invoice["status"].upper()),
        ],
        "item": (
            f"{subscription['subscription_type'].title()} subscription "
            f"— {invoice['period']}"
        ),
        "amount": f"{invoice['amount']:.2f}",
        "notes": (
            "Payment for the September billing cycle has not been received "
            f"as of {AS_OF_DATE}."
        ),
        "footer": (
            f"{COMPANY['legal_name']}  ·  {COMPANY['address']}  ·  "
            "Confidential customer invoice"
        ),
    }


def write_text_invoice_pdf(path: Path) -> None:
    content = _invoice_lines()
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title=f"{COMPANY['name']} {content['invoice_id']}",
        author=COMPANY["legal_name"],
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "HarborTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.HexColor("#12324d"),
        spaceAfter=2,
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "HarborSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#4a5b6a"),
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "HarborHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=colors.HexColor("#12324d"),
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "HarborBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
    )
    note_style = ParagraphStyle(
        "HarborNote",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        textColor=colors.HexColor("#333333"),
        leading=12,
    )

    header = Table(
        [
            [
                Paragraph(content["title"], title_style),
                Paragraph(
                    f"<para align='right'><b>INVOICE</b><br/>"
                    f"{content['invoice_id']}</para>",
                    body_style,
                ),
            ]
        ],
        colWidths=[110 * mm, 65 * mm],
    )
    header.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ]
        )
    )

    bill_to = "<br/>".join(content["bill_to"])
    details = "<br/>".join(f"<b>{label}:</b> {value}" for label, value in content["details"])
    info = Table(
        [
            [
                Paragraph("<b>Bill to</b>", heading_style),
                Paragraph("<b>Invoice details</b>", heading_style),
            ],
            [
                Paragraph(bill_to, body_style),
                Paragraph(details, body_style),
            ],
        ],
        colWidths=[95 * mm, 80 * mm],
    )
    info.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))

    items = Table(
        [
            ["Description", "Amount (EUR)"],
            [content["item"], content["amount"]],
            ["Total due", content["amount"]],
        ],
        colWidths=[135 * mm, 40 * mm],
    )
    items.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12324d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica"),
                ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#e8eef3")),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aabba")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    def add_page_elements(canvas, _document):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#12324d"))
        canvas.rect(0, A4[1] - 8 * mm, A4[0], 8 * mm, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#4a5b6a"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(18 * mm, 12 * mm, content["footer"])
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, "Page 1 of 1")
        canvas.restoreState()

    document.build(
        [
            header,
            Paragraph(content["subtitle"], subtitle_style),
            Spacer(1, 8),
            info,
            Spacer(1, 18),
            items,
            Spacer(1, 16),
            Paragraph("<b>Notes</b>", heading_style),
            Paragraph(content["notes"], note_style),
        ],
        onFirstPage=add_page_elements,
        onLaterPages=add_page_elements,
    )


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def write_scanned_invoice_pdf(path: Path) -> None:
    """Embed a raster image of the invoice so the PDF contains no real text."""
    content = _invoice_lines()
    width, height = 1240, 1754
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(48)
    label_font = _load_font(28)
    body_font = _load_font(26)
    small_font = _load_font(20)

    draw.rectangle((0, 0, width, 36), fill=(18, 50, 77))
    draw.text((72, 70), content["title"], fill=(18, 50, 77), font=title_font)
    draw.text((72, 140), "INVOICE " + content["invoice_id"], fill=(40, 40, 40), font=label_font)
    draw.text((72, 190), content["subtitle"], fill=(80, 80, 80), font=small_font)

    y = 280
    draw.text((72, y), "Bill to", fill=(18, 50, 77), font=label_font)
    for line in content["bill_to"]:
        y += 40
        draw.text((72, y), line, fill=(30, 30, 30), font=body_font)

    y = 280
    draw.text((720, y), "Invoice details", fill=(18, 50, 77), font=label_font)
    for label, value in content["details"]:
        y += 40
        draw.text((720, y), f"{label}: {value}", fill=(30, 30, 30), font=body_font)

    table_top = 620
    draw.rectangle((72, table_top, width - 72, table_top + 56), fill=(18, 50, 77))
    draw.text((90, table_top + 12), "Description", fill="white", font=label_font)
    draw.text((960, table_top + 12), "Amount", fill="white", font=label_font)
    draw.rectangle((72, table_top + 56, width - 72, table_top + 120), outline=(154, 171, 186))
    draw.text((90, table_top + 74), content["item"], fill=(30, 30, 30), font=body_font)
    draw.text((980, table_top + 74), content["amount"], fill=(30, 30, 30), font=body_font)
    draw.rectangle((72, table_top + 120, width - 72, table_top + 184), fill=(232, 238, 243))
    draw.text((90, table_top + 138), "Total due", fill=(18, 50, 77), font=label_font)
    draw.text((980, table_top + 138), content["amount"], fill=(18, 50, 77), font=label_font)

    draw.text((72, 860), "Notes", fill=(18, 50, 77), font=label_font)
    draw.text((72, 910), content["notes"], fill=(50, 50, 50), font=body_font)
    draw.text((72, 1680), content["footer"], fill=(90, 90, 90), font=small_font)

    image_path = path.with_suffix(".png")
    image.save(image_path, "PNG")

    from reportlab.pdfgen import canvas as pdf_canvas

    pdf = pdf_canvas.Canvas(str(path), pagesize=A4)
    pdf.drawImage(str(image_path), 0, 0, width=A4[0], height=A4[1])
    pdf.setTitle(f"{COMPANY['name']} scanned {content['invoice_id']}")
    pdf.save()
    image_path.unlink()


def generate_all() -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / ".gitkeep").touch()

    written = {
        "customers.csv": RAW_DIR / "customers.csv",
        "customers.xlsx": RAW_DIR / "customers.xlsx",
        "interactions.json": RAW_DIR / "interactions.json",
        "api_invoices.json": RAW_DIR / "api_invoices.json",
        "company.sqlite": RAW_DIR / "company.sqlite",
        "invoice_text.pdf": RAW_DIR / "invoice_text.pdf",
        "invoice_scanned.pdf": RAW_DIR / "invoice_scanned.pdf",
    }
    write_csv(written["customers.csv"])
    write_excel(written["customers.xlsx"])
    write_json_files(RAW_DIR)
    write_sqlite(written["company.sqlite"])
    write_text_invoice_pdf(written["invoice_text.pdf"])
    write_scanned_invoice_pdf(written["invoice_scanned.pdf"])
    return written


def main() -> None:
    written = generate_all()
    print(f"Generated HarborLink files for {len(CUSTOMERS)} customers.")
    for name, path in written.items():
        print(f"  {name:22} {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

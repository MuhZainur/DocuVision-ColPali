"""Generate a realistic 3-page corporate financial PDF document for testing."""

from pathlib import Path
import fitz  # PyMuPDF


def create_sample_financial_pdf(output_path: str | Path = "data/samples/sample_financial_report.pdf") -> str:
    """Create a high-fidelity enterprise financial PDF with tables and headers."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()

    # --- PAGE 1: Executive Overview ---
    p1 = doc.new_page(width=595, height=842)  # A4 standard
    p1.insert_text((50, 60), "ACME ENTERPRISE HOLDINGS INC.", fontsize=18, fontname="helv", color=(0.1, 0.2, 0.5))
    p1.insert_text((50, 82), "Fiscal Year 2024 Annual Strategic & Financial Review", fontsize=12, fontname="helv", color=(0.3, 0.3, 0.3))
    p1.draw_line(fitz.Point(50, 95), fitz.Point(545, 95), color=(0.1, 0.3, 0.7), width=2.0)

    p1.insert_text((50, 130), "1. Executive Strategic Summary", fontsize=14, fontname="helv", color=(0.1, 0.1, 0.1))
    body_p1 = (
        "During fiscal year 2024, ACME Enterprise Holdings demonstrated resilient performance across\n"
        "our AI infrastructure and visual document intelligence portfolio. Consolidated corporate\n"
        "revenue reached $1,680 million, representing a 34.4% year-over-year expansion.\n\n"
        "Key Operational Milestones:\n"
        "• Enterprise SaaS customer retention rate increased to 118% net expansion.\n"
        "• ColPali multimodal retrieval deployment reduced OCR latency by 72% across 500k documents.\n"
        "• Operating margin improved to 29.2% due to operational automation and reduced compute overhead.\n"
        "• Cash and liquidity reserves strengthened to $450 million by end of Q4 2024."
    )
    p1.insert_text((50, 155), body_p1, fontsize=11, fontname="helv", color=(0.2, 0.2, 0.2))

    # --- PAGE 2: Consolidated Statements of Operations (Income Statement Table) ---
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 60), "CONSOLIDATED STATEMENTS OF OPERATIONS", fontsize=16, fontname="helv", color=(0.1, 0.2, 0.5))
    p2.insert_text((50, 82), "(Amounts in Millions of USD, Fiscal Years Ended Dec 31)", fontsize=11, fontname="helv", color=(0.3, 0.3, 0.3))
    p2.draw_line(fitz.Point(50, 95), fitz.Point(545, 95), color=(0.1, 0.3, 0.7), width=2.0)

    # Table layout: (x_line_item=50, x_2023=360, x_2024=460)
    table_data = [
        ("Revenue & Financial Metric", "FY 2023", "FY 2024"),
        ("Total Consolidated Revenue", "$1,250", "$1,680"),
        ("Cost of Goods Sold (COGS)", "$620", "$740"),
        ("Gross Profit", "$630", "$940"),
        ("Research & Development (R&D)", "$180", "$260"),
        ("Sales & Marketing Expenses", "$150", "$190"),
        ("General & Administrative (G&A)", "$100", "$120"),
        ("Total Operating Expenses", "$430", "$570"),
        ("Operating Income (EBIT)", "$300", "$490"),
        ("Interest and Other Income", "$15", "$20"),
        ("Income Before Income Taxes", "$315", "$510"),
        ("Provision for Income Taxes", "$80", "$125"),
        ("Net Income (Consolidated)", "$235", "$385"),
    ]

    y_start = 140
    row_height = 28
    table_width = 495

    # Header background rect
    p2.draw_rect(fitz.Rect(50, y_start, 50 + table_width, y_start + row_height), color=(0.1, 0.3, 0.6), fill=(0.9, 0.93, 0.98))

    for idx, (metric, val23, val24) in enumerate(table_data):
        curr_y = y_start + idx * row_height
        # Alternating background
        if idx > 0 and idx % 2 == 1:
            p2.draw_rect(fitz.Rect(50, curr_y, 50 + table_width, curr_y + row_height), fill=(0.97, 0.97, 0.99))
        if idx == len(table_data) - 1:  # Net Income row highlighted
            p2.draw_rect(fitz.Rect(50, curr_y, 50 + table_width, curr_y + row_height), fill=(0.93, 0.98, 0.93), color=(0.2, 0.6, 0.2))

        # Text
        fname = "hebo" if idx in (0, 3, 8, len(table_data) - 1) else "helv"
        p2.insert_text((60, curr_y + 19), metric, fontsize=10.5, fontname=fname, color=(0.1, 0.1, 0.1))
        p2.insert_text((370, curr_y + 19), val23, fontsize=10.5, fontname=fname, color=(0.1, 0.1, 0.1))
        p2.insert_text((470, curr_y + 19), val24, fontsize=10.5, fontname=fname, color=(0.1, 0.1, 0.1))

        # Horizontal divider line
        p2.draw_line(fitz.Point(50, curr_y + row_height), fitz.Point(545, curr_y + row_height), color=(0.85, 0.85, 0.85), width=0.5)

    # Outer border
    p2.draw_rect(fitz.Rect(50, y_start, 50 + table_width, y_start + len(table_data) * row_height), color=(0.7, 0.7, 0.7), width=1.0)

    # --- PAGE 3: Consolidated Balance Sheets ---
    p3 = doc.new_page(width=595, height=842)
    p3.insert_text((50, 60), "CONSOLIDATED BALANCE SHEETS - ASSETS & LIABILITIES", fontsize=16, fontname="helv", color=(0.1, 0.2, 0.5))
    p3.insert_text((50, 82), "(As of December 31, 2024 and 2023, in Millions USD)", fontsize=11, fontname="helv", color=(0.3, 0.3, 0.3))
    p3.draw_line(fitz.Point(50, 95), fitz.Point(545, 95), color=(0.1, 0.3, 0.7), width=2.0)

    bs_data = [
        ("Balance Sheet Item / Asset Class", "FY 2023", "FY 2024"),
        ("Cash and Cash Equivalents", "$320", "$450"),
        ("Marketable Short-term Securities", "$150", "$210"),
        ("Accounts Receivable, net", "$230", "$260"),
        ("Total Current Assets", "$700", "$920"),
        ("Property, Plant and Equipment (PPE)", "$550", "$680"),
        ("Goodwill and Intangible Assets", "$330", "$470"),
        ("Total Assets", "$1,580", "$2,070"),
        ("Current Accounts Payable", "$180", "$210"),
        ("Long-Term Term Loan Debt", "$420", "$480"),
        ("Total Stockholders' Equity", "$980", "$1,380"),
        ("Total Liabilities & Equity", "$1,580", "$2,070"),
    ]

    y_bs = 140
    p3.draw_rect(fitz.Rect(50, y_bs, 50 + table_width, y_bs + row_height), color=(0.1, 0.3, 0.6), fill=(0.9, 0.93, 0.98))

    for idx, (item, v23, v24) in enumerate(bs_data):
        curr_y = y_bs + idx * row_height
        if idx > 0 and idx % 2 == 1:
            p3.draw_rect(fitz.Rect(50, curr_y, 50 + table_width, curr_y + row_height), fill=(0.97, 0.97, 0.99))
        if idx in (4, 7, len(bs_data) - 1):
            p3.draw_rect(fitz.Rect(50, curr_y, 50 + table_width, curr_y + row_height), fill=(0.93, 0.95, 0.99), color=(0.2, 0.4, 0.7))

        fname = "hebo" if idx in (0, 4, 7, 10, len(bs_data) - 1) else "helv"
        p3.insert_text((60, curr_y + 19), item, fontsize=10.5, fontname=fname, color=(0.1, 0.1, 0.1))
        p3.insert_text((370, curr_y + 19), v23, fontsize=10.5, fontname=fname, color=(0.1, 0.1, 0.1))
        p3.insert_text((470, curr_y + 19), v24, fontsize=10.5, fontname=fname, color=(0.1, 0.1, 0.1))
        p3.draw_line(fitz.Point(50, curr_y + row_height), fitz.Point(545, curr_y + row_height), color=(0.85, 0.85, 0.85), width=0.5)

    p3.draw_rect(fitz.Rect(50, y_bs, 50 + table_width, y_bs + len(bs_data) * row_height), color=(0.7, 0.7, 0.7), width=1.0)

    doc.save(str(p.resolve()))
    doc.close()
    return str(p.resolve())


if __name__ == "__main__":
    out = create_sample_financial_pdf()
    print(f"Sample PDF created successfully at: {out}")

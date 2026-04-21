"""Export helpers (PDF/HTML) for chat history."""

from io import BytesIO
from typing import Any, Dict, List
import html

from .formatting import _format_duration, _make_excerpt, _unify_metadata

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False


def _build_pdf_reportlab(history: List[Dict[str, Any]]) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not available")

    buff = BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4)
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("IFRS Assistant Chat Export", styles["Title"]))
    elems.append(Spacer(1, 12))

    for idx, entry in enumerate(history, start=1):
        q = html.escape((entry.get("question") or "").strip())
        a = html.escape((entry.get("answer") or "").strip())
        t = _format_duration(entry.get("time_taken_sec", 0))

        elems.append(Paragraph(f"Q{idx}: {q}", styles["Heading3"]))
        elems.append(Paragraph(a.replace("\n", "<br/>"), styles["BodyText"]))
        elems.append(Paragraph(f"Duration: {t}", styles["Italic"]))

        for src in entry.get("sources", []) or []:
            meta = _unify_metadata(getattr(src, "metadata", {}) or {})
            title = html.escape(meta.get("title", "Reference"))
            excerpt = html.escape(_make_excerpt(getattr(src, "page_content", "") or "", max_chars=300))
            elems.append(Paragraph(f"- {title}", styles["BodyText"]))
            elems.append(Paragraph(f"  {excerpt}", styles["BodyText"]))

        elems.append(Spacer(1, 14))

    doc.build(elems)
    return buff.getvalue()


def _build_pdf_fpdf(history: List[Dict[str, Any]]) -> bytes:
    if not FPDF_AVAILABLE:
        raise RuntimeError("FPDF is not available")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "IFRS Assistant Chat Export", ln=1)

    for idx, entry in enumerate(history, start=1):
        q = (entry.get("question") or "").strip()
        a = (entry.get("answer") or "").strip()
        t = _format_duration(entry.get("time_taken_sec", 0))

        pdf.set_font("Arial", "B", 11)
        pdf.multi_cell(0, 8, f"Q{idx}: {q}")
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6, a)
        pdf.set_font("Arial", "I", 9)
        pdf.multi_cell(0, 6, f"Duration: {t}")
        pdf.ln(2)

    out = pdf.output(dest="S")
    if isinstance(out, bytes):
        return out
    return out.encode("latin-1", errors="ignore")


def _df_to_html_table(columns: List[str], rows: List[List[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body_rows = []
    for row in rows:
        tds = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        body_rows.append(f"<tr>{tds}</tr>")
    body = "".join(body_rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _build_html_export(history: List[Dict[str, Any]]) -> str:
    blocks = []
    blocks.append("<h1>IFRS Assistant Chat Export</h1>")

    for idx, entry in enumerate(history, start=1):
        q = html.escape((entry.get("question") or "").strip())
        a = html.escape((entry.get("answer") or "").strip()).replace("\n", "<br/>")
        t = html.escape(_format_duration(entry.get("time_taken_sec", 0)))

        blocks.append(f"<h3>Q{idx}: {q}</h3>")
        blocks.append(f"<p>{a}</p>")
        blocks.append(f"<p><em>Duration: {t}</em></p>")

        for table in entry.get("tables", []) or []:
            cols = table.get("columns") or []
            rows = table.get("rows") or []
            if cols and rows:
                blocks.append(_df_to_html_table(cols, rows))

    style = (
        "<style>body{font-family:Arial,sans-serif;padding:24px;}"
        "table{border-collapse:collapse;margin:8px 0 16px;}"
        "th,td{border:1px solid #ccc;padding:6px 8px;font-size:12px;}"
        "h1,h3{color:#1f2937;}</style>"
    )
    return f"<!doctype html><html><head><meta charset='utf-8'>{style}</head><body>{''.join(blocks)}</body></html>"

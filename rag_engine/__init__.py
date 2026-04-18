"""Public module surface for rag_engine.

Avoid wildcard-importing `engine.py` at package import time so downstream tools can
import focused modules without pulling the full pipeline and heavyweight deps.
"""

from .answer import (
    answer_with_refine_chain,
    build_confidence_result,
    classify_confidence,
    generate_unified_reference_list,
    retrieve_and_generate_exceptions,
)
from .exports import (
    FPDF_AVAILABLE,
    REPORTLAB_AVAILABLE,
    _build_html_export,
    _build_pdf_fpdf,
    _build_pdf_reportlab,
)
from .formatting import (
    _format_duration,
    _make_excerpt,
    _unify_metadata,
    bold_standards,
    emphasize_headers,
    fix_citation_format,
    format_visible_answer,
    remove_citations,
    sanitize_text,
)
from .retrieval import DBConfig, DBS, build_retriever, fetch_docs, load_index, retrieve_docs_with_score
from .translate import translate_to_arabic

__all__ = [
    "answer_with_refine_chain",
    "build_confidence_result",
    "classify_confidence",
    "generate_unified_reference_list",
    "retrieve_and_generate_exceptions",
    "FPDF_AVAILABLE",
    "REPORTLAB_AVAILABLE",
    "_build_html_export",
    "_build_pdf_fpdf",
    "_build_pdf_reportlab",
    "_format_duration",
    "_make_excerpt",
    "_unify_metadata",
    "bold_standards",
    "emphasize_headers",
    "fix_citation_format",
    "format_visible_answer",
    "remove_citations",
    "sanitize_text",
    "DBConfig",
    "DBS",
    "build_retriever",
    "fetch_docs",
    "load_index",
    "retrieve_docs_with_score",
    "translate_to_arabic",
]

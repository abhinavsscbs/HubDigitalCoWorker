# === IFRS RAG Engine — Pure RAG Logic (No Streamlit Dependencies) ===
# Core RAG functionality for IFRS chatbot:
# - Five FAISS DBs in fixed order: IFRS A -> IFRS B -> IFRS C -> EY -> PwC
# - LLM-based document filtering with score thresholds (0.65)
# - Answer generation with citations and source tracking
# - PDF export utilities (ReportLab preferred, FPDF fallback)
# - Translation and text formatting utilities
#
# This module is shared by both Streamlit UI and Flask backend.

import os
import sys
import re
import time
import random
import unicodedata

# Force unbuffered output for real-time logging
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Tuple, Union, Any, Optional
import datetime
import uuid
import json
import html
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

# Updated LangChain imports (v1.0 compatible)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from rag_config import (
    DB_PATHS,
    REPLACE_EXCEL_PATH,
    STAGE_1_THRESHOLD,
    STAGE_2_PERCENTILE,
    EMBEDDINGS_MODEL,
    EMBEDDINGS_DEVICE,
    RAG_SEED,
)
from llm_client import get_llm
from rag_engine.tables import (
    _canonicalize_all_tables,
    _drop_leading_empty_column,
    _find_markdown_tables,
    _md_table_to_df,
    _mk_pipe_row,
    _mk_pipe_sep,
    _normalize_markdown_tables,
    _normalize_tables_payload,
    _split_answer_and_json,
    _split_pipe_row,
    _strip_markdown_tables_from_text,
    extract_markdown_tables_as_dfs,
)

# --- PDF backends (ReportLab preferred, FPDF fallback) ---
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False


# ------------------- CONFIG -------------------

# ==================== STAGE CONFIGURATION ====================
# Values come from rag_config.py (env-overridable).

@dataclass
class DBConfig:
    path: str
    name: str
    score_threshold: float = 0.65  # minimum similarity score (for backward compatibility)
    top_percentile: float = STAGE_2_PERCENTILE  # Uses global config

# Load env for config overrides
load_dotenv()

# Embeddings model
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS_MODEL,
    model_kwargs={'device': EMBEDDINGS_DEVICE}
)

# LLMs (Cohere-style on-prem endpoint)
LLM_MINI = get_llm("mini")
LLM_FULL = get_llm("full")
LLM = LLM_MINI

# Paths to your FAISS indexes (must contain LangChain FAISS files)
# Using score_threshold instead of fixed k - only documents with similarity >= threshold are retrieved
DBS: List[DBConfig] = [
    DBConfig(path=item["path"], name=item["name"], score_threshold=item["score_threshold"])
    for item in DB_PATHS
]

# Seed
random.seed(RAG_SEED)


# ------------------- FAISS / RETRIEVAL -------------------

def _has_langchain_index(dir_path: str) -> bool:
    return os.path.exists(os.path.join(dir_path, "index.faiss")) or \
           os.path.exists(os.path.join(dir_path, "index.pkl"))

def load_index(dir_path: str) -> FAISS:
    if not _has_langchain_index(dir_path):
        raise RuntimeError(f" {dir_path} is not a LangChain FAISS index (index.faiss/index.pkl missing).")
    return FAISS.load_local(dir_path, EMBEDDINGS, allow_dangerous_deserialization=True)

def build_retriever(dir_path: str, k: int = 5):
    vs = load_index(dir_path)
    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "fetch_k": max(40, 5*k)}
    )

def retrieve_docs_with_score(
    dir_path: str,
    question: str,
    score_threshold: Optional[float] = None,
    top_percentile: Optional[float] = None,  # Uses STAGE_2_PERCENTILE if not specified
    max_k: int = 50
) -> List[Tuple[Document, float]]:
    """
    Retrieve documents using similarity search with score filtering.
    Supports two modes: fixed threshold or percentile-based selection.

    IMPORTANT: FAISS uses L2 distance internally (LOWER = more similar).
    This function converts L2 distance to cosine similarity (HIGHER = more similar).

    Args:
        dir_path: Path to FAISS index
        question: Query string
        score_threshold: Minimum cosine similarity score (fixed threshold mode)
                        If provided, filters docs with score >= threshold
        top_percentile: Top X% of documents to select (percentile mode)
                       If None, uses STAGE_2_PERCENTILE global config
                       If provided and score_threshold is None, selects top X% by score
        max_k: Maximum number of documents to fetch initially

    Returns:
        List of (Document, cosine_similarity) tuples, filtered by threshold or percentile
        cosine_similarity is in 0-1 range where 1 = identical, 0 = orthogonal
    """
    vs = load_index(dir_path)
    # Fetch more documents initially to ensure we get enough after filtering
    docs_with_scores = vs.similarity_search_with_score(question, k=max_k)

    # Convert L2 distance to cosine similarity
    # For normalized vectors: cosine_similarity = 1 - (L2_distance² / 2)
    # This is the mathematically correct conversion formula
    docs_with_similarity = []
    for doc, l2_distance in docs_with_scores:
        # Proper cosine similarity conversion from L2 distance on normalized vectors
        cosine_similarity = 1 - (l2_distance ** 2 / 2)
        cosine_similarity = max(0, min(1, cosine_similarity))  # Clamp to [0, 1]
        docs_with_similarity.append((doc, cosine_similarity))

    # Selection logic: percentile mode (default) or threshold mode
    # If top_percentile not specified, use global config
    if top_percentile is None and score_threshold is None:
        top_percentile = STAGE_2_PERCENTILE

    if top_percentile is not None and score_threshold is None:
        # Percentile-based selection: sort by score and take top X%
        sorted_docs = sorted(docs_with_similarity, key=lambda x: x[1], reverse=True)
        cutoff_index = max(1, int(len(sorted_docs) * top_percentile))
        filtered = sorted_docs[:cutoff_index]
        # Logging removed - will be done at DB config level
        return filtered
    elif score_threshold is not None:
        # Fixed threshold mode (backward compatibility)
        filtered = [(doc, score) for doc, score in docs_with_similarity if score >= score_threshold]
        # Logging removed - will be done at DB config level
        return filtered
    else:
        # Default: return all docs
        return docs_with_similarity

# --- Round-bracket source tag helper that the LLM can quote in answers ---
# def _source_tag(meta: dict, db_label: str) -> str:
#     if db_label=="EY" or "PwC":
#         doc = (meta or {}).get("chapter_name") or (meta or {}).get("source") or "Document"
#     else:
#         doc = (meta or {}).get("chapter") or (meta or {}).get("source") or "Document"
#     # Get paragraph number from metadata (NEW - accurate from extraction)
#     para_num = (meta or {}).get("para_number")
    
#     # Build tag with paragraph number if available
#     if para_num and str(para_num).strip():
#         return f"({db_label} — {doc} — para {para_num})"
#     else:
#         return f"({db_label} — {doc})"

# def _source_tag(meta: dict, db_label: str) -> str:
#     """Build citation tag that gets prepended to chunks."""
#     # Get paragraph number from metadata
#     para_num = (meta or {}).get("para_number")
    
#     # For IFRS A/B/C: use chapter field (e.g., ifrs-16-leases)
#     if db_label in ["IFRS A", "IFRS B", "IFRS C"]:
#         chapter = (meta or {}).get("chapter") or (meta or {}).get("source") or "Document"
#         if para_num and str(para_num).strip() and str(para_num).lower() not in {"none", "unnumbered"}:
#             return f"({db_label} - {chapter} - para {para_num})"
#         else:
#             return f"({db_label} - {chapter})"
    
#     # For EY/PwC: use chapter_name + header (e.g., "15 - Leases (IFRS 16) - Accounting by lessees")
#     else:
#         chapter_name = (meta or {}).get("chapter_name") or (meta or {}).get("source") or "Document"
#         header = (meta or {}).get("header", "")
        
#         if header:
#             # Full format with header section
#             if para_num and str(para_num).strip() and str(para_num).lower() not in {"none", "unnumbered"}:
#                 return f"({db_label} - {chapter_name} - {header} - para {para_num})"
#             else:
#                 return f"({db_label} - {chapter_name} - {header})"
#         else:
#             # Just chapter name
#             if para_num and str(para_num).strip() and str(para_num).lower() not in {"none", "unnumbered"}:
#                 return f"({db_label} - {chapter_name} - para {para_num})"
#             else:
#                 return f"({db_label} - {chapter_name})"

def _source_tag(meta: dict, db_label: str) -> str:
    """Build citation tag that gets prepended to chunks."""
    # Get paragraph number from metadata
    para_num = (meta or {}).get("para_number")
    
    # For IFRS A/B/C: chapter_name contains the standard name (e.g., ias-38-intangible-assets)
    if db_label in ["IFRS A", "IFRS B", "IFRS C"]:
        standard_name = (meta or {}).get("chapter_name") or (meta or {}).get("chapter") or (meta or {}).get("source") or "Document"
        
        if para_num and str(para_num).strip() and str(para_num).lower() not in {"none", "unnumbered"}:
            return f"({db_label} - {standard_name} - para {para_num})"
        else:
            return f"({db_label} - {standard_name})"
    
    # For EY/PwC: chapter_name contains the chapter name (e.g., "18 Intangible assets")
    else:
        chapter_name = (meta or {}).get("chapter_name") or (meta or {}).get("source") or "Document"
        header = (meta or {}).get("header", "")
        
        if header:
            # Full format with header section
            if para_num and str(para_num).strip() and str(para_num).lower() not in {"none", "unnumbered"}:
                return f"({db_label} - {chapter_name} - {header} - para {para_num})"
            else:
                return f"({db_label} - {chapter_name} - {header})"
        else:
            # Just chapter name
            if para_num and str(para_num).strip() and str(para_num).lower() not in {"none", "unnumbered"}:
                return f"({db_label} - {chapter_name} - para {para_num})"
            else:
                return f"({db_label} - {chapter_name})"

# Old fetch_docs definition removed - using score-based version defined later in the file


# ===================== SANITIZATION (PDF-safe text) =====================

_SANITIZE_MAP = {
    "\u25a0": "-",   # black square ■
    "\u25aa": "-",   # small black square ▪
    "\u25cf": "-",   # black circle ●
    "\u2022": "-",   # bullet •
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2212": "-",   # minus sign
    "\u00ad": "-",   # soft hyphen (CRITICAL - often invisible, breaks PDF)
    "\u2010": "-",   # hyphen
    "\u2011": "-",   # non-breaking hyphen
    "\u2012": "-",   # figure dash
    "\u2015": "-",   # horizontal bar
    "\ufe58": "-",   # small em dash
    "\ufe63": "-",   # small hyphen-minus
    "\u00a0": " ",   # NBSP
    "\u200b": "",    # zero-width space
    "\ufeff": "",    # BOM
}

def sanitize_text(text: str) -> str:
    """
    Sanitize text for PDF export by normalizing Unicode and replacing problematic characters.

    Handles:
    - Unicode normalization (NFKC) to standardize character representations
    - Combining diacritical marks removal (e.g., U+0336 COMBINING LONG STROKE OVERLAY)
    - All hyphen variants → standard ASCII hyphen
    - Curly quotes → straight quotes
    - Multiple spaces → single space
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Normalize Unicode to canonical composed form (NFKC)
    # This converts variant forms to their canonical equivalents
    s = unicodedata.normalize('NFKC', text)

    # Step 2: Remove combining diacritical marks (Unicode category 'Mn' - Mark, Nonspacing)
    # This fixes the 'n̶' (n + U+0336) → 'n' issue
    s = ''.join(char for char in s if unicodedata.category(char) != 'Mn')

    # Step 3: Replace special characters using the sanitization map
    for k, v in _SANITIZE_MAP.items():
        s = s.replace(k, v)

    # Step 4: Replace curly quotes with straight quotes
    s = s.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")

    # Step 5: Collapse multiple spaces/tabs into single space
    s = re.sub(r"[ \t]+", " ", s)

    # Step 6: Remove markdown code fence markers
    s = s.replace("```", "")

    return s


## Table parsing moved to rag_engine.tables

# ======================== SUPERFICIAL FIX: Filter Page=0 from Display ========================

def filter_page_zero_references(sources: List[Any]) -> List[Any]:
    """
    SUPERFICIAL FIX: Remove chunks with page=0 from display only.
    Answer generation already used all chunks - this is purely cosmetic.
    
    Args:
        sources: List of Document objects with metadata
    
    Returns:
        Filtered list with page > 0 only
    """
    if not sources:
        return sources
    
    filtered = []
    removed_count = 0
    
    for doc in sources:
        meta = getattr(doc, "metadata", {}) or {}
        page = meta.get("page", 0)
        
        # Skip chunks with invalid page numbers
        if page and page > 0:
            filtered.append(doc)
        else:
            removed_count += 1
            source_db = meta.get("source_db", "Unknown")
            chapter = meta.get("chapter_name") or meta.get("chapter", "")
            print(f"   [DISPLAY FILTER] Removed {source_db} - {chapter} (page={page})")
    
    if removed_count > 0:
        print(f"\n[DISPLAY FILTER] ðŸ§¹ Removed {removed_count} chunks with page=0 from display")
    
    return filtered

#------------------ Reference normalization -----------------------
# ===================== NEW: Paragraph Extractor (LLM #1) =====================

import re
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain


# PARA_ID_RE = re.compile(
#     r"\b(?:(?:para(?:graph)?s?\s*)?)"
#     r"("
#     r"(?:IFRS\s+[A-C])"
#     r"|(?:EY)"
#     r"|(?:PWC)"
#     r"|(?:[A-Z]{0,2}\d+(?:\.\d+){0,3})"
#     r")"
#     r"(?:\s*[-–—]\s*(?:[A-Z]{0,2}\d+(?:\.\d+){0,3}))?"
#     r"\b",
#     re.IGNORECASE
# )

PARA_ID_RE = re.compile(
    r"\b(?:(?:para(?:graph)?s?\s*)?)"
    r"("

    # Existing matches
    r"(?:IFRS\s+[A-C])"
    r"|(?:EY)"
    r"|(?:PWC)"
    r"|(?:[A-Z]{0,2}\d+(?:\.\d+){0,3})"        # 5.3, B5.4.1, 26(d)

    #  New matches
    r"|(?:\d+(?:\([a-z]\)))"                   # 26(d), 14(a)
    r"|(?:IG\s+Example\s+\d+)"                 # IG Example 12
    r"|(?:Illustration\s+\d+(?:-\d+)*)"        # Illustration 3-8
    r"|(?:Example\s+\d+)"                      # Example 12
    r"|(?:Appendix\s+[A-Z])"                   # Appendix A, B, etc.

    r")"
    r"(?:\s*[-–—]\s*(?:[A-Z]{0,2}\d+(?:\.\d+){0,3}))?"
    r"\b",
    re.IGNORECASE
)


DASHES = r"[-–—]"

def _normalize_dashes(s: str) -> str:
    return re.sub(DASHES, "-", s or "")

def _find_para_ids_in_text(text: str) -> List[str]:
    text = _normalize_dashes(text or "")
    return sorted({m.group(1) for m in PARA_ID_RE.finditer(text)})

_SOURCE_KEYS = ("db", "source_db", "index", "vectorstore", "vs", "corpus", "dataset", "collection")

def _detect_source_name(meta: Dict[str, Any]) -> str:
    for k in _SOURCE_KEYS:
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v
    for k in ("source", "document_id", "file"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return f"src:{v}"
    return "default"

def _group_docs_by_source(docs: List[Any]) -> Dict[str, List[Any]]:
    groups: Dict[str, List[Any]] = defaultdict(list)
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        src = _detect_source_name(meta)
        groups[src].append(d)
    return groups

# def _collect_seen_paras_from_docs(docs: List[Any]) -> List[str]:
#     seen = set()
#     for d in docs or []:
#         txt = _normalize_dashes(getattr(d, "page_content", "") or "")
#         for pid in PARA_ID_RE.findall(txt):
#             seen.add(pid)
#         meta = getattr(d, "metadata", {}) or {}
#         for k in ("para_id", "paragraph", "paragraph_ids", "paras"):
#             val = meta.get(k)
#             if isinstance(val, str):
#                 for pid in PARA_ID_RE.findall(_normalize_dashes(val)):
#                     seen.add(pid)
#             elif isinstance(val, (list, tuple)):
#                 for item in val:
#                     if isinstance(item, str):
#                         for pid in PARA_ID_RE.findall(_normalize_dashes(item)):
#                             seen.add(pid)
#     return sorted(seen)

def _collect_seen_paras_from_metadata(docs: List[Any]) -> List[str]:
    """
    Collect paragraph numbers directly from document metadata.
    No more regex scraping - use accurate para_number field.
    """
    seen = set()
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        
        # Primary source: para_number field (accurate from extraction)
        para_num = meta.get("para_number")
        if para_num and str(para_num).strip() and str(para_num).lower() not in {"none", "unnumbered"}:
            seen.add(str(para_num).strip())
        
        # Also keep chapter-level fallback for documents without para numbers
        chapter = meta.get("chapter_name") or meta.get("chapter") or ""
        source_db = meta.get("source_db") or ""
        if chapter:
            chapter_key = f"{source_db} — {chapter}"
            seen.add(chapter_key)
    
    return sorted(seen)

def _format_docs_full(docs: List[Any], max_docs: int = 10) -> str:
    if not docs:
        return "(no snippets)"
    lines: List[str] = []
    for i, d in enumerate(docs[:max_docs], start=1):
        text = (getattr(d, "page_content", "") or "").strip()
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source") or meta.get("document_id") or meta.get("file") or ""
        tag = f"(source={src})" if src else ""
        lines.append(f"{i}. {tag}\n{text}")
    return "\n\n".join(lines)

def _format_docs_round_robin(docs: List[Any], total_max: int = 20, per_source_cap: int = 5) -> str:
    if not docs:
        return "(no snippets)"
    groups = _group_docs_by_source(docs)
    sources = sorted(groups.keys())
    trimmed = {s: groups[s][:per_source_cap] for s in sources}
    ordered: List[Any] = []
    idx = 0
    while len(ordered) < total_max:
        progressed = False
        for s in sources:
            if idx < len(trimmed[s]):
                ordered.append(trimmed[s][idx]); progressed = True
                if len(ordered) >= total_max: break
        if not progressed: break
        idx += 1
    out = []
    for i, d in enumerate(ordered, start=1):
        txt = (getattr(d, "page_content", "") or "").strip()
        meta = getattr(d, "metadata", {}) or {}
        src = _detect_source_name(meta)
        tag = f"(source={src})"
        src2 = meta.get("source") or meta.get("document_id") or meta.get("file") or ""
        if src2:
            tag += f" (ref={src2})"
        out.append(f"{i}. {tag}\n{txt}")
    return "\n\n".join(out)

def _coerce_number(x: str):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "n/a", "none", "-"}:
        return np.nan
    s = s.replace(",", "")
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return np.nan
    if s.startswith("(") and s.endswith(")"):
        try:
            return -float(s[1:-1])
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def _try_coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        vals = out[col].astype(str).tolist()
        parsed = [_coerce_number(v) for v in vals]
        non_nan = [v for v in parsed if not (isinstance(v, float) and np.isnan(v))]
        if len(non_nan) >= max(1, 0.5 * len(vals)):
            out[col] = pd.Series(parsed)
    return out

LLM_EXTRACTOR = get_llm("extractor")

# _EXTRACTOR_PROMPT = (
#     "You are a meticulous IFRS/Accounting paragraph selector. Output STRICT JSON only.\n\n"
#     "You receive:\n"
#     "1) The user's exact question.\n"
#     "2) A list of paragraph IDs and accounting references that actually appear in the provided snippets (Seen_Paras).\n"
#     "   These may include:\n"
#     "   - Pure IDs (e.g., 6.5.8, B5.4.1)\n"
#     "   - IDs with prefixes/suffixes (e.g., IFRS 6.5.8, IAS 2.1, paragraphs 5.4.1, 6.5.8 EY)\n"
#     "   - Standard references (IFRS, IAS, IFRIC, paragraphs, IFRS A/B/C, EY, PwC)\n"
#     "3) The raw snippets text (Context) for reference.\n\n"
#     "Rules:\n"
#     "- Select ONLY paragraph IDs or references that are present in Seen_Paras.\n"
#     "- If the question includes a range like “paragraphs 6.5.8–6.5.14” or “B5.4.1-B5.4.7”, "
#     "  include all IDs in that span, BUT only if each exists in Seen_Paras.\n"
#     "- If a prefix/suffix term (e.g., IFRS, IAS, EY, PwC) appears with a paragraph ID, treat it as one reference (e.g., 'IFRS 6.5.8').\n"
#     "- If the question refers only to a standard (e.g., 'IAS 16' or 'IFRS B'), include it if it exists in Seen_Paras.\n"
#     "- Do NOT invent IDs or terms not in Seen_Paras.\n"
#     "- Output STRICT JSON with this schema and nothing else:\n"
#     "{{\n"
#     '  "paragraph_ids": ["<ID1>", "<ID2>", "..."],\n'
#     '  "reason": "one short sentence"\n'
#     "}}\n\n"
#     "Question:\n{question}\n\n"
#     "Seen_Paras:\n{seen_paras}\n\n"
#     "Context:\n{context}\n"
# )

# def _format_docs_with_ids(docs: List[Any], max_docs: int = 20) -> str:
#     lines: List[str] = []
#     for i, d in enumerate(docs[:max_docs], start=1):
#         text = (getattr(d, "page_content", "") or "").strip()
#         doc_id = (getattr(d, "metadata", {}) or {}).get("_doc_id", "unknown")
#         lines.append(f"{i}. <DOC_ID:{doc_id}>\n{text}")
#     return "\n\n".join(lines)

def _format_docs_with_ids(docs: List[Any], max_docs: int = 40) -> str:
    """
    Format documents with their unique IDs for LLM context.
    Uses round-robin sampling to ensure representation from all source DBs.
    """
    if not docs:
        return "(no snippets)"
    
    # Group docs by source DB
    by_db = {}
    for d in docs:
        db = (d.metadata or {}).get("source_db", "Unknown")
        if db not in by_db:
            by_db[db] = []
        by_db[db].append(d)
    
    # Round-robin selection to get representation from all DBs
    selected = []
    db_names = list(by_db.keys())
    idx = 0
    
    while len(selected) < max_docs and any(by_db.values()):
        db = db_names[idx % len(db_names)]
        if by_db[db]:
            selected.append(by_db[db].pop(0))
        idx += 1
    
    # Format with doc IDs
    lines: List[str] = []
    for i, d in enumerate(selected, start=1):
        text = (getattr(d, "page_content", "") or "").strip()
        meta = d.metadata or {}
        doc_id = meta.get("_doc_id", "unknown")
        source_db = meta.get("source_db", "")
        para_num = meta.get("para_number", "")
        chapter = meta.get("chapter_name") or meta.get("chapter", "")
        
        # Include metadata in the snippet for better context
        meta_info = f"[{source_db}"
        if para_num:
            meta_info += f" | para {para_num}"
        if chapter:
            meta_info += f" | {chapter[:50]}"  # Truncate long chapter names
        meta_info += "]"
        
        lines.append(f"{i}. <DOC_ID:{doc_id}> {meta_info}\n{text[:300]}...")  # Show first 300 chars
    
    return "\n\n".join(lines)

# def _format_docs_with_ids(docs: List[Any], max_docs: int = 20) -> str:
#     """Format documents with their unique IDs for LLM context."""
#     if not docs:
#         return "(no snippets)"
    
#     lines: List[str] = []
#     for i, d in enumerate(docs[:max_docs], start=1):
#         text = (getattr(d, "page_content", "") or "").strip()
#         doc_id = (getattr(d, "metadata", {}) or {}).get("_doc_id", "unknown")
#         lines.append(f"{i}. <DOC_ID:{doc_id}>\n{text}")
#     return "\n\n".join(lines)

# def _filter_docs_by_ids(docs: List[Any], used_ids: List[str]) -> List[Any]:
#     """Filter docs by their unique IDs."""
#     print(f"\n{'~'*80}")
#     print(f"FILTERING: Checking {len(docs)} docs against {len(used_ids)} selected IDs")
#     if not used_ids:
#         return []
    
#     used_ids_set = set(used_ids)
#     return [
#         d for d in docs 
#         if (d.metadata or {}).get("_doc_id") in used_ids_set
#     ]

def  _filter_docs_by_ids(docs: List[Any], used_ids: List[str]) -> List[Any]:
    """Filter docs by their unique IDs (simplified logging)."""
    if not used_ids:
        return []

    used_ids_set = set(used_ids)
    kept = []

    for d in docs:
        doc_id = (d.metadata or {}).get("_doc_id")
        if doc_id in used_ids_set:
            kept.append(d)

    return kept

# _EXTRACTOR_PROMPT = (
#     "You are an IFRS reference extractor. Output STRICT JSON only.\n\n"
#     "You receive:\n"
#     "1) The user's question\n"
#     "2) Context snippets, each starting with a tag like (IFRS A – Document – para X) <DOC_ID:abc123>\n\n"
#     "Rules:\n"
#     "- Extract the DOC_ID values (like abc123) for ONLY the snippets you would cite in your answer\n"
#     "- Output format:\n"
#     "{\n"
#     '  "doc_ids": ["abc123", "def456", ...],\n'
#     '  "reason": "one short sentence"\n'
#     "}\n\n"
#     "Question:\n{question}\n\n"
#     "Context:\n{context}\n"
# )

# _EXTRACTOR_PROMPT = (
#     "You are an IFRS reference extractor. Output STRICT JSON only.\n\n"
#     "You receive:\n"
#     "1) The user's question\n"
#     "2) Context snippets, each prefixed with <DOC_ID:abc123> format\n\n"
#     "Rules:\n"
#     "- Extract ONLY the DOC_ID values (like abc123, def456, etc.) for the snippets you would cite in your answer\n"
#     "- Select snippets that directly answer the user's question\n"
#     "- Output format:\n"
#     "{{\n"
#     '  "doc_ids": ["abc123", "def456"],\n'
#     '  "reason": "one short sentence explaining your selection"\n'
#     "}}\n\n"
#     "Question:\n{question}\n\n"
#     "Context:\n{context}\n"
# )

# _EXTRACTOR_PROMPT = (
#     "You are a meticulous IFRS/Accounting paragraph selector. Output STRICT JSON only.\n\n"
#     "You receive:\n"
#     "1) The user's question\n"
#     "2) Context snippets from multiple sources (IFRS A/B/C, EY, PwC), each with <DOC_ID:abc123> format\n\n"
#     "Your task:\n"
#     "- Select ALL snippets that contain information RELEVANT to answering the question\n"
#     "- Include snippets from DIFFERENT sources (IFRS A, B, C, EY, PwC) if they provide relevant perspectives\n"
#     "- Be INCLUSIVE rather than selective - when in doubt, include it\n"
#     "- A snippet is relevant if it discusses concepts, definitions, requirements, examples, or guidance related to the question\n\n"
#     "Output format:\n"
#     "{{\n"
#     '  "doc_ids": ["abc123", "def456", "xyz789", ...],\n'
#     '  "reason": "brief explanation of what information these snippets provide"\n'
#     "}}\n\n"
#     "Question:\n{question}\n\n"
#     "Context:\n{context}\n"
# )

from langchain_core.messages import SystemMessage, HumanMessage
import json
import difflib

# Use your existing LLM (temperature=0) for deterministic behavior
# If you prefer a cheaper/faster model, pass it as `llm=` argument.

_CITATION_KEYWORDS_EXAMPLE = [
    "PwC", "EY",
    "IFRS"]

def _build_strip_prompt(text: str, keywords: list = None) -> str:
    keys = keywords or _CITATION_KEYWORDS_EXAMPLE
    # join into single-line for inclusion
    kw_line = ", ".join(keys)
    return (
        "You are a precise text transformer. You will RECEIVE an exact text and MUST RETURN the same text\n"
        "but with ALL parenthetical citation blocks removed. A 'parenthetical citation block' means any substring\n"
        "enclosed in parentheses ( ... ) that contains any of the following keywords: " + kw_line + ".\n\n"
        "CRITICAL RULES (must be followed exactly):\n"
        "1) Return ONLY the transformed text. No explanations, no JSON, no extra tokens.\n"
        "2) Do NOT alter any characters other than removing the ENTIRE parenthetical blocks described above.\n"
        "   That means: preserve every letter, punctuation, spacing, newline, tabs — EXACTLY as in input — except remove\n"
        "   the parentheses and everything inside them for the matched citation blocks.\n"
        "3) Do NOT remove parentheses that do NOT contain any of the listed keywords.\n"
        "4) If multiple citation parentheses appear, remove them all. Maintain original spacing/newlines (do not collapse lines).\n"
        "5) If removing a citation creates double spaces, preserve them (we will tidy whitespace outside if desired).\n\n"
        "INPUT TEXT FOLLOWS (do not add or remove any additional text above/below):\n\n"
        "----BEGIN INPUT----\n"
        f"{text}\n"
        "----END INPUT----\n\n"
        "Now output ONLY the cleaned text without ----BEGIN INPUT---- or ----END INPUT---- markers."
    )

def strip_inline_citations_with_llm(text: str, llm=LLM, keywords: Optional[list] = None, max_tokens: int = 32768) -> str:
    """
    Use LLM to remove parenthetical citation blocks containing known citation keywords.
    Returns the cleaned text. If the LLM output fails validation, fall back to a conservative regex-based removal.
    """
    if not text:
        return text

    prompt = _build_strip_prompt(text, keywords)
    # Invoke the LLM deterministically
    resp = llm.invoke([SystemMessage(content="You are a deterministic text transformer."), HumanMessage(content=prompt)])
    cleaned = getattr(resp, "content", "") or ""

    # Quick validation: ensure cleaned is same as input with some parenthetical substrings removed,
    # i.e. cleaned should be *shorter or equal* and should share large sequence similarity.
    try:
        # simple ratio check
        seq = difflib.SequenceMatcher(a=text, b=cleaned)
        ratio = seq.quick_ratio()
    except Exception:
        ratio = 0.0

    # If LLM returned something plausible (high similarity) accept it; else fallback.
    if ratio > 0.7 and len(cleaned) <= len(text):
        return cleaned
    else:
        # Fallback: conservative regex removal (only removes parentheses that contain keywords).
        # This fallback is safer than removing all parentheses.
        import re
        kw_pattern = "|".join([re.escape(k) for k in (keywords or _CITATION_KEYWORDS_EXAMPLE)])
        fallback_re = re.compile(r'\(\s*[^)]*(?:' + kw_pattern + r')[^)]*\)', flags=re.IGNORECASE)
        fallback = fallback_re.sub('', text)
        # Keep whitespace exactly as fallback produced (no extra cleanup).
        return fallback


_EXTRACTOR_PROMPT = (
    "You are an IFRS reference selector helping to compile a comprehensive answer.\n\n"
    "CONTEXT:\n"
    "- User asked: {question}\n"
    "- You have snippets from 5 sources: IFRS A (standard text), IFRS B (examples), IFRS C (basis for conclusions), EY (guidance), PwC (guidance)\n"
    "- Each snippet has format: <DOC_ID:abc123> [Source | para X | Chapter]\n\n"
    "YOUR TASK:\n"
    "Select DOC_IDs for ALL snippets that should be included in the answer. Be COMPREHENSIVE.\n\n"
    "SELECTION CRITERIA (include if ANY apply):\n"
    " Directly answers the question\n"
    " Defines key terms or concepts mentioned in the question\n"
    " Provides requirements, rules, or principles related to the topic\n"
    " Gives examples or illustrations of the concept\n"
    " Explains the rationale or background (basis for conclusions)\n"
    " Provides practical guidance or implementation notes (EY/PwC)\n"
    " Discusses exceptions, special cases, or edge scenarios\n"
    " References related standards or cross-references\n\n"
    "IMPORTANT GUIDELINES:\n"
    "• Include snippets from MULTIPLE sources (IFRS A/B/C AND EY/PwC) - they provide different perspectives\n"
    "• IFRS A/B/C = official standards; EY/PwC = practical interpretation and guidance\n"
    "• When uncertain whether a snippet is relevant → INCLUDE IT (be generous, not strict)\n"
    "• Aim for 12-25 selected snippets for most questions (more for complex topics)\n"
    "• If a snippet discusses the topic AT ALL, include it\n\n"
    "OUTPUT FORMAT (strict JSON):\n"
    "{{\n"
    '  "doc_ids": ["abc123", "def456", "xyz789", ...],\n'
    '  "reason": "Brief summary: Selected X from IFRS A (requirements), Y from IFRS B (examples), Z from EY/PwC (guidance)"\n'
    "}}\n\n"
    "Question: {question}\n\n"
    "Available snippets:\n{context}\n\n"
    "Now output your JSON selection:"
)

stuff_prompt_normal = PromptTemplate(
    input_variables=["text", "question"],
    template="""--------------------------------- MAKE THE ANSWER AS DETAILED AS POSSIBLE -------------------
You are a precise assistant. Use ONLY the provided context below (it may contain multiple source chunks or partial answers). Do not use external knowledge.

Rules:
- Put the **Conclusion:** section at the very top, concise but specific.
- Then include an **Introduction:** section.
- In the body, use bullet points and clear sub-headers.
- Avoid irrelevant details such as classification types or modifications unless explicitly requested.
- When examples are present, interpret them as "illustrations" instead of providing the full text.

Formatting:
- Use consistent bullet points with "–" (single dash, one space after).
- Use straight double quotes `"` instead of curly/inverted quotes.

Tables - CRITICAL FORMATTING RULES:
- Only produce a table if the user explicitly asks for "table", "tabular format", "schedule", "amortization schedule",
  OR if the user asks to "compare", "contrast", "vs", or "differences" between two-or-more items.
- When a comparison is implied, prefer a 2-column or multi-column markdown pipe table with the strict rules below.

#
2. Column Headers:
   - Each header must be SHORT and DISTINCT (e.g., "Year", "Opening Balance", "Interest Expense", "Closing Balance")
   - Never put multiple column names in one header cell
   - Never use semicolons in headers
   - Maximum 10-15 words per header

3. Data Cells - ABSOLUTELY CRITICAL:
   - Each cell contains EXACTLY ONE value
   - NEVER use semicolons (;) to separate multiple values in a cell
   - NEVER use commas to list multiple items in a cell
   - NEVER use line breaks within a cell
   - If you need to show multiple related values, CREATE SEPARATE COLUMNS for each
   - Each cell should be a single number, single text phrase, or single date

4. Pipe Consistency:
   - Count the pipes in your header row
   - EVERY row must have the exact same number of pipes
   - Format: | value1 | value2 | value3 | value4 |

5. Example of CORRECT table format:
```
| Year | Opening Carrying Amount | Contractual Coupon | Principal Repayment | Interest Expense | Amortisation | Closing Carrying Amount |
| --- | --- | --- | --- | --- | --- | --- |
| 2024 | 1,000,000 | 60,000 | 0 | 71,900 | 11,900 | 1,011,900 |
| 2025 | 1,011,900 | 60,000 | 0 | 72,756 | 12,756 | 1,024,656 |
```

6. Example of INCORRECT table (DO NOT DO THIS):
```
| Year | Opening carrying amount; Contractual coupon; Principal repayment; Interest expense; Amortisation; Closing carrying amount |
| --- | --- |
| 2024 | 1,000,000; 60,000; 0; 71,900; 11,900; 1,011,900 |
```

7. Additional Rules:
   - Do NOT merge columns
   - Do NOT use HTML tables
   - Do NOT use code fences around tables
   - Do NOT collapse the table into a single line
   - Do NOT use alignment markers like |:---|---:| (use plain dashes only)
   - DO NOT include any inline citations.
   - *ABSOLUTELY FORBID* any inline citations.
   
########ADDITIONAL RULES — APPLY ONLY WHEN GENERATING JOURNAL ENTRIES:########

- The table MUST have EXACTLY 6 columns in this order:
  Line No | Account Name | Debit | Credit | Currency | Description

- Every row MUST contain exactly 5 pipe characters (|).

- Debit and Credit columns MUST NEVER be empty.
  Use 0 explicitly where applicable.

- Exactly ONE of Debit or Credit MUST be non-zero in each row.

- Do NOT merge, omit, or shift Debit or Credit values across columns.

- Do NOT introduce extra pipes, spacing-based alignment, or wrapped text.

- Use plain numeric values only (no commas, symbols, or formatting).

- If the transaction requires a balancing entry, generate separate rows;
  never combine debit and credit amounts in a single row.

When tables ARE required, follow these STRICT rules:
1. Do NOT include any markdown tables in the free-text answer.
2. Put ALL tables ONLY in the JSON output (see structure below).
3. The JSON tables must follow this structure:
   - ONE header row with distinct column names (as "columns")
   - ONE data row per line (as "rows")

 
**Absolutely forbid** any marketing or next-step language. Do NOT include offers, CTAs, or meta lines such as:
"If you would like", "tell me which format", "I can also", "Body", "let me know", "you can ask for", "I will produce it", etc.


Question:
{question}

Context:
{text}



**CRITICAL - DO NOT CREATE A REFERENCES SECTION:**
- DO NOT create a "References:" or "Reference:" section at the end of your answer
- DO NOT list citations at the end in a separate section
- Your answer should end with the last paragraph of content, not with a reference list

FINAL OUTPUT WRAPPING — MANDATORY AND OVERRIDING

In addition to the formatted answer above, you MUST return a machine-readable JSON object
containing ALL tables you generated.

This JSON object MUST appear AFTER the main answer and MUST be the LAST thing in the response.

The JSON MUST follow this exact structure:

{{
    "answer_text": "<free text answer>",
    "tables": [
        {{
            "table_name": "<short descriptive name>",
            "columns": ["Column1", "Column2", "..."],
            "rows": [
                [value1, value2, ...],
                [value1, value2, ...]
            ]
        }}
    ]
}}

CRITICAL RULES:
- Include ONLY tables that were generated in the answer above.
- The JSON must EXACTLY match the table content already shown.
- The "answer_text" must NOT contain any markdown tables.
- Each row must have the same number of values as columns.
- Each cell must contain exactly one value.
- Use numbers for numeric values and strings for text values.
- Do NOT include explanations, markdown, or extra text inside the JSON.
- If NO tables were generated, return: {{ "tables": [] }}


The JSON must be valid and parseable.
Do NOT add any text after the JSON.
"""
)

stuff_prompt_ooc = PromptTemplate(
    input_variables=["text", "question"],
    template="""--------------------------------- MAKE THE ANSWER AS DETAILED AS POSSIBLE -------------------
You are a precise assistant. Use ONLY the provided context below (it may contain multiple source chunks or partial answers). Do not use external knowledge.

️ Note: The user's question appears likely to be **out of context** for this knowledge base. Be extremely strict and if you feel the question is not related to the context decline to answer.

IF YOU HAVE CONCLUDED THAT THE QUESTION IS NOT RELATED TO THE SOURCE, DO NOT GIVE ANY ANSWER, ONLY RESPOND WITH "I DON'T KNOW".

Question:
{question}

Context:
{text}

"""
)

# ═══════════════════════════════════════════════════════════════════════════
# Exception Section Prompts
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# IMPROVED Exception Section Prompts - Expert Level
# ═══════════════════════════════════════════════════════════════════════════

exception_identification_prompt = PromptTemplate(
    input_variables=["question", "main_answer"],
    template="""You are an IFRS expert assistant. Your task is to identify exception-related topics that should be researched.

**Original Question:**
{question}

**Main Answer Generated:**
{main_answer}

**Your Task:**
Analyze the question and answer to identify if there are any EXCEPTIONS, WAIVERS, EXEMPTIONS, SPECIAL CONDITIONS, PRACTICAL EXPEDIENTS, or RELIEFS that **DIRECTLY IMPACT** the answer to this specific question.

**What qualifies as a RELEVANT exception topic:**
1. Alternative treatments that would CHANGE the answer for the specific scenario in the question
2. Waivers/exemptions that the user MUST be aware of for their exact situation
3. Conditions that could ALTER the accounting treatment described in the main answer
4. Practical expedients that provide MEANINGFUL alternatives for this specific case
5. Scope exclusions that might EXEMPT the transaction/entity from the rules discussed

**What does NOT qualify (DO NOT search for these):**
- General transition provisions unrelated to the question
- Exceptions for completely different transaction types not mentioned in the question
- Technical details about business combinations unless specifically asked
- Peripheral policy choices that don't materially affect the core answer
- Exceptions that apply to edge cases not relevant to the question

**Critical Test:**
For each potential search topic, ask: "Would knowing this exception CHANGE or QUALIFY the main answer for THIS specific question?"
If NO → Don't search for it.

**Output Format:**
Return a JSON object with the following structure:
{{
  "has_exceptions": true/false,
  "search_queries": [
    "exception topic 1 to search for",
    "exception topic 2 to search for",
    ...
  ],
  "reason": "Brief explanation of why these exception topics are DIRECTLY relevant to THIS question"
}}

**Rules:**
- If NO exception topics are DIRECTLY relevant to this question, return {{"has_exceptions": false, "search_queries": [], "reason": "No exceptions directly applicable"}}
- Keep search queries focused and specific (e.g., "short-term lease exception", "investment property exemption from depreciation")
- Maximum 3-4 search queries (focus on the most critical ones)
- Each query should target exceptions that could materially affect the answer to THIS question

Return ONLY the JSON object, no additional text.
"""
)

exception_generation_prompt = PromptTemplate(
    input_variables=["question", "main_answer", "exception_context"],
    template="""You are an IFRS expert assistant. Your task is to generate an "Exceptions, Waivers & Special Conditions" section.

**Original Question:**
{question}

**Main Answer:**
{main_answer}

**Exception-Related Context (retrieved documents):**
{exception_context}

**═══════════════════════════════════════════════════════════════════════════**
**🎯 CRITICAL RELEVANCE TEST - READ THIS FIRST:**

Before including ANY exception, ask yourself:
1. **Does this exception directly relate to the SPECIFIC scenario in the user's question?**
2. **Would this exception CHANGE or MATERIALLY QUALIFY the main answer?**
3. **Is this something the user MUST know to avoid misapplying the guidance?**

If you answer NO to any of these → DO NOT include it.

**═══════════════════════════════════════════════════════════════════════════**
**🔍 HOW TO IDENTIFY EXCEPTIONS IN THE SOURCE MATERIAL:**

When reading the context chunks, a statement likely describes an EXCEPTION (not the general rule) if it exhibits these **STRUCTURAL PATTERNS:**

**1. INTRODUCES ALTERNATIVE TREATMENT:**
   - The text describes a different accounting treatment than what was previously stated as the standard approach
   - It presents a path that diverges from the primary requirement
   - The treatment described is presented as optional or available under specific circumstances

**2. IMPOSES ADDITIONAL CONDITIONS:**
   - Beyond the general rule, the text requires satisfaction of extra criteria, tests, or thresholds
   - References recognition criteria from a DIFFERENT standard
   - Mentions multiple conditions that must ALL be satisfied before the treatment applies

**3. NARROWS THE SCOPE OF APPLICATION:**
   - The text carves out a subset of transactions, entities, or situations from the broader category
   - It describes situations where you must first disaggregate, separate, or identify distinct components
   - It treats parts of a transaction differently from the whole

**4. REQUIRES CASE-BY-CASE JUDGMENT:**
   - The text explicitly or implicitly requires evaluating individual facts and circumstances
   - It indicates that the same type of transaction could receive different treatment depending on characteristics
   - The guidance shifts from prescriptive ("shall") to judgment-based assessment

**5. REFERENCES SUBSTANCE OVER FORM:**
   - The text emphasizes what rights or services are actually received, not what the arrangement is called
   - It requires looking beyond the transaction label to its economic substance

**CRITICAL TEST:** Does this text describe a CONDITIONAL treatment that only applies when something additional is present/satisfied? If yes → It's likely an EXCEPTION.

**═══════════════════════════════════════════════════════════════════════════**

**Your Task:**
1. Review the exception-related context chunks
2. Identify genuine exceptions, waivers, exemptions, special conditions, practical expedients, or reliefs
3. **ONLY include exceptions that pass the CRITICAL RELEVANCE TEST above**
4. Generate a concise section listing these items with proper citations

**STRICT FILTERING RULES:**
❌ **EXCLUDE** these types of exceptions even if they appear in the context:
- Transition provisions or first-time adoption rules (unless the question is specifically about transition)
- Business combination accounting (unless the question explicitly asks about acquisitions)
- Exceptions for completely different transaction types not mentioned in the question
- General policy choices that don't affect the specific scenario
- Edge cases that are clearly outside the scope of the question
- Disclosure-only exemptions when the question is about recognition/measurement

✅ **INCLUDE** only these types of exceptions:
- Exceptions that create a DIFFERENT accounting outcome for the user's scenario
- Exemptions that might RELIEVE the user from requirements discussed in the main answer
- Conditions that QUALIFY when the main answer applies vs. when it doesn't
- Practical expedients that offer MEANINGFUL simplification for the specific case

**Output Format:**
If exceptions are found that pass the relevance test, return:
```
**Exceptions, Waivers & Special Conditions:**
– **Exception:** [Description] <citation>source</citation>
– **Practical Expedient:** [Description] <citation>source</citation>
– **Exemption:** [Description] <citation>source</citation>
...
```

If NO relevant exceptions are found in the context, return EXACTLY:
```
NO_EXCEPTIONS_FOUND
```

**Rules:**
- Be STRICT: only include items that are genuinely relevant to THIS question
- Each bullet must have a citation in format: <citation>source</citation>
- Use categories: Exception, Waiver, Exemption, Special Condition, Practical Expedient, Relief
- Keep descriptions concise (1-2 sentences max per item)
- Maximum 5 items (if you have more, prioritize the most impactful ones)
- Write in clear, practical language that explains the IMPACT on the user's scenario

Return ONLY the formatted section or "NO_EXCEPTIONS_FOUND", no additional text.
"""
)

exception_filtering_prompt = PromptTemplate(
    input_variables=["question", "main_answer", "exception_section"],
    template="""You are an IFRS expert assistant performing STRICT RELEVANCE FILTERING on an exception section.

**Original Question:**
{question}

**Main Answer:**
{main_answer}

**Current Exception Section (unfiltered):**
{exception_section}

**═══════════════════════════════════════════════════════════════════════════**
**🎯 YOUR MISSION: AGGRESSIVE FILTERING**

Your job is to be RUTHLESSLY STRICT. Remove ANY exception that doesn't pass ALL of these tests:

**✅ KEEP AN EXCEPTION ONLY IF ALL OF THESE ARE TRUE:**

1. **Direct Question Relevance Test:**
   - The exception directly addresses something mentioned or implied in the user's question
   - It relates to the SAME transaction type, entity type, or scenario the user is asking about
   
2. **Main Answer Impact Test:**
   - The exception would CHANGE, QUALIFY, or ADD CRITICAL NUANCE to the main answer
   - Without this exception, the user might misapply the guidance from the main answer
   - It describes an alternative path for the SPECIFIC scenario in the question

3. **Practical Significance Test:**
   - The exception has a MATERIAL impact on how the user should handle their situation
   - It's not just technical detail - it actually affects decision-making
   - It's something a practitioner MUST be aware of for this specific case

4. **Specificity Test:**
   - The exception is SPECIFICALLY relevant, not just broadly related
   - It's not about a different phase (e.g., transition when question is about ongoing accounting)
   - It's not about a different transaction type (e.g., business combinations when question is about regular transactions)

**❌ REMOVE AN EXCEPTION IF ANY OF THESE ARE TRUE:**

1. **Wrong Context:**
   - About business combinations when question doesn't mention acquisitions
   - About transition/first-time adoption when question is about ongoing accounting
   - About different transaction types not mentioned in the question
   - About disclosure when question is about recognition/measurement

2. **Too Peripheral:**
   - General policy choices that don't materially affect the answer
   - Edge cases clearly outside the question's scope
   - Related to different stages of accounting not asked about
   - Technical details that don't change the practical outcome

3. **Redundant with Main Answer:**
   - Information already covered adequately in the main answer
   - Doesn't add meaningful new information
   - Just restates what's already clear

4. **Low Impact:**
   - Doesn't materially affect how the user should apply the guidance
   - Minor technical point that doesn't change decision-making
   - Applies to rare situations unlikely to be relevant

**═══════════════════════════════════════════════════════════════════════════**

**SPECIFIC EXAMPLES OF WHAT TO REMOVE:**

**Example 1: Question about lease classification**
❌ REMOVE: "Relief: At business combinations, an acquirer is not required to recognize right-of-use assets..."
→ Why? Question is about ongoing lease accounting, not business combinations

**Example 2: Question about revenue recognition**
❌ REMOVE: "Exception: First-time adopters may use practical expedients..."
→ Why? Question is about ongoing accounting, not transition

**Example 3: Question about specific asset type**
❌ REMOVE: "Policy Choice: Entities may choose to..."
→ Why? If it's a general policy choice not specific to the question's scenario

**Example 4: Question about recognition**
❌ REMOVE: "Exception: Disclosure requirements may be modified if..."
→ Why? Question is about recognition, not disclosure

**═══════════════════════════════════════════════════════════════════════════**

**Your Task:**
Review each exception point in the section above. For each point:

1. Apply ALL four ✅ KEEP tests - if ANY test fails, remove it
2. Check against ALL ❌ REMOVE criteria - if ANY applies, remove it
3. When in doubt, REMOVE IT (err on the side of being too strict)

**Target: Keep only 2-3 of the MOST CRITICAL exceptions.** If you find yourself keeping more than 3, you're not being strict enough.

**Output Format:**
Return the filtered section using the EXACT same format WITHOUT inline citations:
```
**Exceptions, Waivers & Special Conditions:**
– **Exception:** [Description]
– **Practical Expedient:** [Description]
...
```

**Important:**
- Remove the <citation> tags from the output (keep only the description text)
- If ALL points should be removed (none pass the strict tests), return EXACTLY: NO_EXCEPTIONS_FOUND
- Maximum 3 points in the output (be extremely selective)
- Each kept point must be genuinely CRITICAL to understanding the answer to THIS question

Return ONLY the filtered section or "NO_EXCEPTIONS_FOUND", no additional text or explanation.
"""
)

LLM_RELEVANCE = get_llm("relevance")

_SYSTEM_PROMPT_RELEVANCE = (
    "You are a relevance judge. Make a binary decision about whether the "
    "user’s question can be meaningfully answered using ONLY the provided snippets.\n\n"
    "Decision rules:\n"
    "- If at least one snippet clearly addresses the user’s information need, OUTPUT label=relevant.\n"
    "- If the user’s question is about IFRS, US GAAP, International Accounting Standards (IAS), "
    "or any accounting treatment, OUTPUT label=relevant (even if snippets are only loosely related).\n"
    "- Otherwise OUTPUT label=irrelevant.\n\n"
    "Output JSON only, no extra text."
)


_HUMAN_PROMPT_TEMPLATE_RELEVANCE = """Question:
{question}

Snippets (from one DB):
{snippets_block}

Instructions:
1) Read ONLY these snippets.
2) Decide: 
   - 'relevant' if at least one snippet substantially helps answer the question, 
     OR if the question is related to IFRS, US GAAP, IAS, or accounting treatment (loose match allowed).
   - 'irrelevant' otherwise.
3) Return STRICT JSON of the form:
{{
  "label": "relevant" | "irrelevant",
  "reason": "one short sentence explaining why"
}}
"""

def _parse_extractor_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {"doc_ids": [], "reason": "Empty"}
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("doc_ids", []), list):
            return data
    except Exception:
        pass
    m = re.search(r"\{(?:.|\n)*\}", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and isinstance(data.get("doc_ids", []), list):
                return data
        except Exception:
            pass
    return {"doc_ids": [], "reason": "Could not parse"}

# def _extract_allowed_paras_with_llm(question: str, docs: List[Any]) -> List[str]:
#     if not docs:
#         return []
#     seen_paras = _collect_seen_paras_from_docs(docs)
#     if not seen_paras:
#         return []
#     ctx = _format_docs_round_robin(docs, total_max=20, per_source_cap=5)
#     msg = _EXTRACTOR_PROMPT.format(
#         question=_normalize_dashes((question or "").strip()),
#         seen_paras=", ".join(seen_paras),
#         context=ctx
#     )
#     resp = LLM_EXTRACTOR.invoke([SystemMessage(content="Output STRICT JSON only."), HumanMessage(content=msg)])
#     parsed = _parse_extractor_json(getattr(resp, "content", "") or "")
#     picked = sorted(set(parsed.get("paragraph_ids", [])) & set(seen_paras))
#     return picked

# def _extract_allowed_paras_with_llm(question: str, docs: List[Any]) -> List[str]:
#     """Extract relevant paragraph references using metadata-based para numbers."""
#     if not docs:
#         return []
    
#     # Use metadata instead of regex scraping
#     seen_paras = _collect_seen_paras_from_metadata(docs)
    
#     if not seen_paras:
#         return []
    
#     ctx = _format_docs_round_robin(docs, total_max=20, per_source_cap=5)
#     msg = _EXTRACTOR_PROMPT.format(
#         question=_normalize_dashes((question or "").strip()),
#         seen_paras=", ".join(seen_paras),
#         context=ctx
#     )
#     resp = LLM_EXTRACTOR.invoke([
#         SystemMessage(content="Output STRICT JSON only."), 
#         HumanMessage(content=msg)
#     ])
#     parsed = _parse_extractor_json(getattr(resp, "content", "") or "")
#     picked = sorted(set(parsed.get("paragraph_ids", [])) & set(seen_paras))
#     return picked

def _extract_allowed_doc_ids_with_llm(question: str, docs: List[Any]) -> List[str]:
    """Extract relevant document IDs using LLM - returns list of doc_ids."""
    if not docs:
        return []
    
    #  CHANGED: Increase from 20 to 40, use round-robin sampling
    ctx = _format_docs_with_ids(docs, max_docs=55)
    
    msg = _EXTRACTOR_PROMPT.format(
        question=_normalize_dashes((question or "").strip()),
        context=ctx
    )
    
    resp = LLM_EXTRACTOR.invoke([
        SystemMessage(content="Output STRICT JSON only."), 
        HumanMessage(content=msg)
    ])
    
    parsed = _parse_extractor_json(getattr(resp, "content", "") or "")
    doc_ids = parsed.get("doc_ids", [])
    
    # Validate that these doc_ids actually exist in the docs
    valid_ids = {(d.metadata or {}).get("_doc_id") for d in docs if (d.metadata or {}).get("_doc_id")}
    filtered_ids = [doc_id for doc_id in doc_ids if doc_id in valid_ids]

    # Count by source DB
    selected_by_db = {}
    for doc in docs:
        if (doc.metadata or {}).get("_doc_id") in filtered_ids:
            db = (doc.metadata or {}).get("source_db", "Unknown")
            selected_by_db[db] = selected_by_db.get(db, 0) + 1

    # Logging removed - done in main pipeline
    # Store counts for later logging
    import threading
    if not hasattr(threading.current_thread(), '_llm_selection_stats'):
        threading.current_thread()._llm_selection_stats = {}
    threading.current_thread()._llm_selection_stats = {
        'total_selected': len(filtered_ids),
        'total_docs': len(docs),
        'by_db': selected_by_db,
        'reason': parsed.get('reason', 'N/A')
    }

    return filtered_ids

def _parse_llm_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {"label": "irrelevant", "reason": "Empty"}
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{(?:.|\n)*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"label": "irrelevant", "reason": "Parse error"}

def _normalize_label(label: Optional[str]) -> str:
    lab = (label or "").strip().lower()
    return "relevant" if lab == "relevant" else "irrelevant"

def get_query_relevance_llm(dir_path: str, question: str, score_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Check if a database contains relevant documents for the question.
    Uses a lower threshold (0.5) for relevance check to be more inclusive.

    Args:
        dir_path: Path to FAISS index
        question: User query
        score_threshold: Minimum similarity score (default 0.5 for relevance check)

    Returns:
        Dict with 'label' (relevant/irrelevant), 'reason', and 'db' path
    """
    # Use score-based retrieval with lower threshold for relevance check
    docs_with_scores = retrieve_docs_with_score(dir_path, question, score_threshold=score_threshold, max_k=20)
    docs = [doc for doc, score in docs_with_scores]

    if not docs:
        # No documents passed the threshold, mark as irrelevant
        return {"label": "irrelevant", "reason": f"No documents with similarity >= {score_threshold}", "db": dir_path}

    snippets_block = _format_docs_full(docs, max_docs=10)
    human = _HUMAN_PROMPT_TEMPLATE_RELEVANCE.format(
        question=(question or "").strip(),
        snippets_block=snippets_block
    )
    resp = LLM_RELEVANCE.invoke([SystemMessage(content=_SYSTEM_PROMPT_RELEVANCE), HumanMessage(content=human)])
    parsed = _parse_llm_json(getattr(resp, "content", "") or "")
    label = _normalize_label(parsed.get("label"))
    reason = (parsed.get("reason") or "").strip()
    return {"label": label, "reason": reason, "db": dir_path}

# def _filter_docs_by_paras(docs: List[Any], allowed_paras: List[str]) -> List[Any]:
#     if not allowed_paras:
#         return docs
#     allowed = set(allowed_paras)
#     kept = []
#     for d in docs or []:
#         txt = _normalize_dashes(getattr(d, "page_content", "") or "")
#         ids_in_doc = set(PARA_ID_RE.findall(txt))
#         if ids_in_doc & allowed:
#             kept.append(d)
#     return kept if kept else docs

# def _filter_docs_by_paras(docs: List[Any], allowed_paras: List[str]) -> List[Any]:
#     if not allowed_paras:
#         return docs
#     allowed = set(allowed_paras)
#     kept = []
#     for d in docs or []:
#         txt = _normalize_dashes(getattr(d, "page_content", "") or "")
#         ids_in_doc = set(PARA_ID_RE.findall(txt))

#         # check chapter-level fallback match
#         meta = d.metadata or {}
#         chapter = meta.get("chapter_name") or meta.get("chapter") or ""
#         source_db = meta.get("source_db") or ""
#         chapter_key = f"{source_db} — {chapter}" if chapter else None

#         if ids_in_doc & allowed or (chapter_key and chapter_key in allowed):
#             kept.append(d)

#     return kept if kept else docs

def _filter_docs_by_paras(docs: List[Any], allowed_paras: List[str]) -> List[Any]:
    if not allowed_paras:
        return docs
    
    allowed = set(allowed_paras)
    kept = []
    
    # Separate paragraph IDs from chapter-level keys
    para_ids = {p for p in allowed_paras if not ("–" in p or "—" in p)}
    chapter_keys = {p for p in allowed_paras if ("–" in p or "—" in p)}
    
    for d in docs or []:
        meta = d.metadata or {}
        
        # Priority 1: Match specific paragraph numbers
        para_num = meta.get("para_number")
        if para_num and str(para_num).strip() in para_ids:
            kept.append(d)
            continue
        
        # Priority 2: Only use chapter fallback if NO paragraph IDs were extracted
        if not para_ids:  # ← Only fall back if we have no specific paragraphs
            chapter = meta.get("chapter_name") or meta.get("chapter") or ""
            source_db = meta.get("source_db") or ""
            chapter_key = f"{source_db} – {chapter}" if chapter else None
            
            if chapter_key and chapter_key in chapter_keys:
                kept.append(d)
    
    return kept if kept else docs

import uuid

def fetch_docs(question: str, cfg: DBConfig) -> List[Document]:
    """
    Fetch documents using percentile-based selection (default: top 30%).
    Returns top X% of documents by cosine similarity score.
    """
    # Use percentile-based retrieval (default mode)
    docs_with_scores = retrieve_docs_with_score(
        cfg.path,
        question,
        score_threshold=None,  # Use percentile mode, not threshold
        top_percentile=cfg.top_percentile,  # Top 30% by default
        max_k=50  # Fetch up to 50 docs initially, then select top X%
    )

    cleaned: List[Document] = []
    for d, similarity_score in docs_with_scores:
        content = (d.page_content or "").strip()
        tag = _source_tag(d.metadata, cfg.name)
        visible_chunk = f"{tag} {content}"

        #  Add unique ID and similarity score to metadata
        unique_id = str(uuid.uuid4())[:8]  # Short ID like "a3b4c5d6"

        cleaned.append(
            Document(
                page_content=visible_chunk,
                metadata={
                    **(d.metadata or {}),
                    "source_db": cfg.name,
                    "_doc_id": unique_id,  # Hidden ID for tracking
                    "_similarity_score": similarity_score  # Store similarity score
                }
            )
        )

    # Logging removed - done centrally in pipeline
    return cleaned

def subset_references(bottom_refs: List[str], used_refs: List[str]) -> List[str]:
    """Subset bottom reference list by matching against 'References used' fallback entries."""
    keep = []
    for ref in bottom_refs:
        for used in used_refs:
            if "—" in used:
                src, chap = [x.strip() for x in used.split("—", 1)]
                # loose match: check source and chapter substring
                if src in ref and chap.split()[0] in ref:
                    keep.append(ref)
                    break
    return keep


# def answer_with_refine_chain(question: str):
#     results = []
#     for cfg in DBS:
#         s = get_query_relevance_llm(cfg.path, question, k=5)
#         results.append(s)
#         print(s)
#     if any(r["label"] == "relevant" for r in results):
#         all_docs_in_order: List[Document] = []
#         offsets: List[int] = []
#         for cfg in DBS:
#             docs = fetch_docs(question, cfg)
#             all_docs_in_order.extend(docs)
#             offsets.append(len(all_docs_in_order))
#         allowed_paras: List[str] = _extract_allowed_paras_with_llm(question, all_docs_in_order)
#         #  Fallback: if no explicit paragraph IDs detected, use chapter/source metadata
#         if not allowed_paras:
#             allowed_paras = []
#             for d in all_docs_in_order:
#                 meta = d.metadata or {}
#                 chapter = meta.get("chapter_name") or meta.get("chapter") or ""
#                 source_db = meta.get("source_db") or ""
#                 if chapter:
#                     allowed_paras.append(f"{source_db} — {chapter}")
#             allowed_paras = sorted(set(allowed_paras))  # dedupe
#         filtered_docs = _filter_docs_by_paras(all_docs_in_order, allowed_paras)

#         refine_chain = load_summarize_chain(
#             LLM,
#             chain_type="stuff",
#             prompt=stuff_prompt_normal,
#             verbose=True,
#         )
#         out = refine_chain.invoke({
#             "input_documents": filtered_docs,
#             "question": question,
#         })
#         final_answer: str = out.get("output_text", "")
#         if 'allowed_paras' in locals():
#             final_answer = f"{final_answer}\n\n**References used:** " + (", ".join(allowed_paras) if allowed_paras else "None observed.")
#         inter: List[str] = out.get("intermediate_steps", []) or []
#         stage_answers: Dict[str, str] = {}
#         if len(offsets) >= 1 and len(inter) >= offsets[0]:
#             stage_answers["STAGE 1 (A)"] = inter[offsets[0]-1]
#         if len(offsets) >= 2 and len(inter) >= offsets[1]:
#             stage_answers["STAGE 2 (A+B)"] = inter[offsets[1]-1]
#         if len(offsets) >= 3 and len(inter) >= offsets[2]:
#             stage_answers["STAGE 3 (A+B+C)"] = inter[offsets[2]-1]
#         if len(offsets) >= 4 and len(inter) >= offsets[3]:
#             stage_answers["STAGE 4 (A+B+C+EY)"] = inter[offsets[3]-1]
#         return {
#             "answer": final_answer,
#             "stage_answers": stage_answers,
#             "sources": filtered_docs,
#             "offsets": offsets,
#             "thresholds": None,
#             "scores": None,
#             "ooc_mode": False,
#             "prompt_used": "NORMAL",
#         }
#     else:
#         return {
#             "answer": "I DON'T KNOW",
#             "stage_answers": None,
#             "sources": None,
#             "offsets": None,
#             "thresholds": None,
#             "scores": None,
#             "ooc_mode": True,
#             "prompt_used": "OOC",
#         }

# def answer_with_refine_chain(question: str):
#     # Check relevance across all DBs
#     results = []
#     for cfg in DBS:
#         s = get_query_relevance_llm(cfg.path, question, k=5)
#         results.append(s)
#         print(s)
    
#     if any(r["label"] == "relevant" for r in results):
#         # Fetch all docs with unique IDs
#         all_docs_in_order: List[Document] = []
#         offsets: List[int] = []
        
#         for cfg in DBS:
#             docs = fetch_docs(question, cfg)  # Now includes _doc_id in metadata
#             all_docs_in_order.extend(docs)
#             offsets.append(len(all_docs_in_order))

#         #  ADD THIS PRINT
#         print(f"\n{'#'*80}")
#         print(f" TOTAL DOCS FETCHED: {len(all_docs_in_order)}")
#         db_counts = {}
#         for doc in all_docs_in_order:
#             db = (doc.metadata or {}).get('source_db', 'Unknown')
#             db_counts[db] = db_counts.get(db, 0) + 1
#         for db, count in db_counts.items():
#             print(f"  - {db}: {count} chunks")
#         print(f"{'#'*80}\n")
        
#         #  Extract relevant doc IDs using LLM (replaces paragraph extraction)
#         used_doc_ids: List[str] = _extract_allowed_doc_ids_with_llm(question, all_docs_in_order)
        
#         #  Filter docs strictly by doc IDs (no fallbacks!)
#         filtered_docs = _filter_docs_by_ids(all_docs_in_order, used_doc_ids)
        
#         # If no docs matched, return a clear message
#         if not filtered_docs:
#             return {
#                 "answer": "I couldn't identify specific relevant references for this question.",
#                 "stage_answers": {},
#                 "sources": [],
#                 "offsets": offsets,
#                 "thresholds": None,
#                 "scores": None,
#                 "ooc_mode": False,
#                 "prompt_used": "NORMAL",
#             }
        
#         # Generate answer using filtered docs
#         refine_chain = load_summarize_chain(
#             LLM,
#             chain_type="stuff",
#             prompt=stuff_prompt_normal,
#             verbose=True,
#         )
        
#         out = refine_chain.invoke({
#             "input_documents": filtered_docs,
#             "question": question,
#         })
        
#         final_answer: str = out.get("output_text", "")
        
#         #  Build "References used" section from actual metadata
#         refs_used = []
#         for doc in filtered_docs:
#             meta = doc.metadata or {}
#             para_num = meta.get("para_number", "")
#             chapter = meta.get("chapter_name") or meta.get("chapter", "")
#             source_db = meta.get("source_db", "")
#             header = meta.get("header", "")
            
#             # For IFRS A, B, C: use paragraph number
#             if source_db in ["IFRS A", "IFRS B", "IFRS C"]:
#                 if para_num and str(para_num).lower() not in {"none", "unnumbered", ""}:
#                     refs_used.append(f"{source_db} – {chapter} – para {para_num}")
#                 elif chapter:
#                     refs_used.append(f"{source_db} – {chapter}")
#             # For EY and PwC: use header
#             elif source_db in ["EY", "PwC"]:
#                 if header:
#                     refs_used.append(f"{source_db} – {chapter} – {header}")
#                 elif chapter:
#                     refs_used.append(f"{source_db} – {chapter}")

#         refs_used = sorted(set(refs_used))  # Deduplicate
        
#         if refs_used:
#             # Format as bulleted list
#             refs_list = "\n".join([f"- {ref}" for ref in refs_used])
#             final_answer = f"{final_answer}\n\n**References used:**\n{refs_list}"
        
#         # Handle stage answers
#         inter: List[str] = out.get("intermediate_steps", []) or []
#         stage_answers: Dict[str, str] = {}
#         if len(offsets) >= 1 and len(inter) >= offsets[0]:
#             stage_answers["STAGE 1 (A)"] = inter[offsets[0]-1]
#         if len(offsets) >= 2 and len(inter) >= offsets[1]:
#             stage_answers["STAGE 2 (A+B)"] = inter[offsets[1]-1]
#         if len(offsets) >= 3 and len(inter) >= offsets[2]:
#             stage_answers["STAGE 3 (A+B+C)"] = inter[offsets[2]-1]
#         if len(offsets) >= 4 and len(inter) >= offsets[3]:
#             stage_answers["STAGE 4 (A+B+C+EY)"] = inter[offsets[3]-1]
        
#         return {
#             "answer": final_answer,
#             "stage_answers": stage_answers,
#             "sources": filtered_docs,  #  Only the docs that were actually used
#             "offsets": offsets,
#             "thresholds": None,
#             "scores": None,
#             "ooc_mode": False,
#             "prompt_used": "NORMAL",
#         }
#     else:
#         return {
#             "answer": "I DON'T KNOW",
#             "stage_answers": None,
#             "sources": None,
#             "offsets": None,
#             "thresholds": None,
#             "scores": None,
#             "ooc_mode": True,
#             "prompt_used": "OOC",
#         }

# ═══════════════════════════════════════════════════════════════════════════
# New Exception Handling Functions
# ═══════════════════════════════════════════════════════════════════════════

def extract_citations_from_text(text: str) -> List[str]:
    """
    Extract all round-bracket citations from text.

    Example citations:
    - (IFRS A - ifrs-16-leases - para 62)
    - (PwC - 15 - Leases (IFRS 16) - para 15.105)
    - (IFRS B - ifrs-16-leases - Example 20)

    Returns:
        List of unique citation strings found in the text
    """
    # Pattern to match citations in format: (Source - ... - para/Example X)
    # This matches any content within parentheses that looks like our citation format
    pattern = r'\([A-Z][^\)]*?(?:para|Example|IG)[^\)]*?\)'

    citations = re.findall(pattern, text)

    # Clean up and deduplicate
    unique_citations = list(set(citations))

    print(f"\n{'='*80}")
    print(f"CITATION EXTRACTION")
    print(f"{'='*80}")
    print(f"Found {len(unique_citations)} unique citations in text")
    print(f"{'='*80}\n")

    return unique_citations


def map_citations_to_doc_ids(citations: List[str], all_docs: List[Document]) -> List[str]:
    """
    Map citations from text back to document IDs.

    Args:
        citations: List of citation strings extracted from text
        all_docs: All documents that were available (with _doc_id metadata)

    Returns:
        List of _doc_id values for documents that were actually cited
    """
    cited_doc_ids = []

    for citation in citations:
        # Parse the citation to extract key components
        # Example: (IFRS A - ifrs-16-leases - para 62)
        # We need to match this against document metadata

        for doc in all_docs:
            meta = doc.metadata or {}
            source_db = meta.get("source_db", "")
            chapter = meta.get("chapter") or meta.get("chapter_name", "")
            para_num = str(meta.get("para_number", ""))

            # Build expected citation format from metadata
            if source_db in ["IFRS A", "IFRS B", "IFRS C"]:
                expected_citation = f"({source_db} - {chapter} - para {para_num})"
            elif source_db in ["EY", "PwC"]:
                header = meta.get("header", "")
                if header:
                    expected_citation = f"({source_db} - {chapter} - {header} - para {para_num})"
                else:
                    expected_citation = f"({source_db} - {chapter} - para {para_num})"
            else:
                continue

            # Check if this citation matches the document
            # Use fuzzy matching since citations might have slight variations
            if citation.strip() in expected_citation or expected_citation in citation.strip():
                doc_id = meta.get("_doc_id")
                if doc_id and doc_id not in cited_doc_ids:
                    cited_doc_ids.append(doc_id)

    print(f"Mapped {len(citations)} citations to {len(cited_doc_ids)} document IDs")
    return cited_doc_ids


def retrieve_and_generate_exceptions(
    question: str,
    main_answer: str,
    llm: Any
) -> Dict[str, Any]:
    """
    Identify exception topics, retrieve relevant documents, and generate exception section.

    Args:
        question: Original user question
        main_answer: Main answer generated (with citations)
        llm: LLM instance to use

    Returns:
        Dictionary with:
        - exception_section: Generated exception section text (or empty string)
        - exception_docs: Documents used for exception generation
        - has_exceptions: Boolean indicating if exceptions were found
    """
    print(f"\n{'='*80}")
    print(f"EXCEPTION SECTION GENERATION - STAGE 1: TOPIC IDENTIFICATION")
    print(f"{'='*80}")

    # Step 1: Identify exception topics to search for
    identification_chain = LLMChain(
        llm=LLM_MINI,  # Use lighter model for topic identification
        prompt=exception_identification_prompt,
        verbose=True
    )

    id_result = identification_chain.invoke({
        "question": question,
        "main_answer": main_answer
    })

    id_output = id_result.get("text", "")

    # Parse JSON response
    try:
        id_data = json.loads(id_output.strip())
    except json.JSONDecodeError:
        # Try to extract JSON from response
        match = re.search(r'\{[^\}]+\}', id_output, re.DOTALL)
        if match:
            try:
                id_data = json.loads(match.group(0))
            except:
                id_data = {"has_exceptions": False, "search_queries": [], "reason": "Parse error"}
        else:
            id_data = {"has_exceptions": False, "search_queries": [], "reason": "Parse error"}

    has_exceptions = id_data.get("has_exceptions", False)
    search_queries = id_data.get("search_queries", [])
    reason = id_data.get("reason", "")

    print(f"Has exceptions: {has_exceptions}")
    print(f"Reason: {reason}")
    if search_queries:
        print(f"Search queries: {', '.join(search_queries)}")
    print(f"{'='*80}\n")

    if not has_exceptions or not search_queries:
        return {
            "exception_section": "",
            "exception_docs": [],
            "has_exceptions": False
        }

    # Step 2: Retrieve documents for each exception query
    print(f"{'='*80}")
    print(f"EXCEPTION SECTION GENERATION - STAGE 2: DOCUMENT RETRIEVAL")
    print(f"{'='*80}")

    all_exception_docs: List[Document] = []

    for query in search_queries[:5]:  # Limit to 5 queries max
        print(f"\nSearching for: {query}")

        # Fetch from all 5 databases (same as main answer)
        for cfg in DBS:
            docs = fetch_docs(query, cfg)
            all_exception_docs.extend(docs)
            if docs:
                print(f"  {cfg.name:8} Retrieved {len(docs):2d} docs")

    # Deduplicate by _doc_id
    seen_ids = set()
    unique_exception_docs = []
    for doc in all_exception_docs:
        doc_id = doc.metadata.get("_doc_id")
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_exception_docs.append(doc)

    print(f"\nTotal unique exception documents: {len(unique_exception_docs)}")
    print(f"{'='*80}\n")

    if not unique_exception_docs:
        return {
            "exception_section": "",
            "exception_docs": [],
            "has_exceptions": False
        }

    # Step 3: Apply LLM filtering to exception docs
    print(f"{'='*80}")
    print(f"EXCEPTION SECTION GENERATION - STAGE 3: LLM FILTERING")
    print(f"{'='*80}")

    exception_doc_ids = _extract_allowed_doc_ids_with_llm(question, unique_exception_docs)
    filtered_exception_docs = _filter_docs_by_ids(unique_exception_docs, exception_doc_ids)

    print(f"Filtered to {len(filtered_exception_docs)} relevant exception documents")
    print(f"{'='*80}\n")

    if not filtered_exception_docs:
        return {
            "exception_section": "",
            "exception_docs": [],
            "has_exceptions": False
        }

    # Step 4: Generate exception section
    print(f"{'='*80}")
    print(f"EXCEPTION SECTION GENERATION - STAGE 4: CONTENT GENERATION")
    print(f"{'='*80}")

    # Prepare exception context (with source tags at the beginning of each chunk)
    exception_context_parts = []
    for doc in filtered_exception_docs:
        meta = doc.metadata or {}
        db_label = meta.get("source_db", "Unknown")
        source_tag = _source_tag(meta, db_label)
        chunk_text = getattr(doc, "page_content", "")
        exception_context_parts.append(f"{source_tag} {chunk_text}")

    exception_context = "\n\n".join(exception_context_parts)

    generation_chain = LLMChain(
        llm=LLM_FULL,  # Use powerful model for exception section generation
        prompt=exception_generation_prompt,
        verbose=True
    )

    gen_result = generation_chain.invoke({
        "question": question,
        "main_answer": main_answer,
        "exception_context": exception_context
    })

    exception_section = gen_result.get("text", "").strip()

    # Check if exceptions were found
    if "NO_EXCEPTIONS_FOUND" in exception_section:
        print("No relevant exceptions found in retrieved documents")
        print(f"{'='*80}\n")
        return {
            "exception_section": "",
            "exception_docs": [],
            "has_exceptions": False
        }

    print(f"Exception section generated successfully (initial version)")
    print(f"{'='*80}\n")

    # Step 5: Filter exception section to keep only absolutely relevant points
    print(f"{'='*80}")
    print(f"EXCEPTION SECTION GENERATION - STAGE 5: RELEVANCE FILTERING")
    print(f"{'='*80}")
    print(f"Filtering exception points using gpt-5-mini to keep only absolutely relevant items...")

    filtering_chain = LLMChain(
        llm=LLM_MINI,  # Use lighter model for filtering
        prompt=exception_filtering_prompt,
        verbose=True
    )

    filter_result = filtering_chain.invoke({
        "question": question,
        "main_answer": main_answer,
        "exception_section": exception_section
    })

    filtered_exception_section = filter_result.get("text", "").strip()

    # Check if all exceptions were filtered out
    if "NO_EXCEPTIONS_FOUND" in filtered_exception_section:
        print("All exception points filtered out as not absolutely relevant")
        print(f"{'='*80}\n")
        return {
            "exception_section": "",
            "exception_docs": [],
            "has_exceptions": False
        }

    print(f"Exception section filtered successfully")
    print(f"{'='*80}\n")

    return {
        "exception_section": filtered_exception_section,
        "exception_docs": filtered_exception_docs,
        "has_exceptions": True
    }


def generate_unified_reference_list(
    all_docs: List[Document],
    cited_doc_ids: List[str]
) -> str:
    """
    Generate a unified reference list with excerpts for all cited documents.

    Args:
        all_docs: All documents that were available
        cited_doc_ids: List of _doc_id values that were actually cited in the answer

    Returns:
        Formatted reference list string with excerpts
    """
    # Filter to only cited documents
    cited_docs = [doc for doc in all_docs if doc.metadata.get("_doc_id") in cited_doc_ids]

    # Filter out page=0 references
    cited_docs = filter_page_zero_references(cited_docs)

    if not cited_docs:
        return ""

    # Build reference list with excerpts
    refs_parts = []

    for i, doc in enumerate(cited_docs, 1):
        meta = doc.metadata or {}

        # Get unified metadata
        unified_meta = _unify_metadata(meta)
        title = unified_meta.get("title") or "Reference"

        # Build caption
        caption_parts = [
            meta.get("source_db") or "",
            meta.get("chapter_name") or meta.get("chapter") or "",
            f"Page {meta.get('page')}" if meta.get("page") else "",
            f"Para {meta.get('para_number')}" if meta.get('para_number') else ""
        ]
        caption = " | ".join([p for p in caption_parts if p])

        # Get excerpt
        chunk_text = getattr(doc, "page_content", "") or ""
        excerpt = _make_excerpt(chunk_text)

        # Format reference entry
        ref_entry = f"{i}. **{title}**\n   {caption}\n   {excerpt}\n"
        refs_parts.append(ref_entry)

    references_text = "\n".join(refs_parts)

    return f"\n\n**References:**\n\n{references_text}"


def answer_with_refine_chain(question: str, llm: Optional[Any] = None):
    """
    Main RAG function with 5-DB sequential retrieval and LLM-based filtering.
    New architecture:
    1. Generate main answer with citations
    2. Generate exception section (separate retrieval + generation)
    3. Extract citations from both sections
    4. Generate unified reference list
    5. Remove inline citations before display

    âœ… Filters page=0 from display (References used list + sources returned).
    âœ… Answer generation still uses ALL chunks (no information loss).
    """
    # Use provided LLM or fall back to LLM_FULL for generation
    if llm is None:
        llm = LLM_FULL

    # Step 1: Check relevance across all DBs
    print(f"\n{'='*80}")
    print(f"STAGE 1: RELEVANCE CHECK (threshold={STAGE_1_THRESHOLD})")
    print(f"{'='*80}")

    results = []
    for cfg in DBS:
        s = get_query_relevance_llm(cfg.path, question, score_threshold=STAGE_1_THRESHOLD)
        results.append(s)
        status = "RELEVANT" if s["label"] == "relevant" else "NOT RELEVANT"
        print(f"{cfg.name:8} {status}")
    
    if any(r["label"] == "relevant" for r in results):
        # Step 2: Fetch all docs with unique IDs
        print(f"\n{'='*80}")
        print(f"STAGE 2: DOCUMENT RETRIEVAL (top {STAGE_2_PERCENTILE*100:.0f}% from each DB)")
        print(f"{'='*80}")

        all_docs_in_order: List[Document] = []
        offsets: List[int] = []

        for cfg in DBS:
            docs = fetch_docs(question, cfg)  # Now includes _doc_id in metadata
            all_docs_in_order.extend(docs)
            offsets.append(len(all_docs_in_order))

            # Show retrieval stats for this DB
            if docs:
                scores = [d.metadata.get("_similarity_score", 0) for d in docs]
                cutoff_score = min(scores) if scores else 0
                print(f"{cfg.name:8} Retrieved {len(docs):2d} docs (cutoff score: {cutoff_score:.3f})")
            else:
                print(f"{cfg.name:8} Retrieved  0 docs")

        # Show total
        print(f"\nTOTAL: {len(all_docs_in_order)} documents fetched")
        print(f"{'='*80}\n")
        
        # Step 3: Extract relevant doc IDs using LLM (replaces paragraph extraction)
        print(f"{'='*80}")
        print(f"STAGE 3: LLM FILTERING")
        print(f"{'='*80}")

        used_doc_ids: List[str] = _extract_allowed_doc_ids_with_llm(question, all_docs_in_order)

        # Retrieve stored stats
        import threading
        stats = getattr(threading.current_thread(), '_llm_selection_stats', {})
        if stats:
            print(f"Selected {stats['total_selected']}/{stats['total_docs']} documents:")
            for db in ["IFRS A", "IFRS B", "IFRS C", "EY", "PwC"]:
                count = stats['by_db'].get(db, 0)
                print(f"  {db:8} {count:2d} docs")
            print(f"\nSelection reason: {stats['reason']}")
        print(f"{'='*80}\n")
        
        # Step 4: Filter docs strictly by doc IDs (no fallbacks!)
        filtered_docs = _filter_docs_by_ids(all_docs_in_order, used_doc_ids)
        
        # If no docs matched, return a clear message
        if not filtered_docs:
            return {
                "answer": "I couldn't identify specific relevant references for this question.",
                "answer_text": "I couldn't identify specific relevant references for this question.",
                "tables": [],
                "exception_section": "",  # No exception section when no docs found
                "stage_answers": {},
                "sources": [],
                "offsets": offsets,
                "thresholds": None,
                "scores": None,
                "ooc_mode": False,
                "prompt_used": "NORMAL",
            }
        
        # Step 5: Generate MAIN answer using filtered docs (with inline citations)
        print(f"{'='*80}")
        print(f"STAGE 4: MAIN ANSWER GENERATION")
        print(f"{'='*80}")

        refine_chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=stuff_prompt_normal,
            verbose=True,
        )

        out = refine_chain.invoke({
            "input_documents": filtered_docs,  # ← Uses ALL docs (including page=0)
            "question": question,
        })

        main_answer_with_citations: str = out.get("output_text", "")

        print(f"Main answer generated (with inline citations)")
        print(f"{'='*80}")
        print(f"FINAL MAIN ANSWER (with citations):")
        print(f"{'-'*80}")
        print(main_answer_with_citations)
        print(f"{'-'*80}\n")
        print(f"{'='*80}\n")

        # Split trailing JSON (if present) from the answer body
        answer_body, answer_json = _split_answer_and_json(main_answer_with_citations)
        tables_payload = _normalize_tables_payload((answer_json or {}).get("tables"))
        answer_text = (answer_json or {}).get("answer_text") or answer_body
        answer_text = _strip_markdown_tables_from_text(str(answer_text))

        # =====================================================================
        # âœ… STEP 6: GENERATE EXCEPTION SECTION (separate retrieval + generation)
        # =====================================================================
        exception_result = retrieve_and_generate_exceptions(
            question=question,
            main_answer=main_answer_with_citations,
            llm=llm
        )

        exception_section = exception_result["exception_section"]
        exception_docs = exception_result["exception_docs"]
        has_exceptions = exception_result["has_exceptions"]

        print(f"{'='*80}")
        print(f"FINAL EXCEPTION SECTION (with citations):")
        print(f"{'-'*80}")
        if exception_section:
            print(exception_section)
        else:
            print("(No exception section generated)")
        print(f"{'-'*80}\n")
        print(f"{'='*80}\n")

        # =====================================================================
        # âœ… STEP 7: EXTRACT CITATIONS & GENERATE UNIFIED REFERENCE LIST
        # =====================================================================
        print(f"{'='*80}")
        print(f"STAGE 5: CITATION EXTRACTION & REFERENCE GENERATION")
        print(f"{'='*80}")

        # Combine all available documents (main + exception)
        all_available_docs = filtered_docs + exception_docs

        # Extract citations from main answer
        main_citations = extract_citations_from_text(answer_body)

        # Extract citations from exception section (if exists)
        exception_citations = []
        if exception_section:
            exception_citations = extract_citations_from_text(exception_section)

        # Combine all citations
        all_citations = main_citations + exception_citations

        # Map citations to doc IDs
        cited_doc_ids = map_citations_to_doc_ids(all_citations, all_available_docs)

        print(f"{'='*80}\n")

        # =====================================================================
        # âœ… STEP 8: COMBINE ANSWER COMPONENTS
        # =====================================================================
        print(f"{'='*80}")
        print(f"STAGE 6: FINAL ANSWER ASSEMBLY")
        print(f"{'='*80}")

        # Start with main answer
        main_answer = answer_body

        # CRITICAL: Remove any LLM-generated "References:" section from main answer
        # This is belt-and-suspenders with the prompt instruction
        references_pattern = r'\n+\*\*References?:?\*\*.*'
        if re.search(references_pattern, main_answer, re.DOTALL):
            main_answer = re.sub(references_pattern, '', main_answer, flags=re.DOTALL)
            print(f"Stripped LLM-generated References section from main answer")
        if re.search(references_pattern, answer_text, re.DOTALL):
            answer_text = re.sub(references_pattern, '', answer_text, flags=re.DOTALL)

        # DON'T combine exception section with main answer here
        # Let the frontend handle combining and formatting for better control
        # Both sections will go through format_visible_answer() independently in the UI

        # Final answer (main answer only, WITH citations - will be formatted in UI layer)
        final_answer = answer_text

        # final_answer = strip_inline_citations_with_llm(final_answer)

        # Exception section (WITH citations - will be formatted in UI layer)
        final_exception_section = exception_section if exception_section else ""
        # final_exception_section = strip_inline_citations_with_llm(final_exception_section)

        print(f"Final answer components prepared")
        print(f"  - Main answer: ✓")
        print(f"  - Exception section: {'✓' if exception_section else '✗'}")
        print(f"  - Citations kept for UI formatting: ✓")
        print(f"  - Total documents used: {len(all_available_docs)}")
        print(f"  - Documents cited: {len(cited_doc_ids)}")
        print(f"  - Frontend will combine sections with proper formatting")
        print(f"{'='*80}\n")

        # =====================================================================
        # âœ… STEP 9: FILTER SOURCES FOR DISPLAY (UI + PDF)
        # =====================================================================
        # Return ALL documents used (main + exception), not just cited ones
        # Filter out page=0 references for display
        display_sources = filter_page_zero_references(all_available_docs)

        print(f"Sources prepared for display: {len(display_sources)} documents (page=0 filtered out)")
        
        # Step 8: Handle stage answers
        inter: List[str] = out.get("intermediate_steps", []) or []
        stage_answers: Dict[str, str] = {}
        
        if len(offsets) >= 1 and len(inter) >= offsets[0]:
            stage_answers["STAGE 1 (A)"] = inter[offsets[0]-1]
        if len(offsets) >= 2 and len(inter) >= offsets[1]:
            stage_answers["STAGE 2 (A+B)"] = inter[offsets[1]-1]
        if len(offsets) >= 3 and len(inter) >= offsets[2]:
            stage_answers["STAGE 3 (A+B+C)"] = inter[offsets[2]-1]
        if len(offsets) >= 4 and len(inter) >= offsets[3]:
            stage_answers["STAGE 4 (A+B+C+EY)"] = inter[offsets[3]-1]

        # Print what's being sent to frontend
        print(f"\n{'='*80}")
        print(f"DATA BEING SENT TO FRONTEND")
        print(f"{'='*80}")
        print(f"answer (main answer): {len(final_answer)} characters")
        print(f"tables: {len(tables_payload)}")
        print(f"exception_section: {len(final_exception_section)} characters" if final_exception_section else "exception_section: (empty)")
        print(f"sources: {len(display_sources)} documents")
        print(f"stage_answers: {len(stage_answers)} stages")
        print(f"\n>> The frontend will combine 'answer' and 'exception_section' for display")
        print(f"{'='*80}\n")

        return {
            "answer": final_answer,
            "answer_text": final_answer,
            "tables": tables_payload,
            "exception_section": final_exception_section,  # ← Exception section (separate from main answer)
            "stage_answers": stage_answers,
            "sources": display_sources,  # ← âœ… Filtered sources (no page=0)
            "offsets": offsets,
            "thresholds": None,
            "scores": None,
            "ooc_mode": False,
            "prompt_used": "NORMAL",
        }
    
    else:
        # Out of context - no relevant sources
        return {
            "answer": "I DON'T KNOW",
            "answer_text": "I DON'T KNOW",
            "tables": [],
            "exception_section": "",  # No exception section for out-of-context queries
            "stage_answers": None,
            "sources": None,
            "offsets": None,
            "thresholds": None,
            "scores": None,
            "ooc_mode": True,
            "prompt_used": "OOC",
        }

# ------------------- Formatting / Utilities (non-table) -------------------

def replace_keywords(text: str, path: str = REPLACE_EXCEL_PATH) -> str:
    if not os.path.exists(path):
        return text
    try:
        df = pd.read_excel(path, sheet_name="English")
        repl = pd.Series(df["Replacement"].values, index=df["Word"].str.lower()).to_dict()
        if not repl:
            return text
        pattern = re.compile("|".join(re.escape(k) for k in repl.keys()), re.IGNORECASE)
        return pattern.sub(lambda m: repl.get(m.group().lower(), m.group()), text)
    except Exception:
        return text

def emphasize_headers(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(
        f"**{l}**" if re.match(r".+:\s*$", l) and not l.strip().startswith("**") else l
        for l in lines
    )

def bold_standards(text: str) -> str:
    pattern = re.compile(r"\b(?:IFRS|IAS|IFRIC|SIC)\s?(?:\d{1,3}|for\s+SMEs)\b", re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", text)

def _dedupe_tokens_case_insensitive(s: str) -> str:
    tokens = s.split()
    out, prev = [], None
    for t in tokens:
        if prev is None or t.lower() != prev.lower():
            out.append(t)
        prev = t
    return " ".join(out)

def fix_citation_format(text: str) -> str:
    if not isinstance(text, str) or "[" not in text or "]" not in text:
        return text
    text = re.sub(r"\]\s*\n+\s*\[", "] [", text)
    def _clean_block(m):
        inside = m.group(1)
        inside = re.sub(r"\s+", " ", inside).strip()
        parts = [p.strip() for p in inside.split(";")]
        parts = [_dedupe_tokens_case_insensitive(p) for p in parts if p]
        return "[" + "; ".join(parts) + "]"
    return re.sub(r"\[([^\]]+)\]", _clean_block, text)

def remove_citations(text: str) -> str:
    """
    Remove both square-bracket and round-bracket citations from text.

    Examples:
    - Square brackets: [1], [2], [3]
    - Round brackets: (IFRS A - ias-38 - para 60), (PwC - 21 - Intangible assets (IAS 38) - para 69)
    """
    if not text:
        return ""

    # Remove square-bracket citations [1], [2], etc.
    text = re.sub(r"\[.*?\]", "", text)

    # Remove round-bracket citations like (IFRS A - standard - para X)
    # Pattern matches: (Source - ... - para/Example/IG ...)
    text = re.sub(r'\([A-Z][^\)]*?(?:para|Example|IG)[^\)]*?\)', '', text)

    return text.strip()

def format_visible_answer(text: str) -> str:
    text = replace_keywords(text)
    text = emphasize_headers(text)
    text = bold_standards(text)
    text = fix_citation_format(text)
    text = remove_citations(text)
    return text.strip()

def _make_excerpt(text: str, max_chars: int = 900) -> str:
    if not text:
        return "—"
    text = text.strip()
    return (text[:max_chars] + " …") if len(text) > max_chars else text

# def _reference_title(header: str, page, source: str, chapter: str , chapter_name: str , scores:float , thresholds:float) -> str:
#     source_name = os.path.basename(source) if isinstance(source, str) else source
#     page_disp = f"p.{page}" if (isinstance(page, int) or str(page).isdigit()) else str(page)
#     chap_disp = chapter if chapter else ""
#     scores = scores if scores else 0.
#     thresholds = thresholds if thresholds else 0.
#     chap_name_disp = chapter_name if chapter_name else ""
#     parts = [h for h in [header,scores,thresholds, chap_disp,chap_name_disp, page_disp, source_name] if h]
#     return " • ".join(parts) if parts else "Reference"

def _reference_title(header: str, page, source: str, chapter: str, chapter_name: str, 
                     para_number, scores: float, thresholds: float) -> str:
    """Build reference title including paragraph number."""
    source_name = os.path.basename(source) if isinstance(source, str) else source
    page_disp = f"p.{page}" if (isinstance(page, int) or str(page).isdigit()) else str(page)
    chap_disp = chapter if chapter else ""
    
    # Include paragraph number if available
    para_disp = f"para {para_number}" if para_number and str(para_number).lower() not in {"none", "unnumbered", ""} else ""
    
    chap_name_disp = chapter_name if chapter_name else ""
    scores = scores if scores else 0.
    thresholds = thresholds if thresholds else 0.
    
    parts = [h for h in [header, para_disp, chap_disp, chap_name_disp, page_disp, source_name] if h]
    return " • ".join(parts) if parts else "Reference"

def _format_duration(seconds: float) -> str:
    if seconds is None:
        return "0 minutes and 0 seconds"
    if seconds < 0:
        seconds = 0
    total = int(round(seconds))
    m, s = divmod(total, 60)
    return f"{m} minutes and {s} seconds"

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")

def _protect_markdown_blocks(text: str) -> Tuple[str, Dict[str, str]]:
    if not text:
        return text, {}

    placeholders: Dict[str, str] = {}
    placeholder_idx = 0

    def _stash(chunk: str, prefix: str) -> str:
        nonlocal placeholder_idx
        key = f"<<{prefix}_{placeholder_idx}>>"
        placeholder_idx += 1
        placeholders[key] = chunk
        return key

    # Protect code fences to avoid accidental table detection/translation.
    text = _CODE_FENCE_RE.sub(lambda m: _stash(m.group(0), "CODE_BLOCK"), text)

    # Protect inline code spans.
    text = _INLINE_CODE_RE.sub(lambda m: _stash(m.group(0), "INLINE_CODE"), text)

    return text, placeholders

def _restore_placeholders(text: str, placeholders: Dict[str, str]) -> str:
    if not placeholders:
        return text
    for key, chunk in placeholders.items():
        text = text.replace(key, chunk)
    return text

def _translate_markdown_table_block(table_md: str) -> str:
    try:
        df = _md_table_to_df(table_md)
    except Exception:
        return table_md

    df = df.fillna("")
    headers = [str(c).strip() for c in df.columns]

    lines = []
    for i, h in enumerate(headers):
        lines.append(f"H{i:03d}||{h}")
    for r in range(len(df)):
        for c, h in enumerate(headers):
            lines.append(f"C{r:03d}_{c:03d}||{str(df.iloc[r, c]).strip()}")

    payload = "\n".join(lines)
    prompt = (
        "Translate the text to Arabic line-by-line.\n"
        "Rules:\n"
        "- Keep the prefix (H### or C###_###), the '||' delimiter, and any punctuation intact.\n"
        "- Preserve numbers, percentages, and source tags in parentheses.\n"
        "- Do not join or split lines; return the same number of lines.\n\n"
        "INPUT:\n"
        f"{payload}\n\n"
        "OUTPUT:"
    )

    resp = LLM.invoke([
        {"role": "system", "content": "You are a precise line-by-line translator that preserves formatting."},
        {"role": "user", "content": prompt},
    ])
    raw = (resp.content or "").strip()
    if not raw:
        return table_md

    translated_map: Dict[str, str] = {}
    for line in raw.splitlines():
        if "||" not in line:
            continue
        key, content = line.split("||", 1)
        translated_map[key.strip()] = content

    if len(translated_map) < len(lines):
        return table_md

    t_headers = [translated_map.get(f"H{i:03d}", headers[i]) for i in range(len(headers))]
    out_lines = [_mk_pipe_row(t_headers), _mk_pipe_sep(len(t_headers))]
    for r in range(len(df)):
        row_cells = []
        for c in range(len(headers)):
            key = f"C{r:03d}_{c:03d}"
            row_cells.append(translated_map.get(key, str(df.iloc[r, c]).strip()))
        out_lines.append(_mk_pipe_row(row_cells))

    return "\n".join(out_lines)

def _extract_and_translate_tables(text: str) -> Tuple[str, Dict[str, str]]:
    if not text:
        return text, {}

    placeholders: Dict[str, str] = {}
    lines = text.splitlines()
    blocks = _find_markdown_tables(text)
    if not blocks:
        return text, {}

    rebuilt = []
    cursor = 0
    for i, (start, end, block_txt) in enumerate(blocks):
        if start > cursor:
            rebuilt.append("\n".join(lines[cursor:start]))
        key = f"<<TABLE_BLOCK_{i}>>"
        placeholders[key] = _translate_markdown_table_block(block_txt)
        rebuilt.append(key)
        cursor = end
    rebuilt.append("\n".join(lines[cursor:]))
    return "\n".join(rebuilt), placeholders

def _translate_preserve_format(text: str) -> str:
    if not text:
        return ""

    table_stripped_text, table_placeholders = _extract_and_translate_tables(text)
    protected_text, placeholders = _protect_markdown_blocks(table_stripped_text)
    lines = protected_text.splitlines()
    payload = "\n".join(f"{i:04d}||{line}" for i, line in enumerate(lines))

    prompt = (
        "Translate the text to Arabic line-by-line.\n"
        "Rules:\n"
        "- Keep the 4-digit line number and the '||' delimiter unchanged.\n"
        "- Preserve all markdown symbols (#, -, *, >, |) and spacing.\n"
        "- Do NOT translate placeholders like <<CODE_BLOCK_0>>, <<TABLE_BLOCK_0>>, <<INLINE_CODE_0>>.\n"
        "- Do not join or split lines; return the same number of lines.\n\n"
        "INPUT:\n"
        f"{payload}\n\n"
        "OUTPUT:"
    )

    resp = LLM.invoke([
        {"role": "system", "content": "You are a precise line-by-line translator that preserves formatting."},
        {"role": "user", "content": prompt},
    ])
    raw = (resp.content or "").strip()

    out_lines = []
    for line in raw.splitlines():
        if "||" not in line:
            continue
        _, content = line.split("||", 1)
        out_lines.append(content)

    if len(out_lines) != len(lines):
        # Fallback to original if the line mapping breaks.
        return text

    translated = "\n".join(out_lines)
    translated = _restore_placeholders(translated, placeholders)
    translated = _restore_placeholders(translated, table_placeholders)
    return translated

def translate_to_arabic(text: str) -> str:
    try:
        out = _translate_preserve_format(text)
        out = remove_citations(fix_citation_format(out))
        return out
    except Exception as e:
        return f"️ Translation failed: {e}"

# def _unify_metadata(meta: dict) -> dict:
#     if not isinstance(meta, dict):
#         meta = {}
#     doc_name_raw = meta.get("doc_name") or meta.get("document_name") or meta.get("source") or meta.get("document") or ""
#     doc_name = str(doc_name_raw) or "Document"
#     publisher = meta.get("publisher") or ""
#     chapter = meta.get("chapter") or meta.get("chapter_title") or ""
#     chapter_name = meta.get("chapter_name") or ""
#     header = meta.get("header") or meta.get("section") or meta.get("title") or ""
#     page = meta.get("page", meta.get("page_number", meta.get("start_page", 0)))
#     try:
#         page = int(page) if page is not None and page != "" else 0
#     except Exception:
#         page = 0
#     unified = {
#         "doc_name": doc_name,
#         "publisher": publisher,
#         "chapter": str(chapter) if chapter is not None else "",
#         "header": str(header) if header is not None else "",
#         "page": page,
#         "chapter_name" : chapter_name,
#     }
#     return unified

def _unify_metadata(meta: dict) -> dict:
    if not isinstance(meta, dict):
        meta = {}
    
    doc_name_raw = meta.get("doc_name") or meta.get("document_name") or meta.get("source") or meta.get("document") or ""
    doc_name = str(doc_name_raw) or "Document"
    publisher = meta.get("publisher") or ""
    chapter = meta.get("chapter") or meta.get("chapter_title") or ""
    chapter_name = meta.get("chapter_name") or ""
    header = meta.get("header") or meta.get("section") or meta.get("title") or ""
    para_number = meta.get("para_number") or meta.get("paragraph") or ""  # NEW
    
    page = meta.get("page", meta.get("page_number", meta.get("start_page", 0)))
    try:
        page = int(page) if page is not None and page != "" else 0
    except Exception:
        page = 0
    
    unified = {
        "doc_name": doc_name,
        "publisher": publisher,
        "chapter": str(chapter) if chapter is not None else "",
        "header": str(header) if header is not None else "",
        "page": page,
        "chapter_name": chapter_name,
        "para_number": str(para_number) if para_number else "",  # NEW
    }
    return unified


# ------------------- PDF Export Helper Functions -------------------

_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_HTML_TAG_RE = re.compile(r"(<[^>]+>)")

def _contains_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text or ""))

def _shape_arabic_text(text: str) -> str:
    if not text:
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except Exception:
        return text
    lines = text.splitlines()
    shaped = []
    for line in lines:
        reshaped = arabic_reshaper.reshape(line)
        shaped.append(get_display(reshaped))
    return "\n".join(shaped)

def _shape_arabic_preserve_tags(text: str) -> str:
    if not _contains_arabic(text):
        return text
    parts = _HTML_TAG_RE.split(text)
    out = []
    for part in parts:
        if part.startswith("<") and part.endswith(">"):
            out.append(part)
        else:
            out.append(_shape_arabic_text(part))
    return "".join(out)

def _md_to_html(text: str, rtl: bool = False) -> str:
    text = sanitize_text(text)
    text = remove_citations(fix_citation_format(text))
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    html = html.replace("\n", "<br/>")
    return _shape_arabic_preserve_tags(html) if rtl else html

def _register_arabic_font() -> str:
    if not REPORTLAB_AVAILABLE:
        return ""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        return ""

    font_candidates = [
        os.getenv("ARABIC_PDF_FONT_PATH", ""),
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in font_candidates:
        if path and os.path.exists(path):
            try:
                font_name = "ArabicFont"
                pdfmetrics.registerFont(TTFont(font_name, path))
                return font_name
            except Exception:
                continue
    return ""


def _split_into_segments(text_block: str) -> List[Tuple[str, Union[str, pd.DataFrame]]]:
    text_block = _canonicalize_all_tables(text_block or "")
    lines = text_block.splitlines()
    tables = _find_markdown_tables(text_block)
    segments: List[Tuple[str, Union[str, pd.DataFrame]]] = []
    cursor = 0
    for (start, end, block) in tables:
        if start > cursor:
            prose = "\n".join(lines[cursor:start]).strip()
            if prose:
                segments.append(("text", prose))
        try:
            df = _md_table_to_df(block)
            df = _drop_leading_empty_column(df)
            segments.append(("table", df))
        except Exception:
            segments.append(("text", block))
        cursor = end
    if cursor < len(lines):
        tail = "\n".join(lines[cursor:]).strip()
        if tail:
            segments.append(("text", tail))
    return segments


def _tables_payload_to_dfs(tables_payload: Any) -> List[Tuple[str, pd.DataFrame]]:
    out: List[Tuple[str, pd.DataFrame]] = []
    for t in _normalize_tables_payload(tables_payload):
        cols = t.get("columns") or []
        rows = t.get("rows") or []
        try:
            df = pd.DataFrame(rows, columns=cols)
        except Exception:
            continue
        df = df.fillna("").applymap(_normalize_table_cell)
        df = _drop_leading_empty_column(df)
        out.append((t.get("table_name") or "", df))
    return out


def _build_pdf_reportlab(history: list) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    meta_style = ParagraphStyle("meta", parent=styles["Normal"], textColor=colors.grey, fontSize=9)
    user_style = ParagraphStyle("user", parent=styles["Normal"], spaceBefore=6, spaceAfter=2, alignment=TA_LEFT)
    bot_style = ParagraphStyle("bot", parent=styles["Normal"], spaceBefore=2, spaceAfter=10, alignment=TA_LEFT)
    header_style = styles["Heading3"]
    excerpt_style = ParagraphStyle("excerpt", parent=styles["Normal"], fontSize=9, textColor=colors.black)
    cell_style = ParagraphStyle("cell", parent=styles["Normal"], fontSize=8, leading=10, spaceAfter=0, spaceBefore=0, alignment=TA_LEFT)

    arabic_font = _register_arabic_font()
    arabic_font_name = arabic_font or styles["Normal"].fontName
    user_style_rtl = ParagraphStyle(
        "user_rtl",
        parent=styles["Normal"],
        fontName=arabic_font_name,
        spaceBefore=6,
        spaceAfter=2,
        alignment=TA_RIGHT,
    )
    bot_style_rtl = ParagraphStyle(
        "bot_rtl",
        parent=styles["Normal"],
        fontName=arabic_font_name,
        spaceBefore=2,
        spaceAfter=10,
        alignment=TA_RIGHT,
    )
    cell_style_rtl = ParagraphStyle(
        "cell_rtl",
        parent=styles["Normal"],
        fontName=arabic_font_name,
        fontSize=8,
        leading=10,
        spaceAfter=0,
        spaceBefore=0,
        alignment=TA_RIGHT,
    )

    story = []
    story.append(Paragraph("IFRS Chat – Conversation Export", title_style))
    story.append(Spacer(1, 8))

    tbl_style = TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ])

    usable_width = doc.width

    def _render_table_df(df: pd.DataFrame, title: str = ""):
        df = _drop_leading_empty_column(df)
        if df is None or df.empty:
            return
        if title:
            story.append(Paragraph(f"<b>{sanitize_text(title)}</b>", styles["Normal"]))
        table_rtl = _contains_arabic(" ".join([str(c) for c in df.columns] + df.astype(str).values.flatten().tolist()))
        cell_style_use = cell_style_rtl if table_rtl else cell_style
        header_cells = [Paragraph(f"<b>{_shape_arabic_text(sanitize_text(str(c))) if table_rtl else sanitize_text(str(c))}</b>", cell_style_use) for c in df.columns]
        data_rows = []
        for _, row in df.iterrows():
            row_cells = []
            for c in df.columns:
                cell_text = sanitize_text(str(row[c]))
                if table_rtl:
                    cell_text = _shape_arabic_text(cell_text)
                row_cells.append(Paragraph(cell_text, cell_style_use))
            data_rows.append(row_cells)
        data = [header_cells] + data_rows

        counts = []
        for c in df.columns:
            col_vals = [str(c)] + df[c].astype(str).tolist()
            counts.append(max(len(v) for v in col_vals))
        tot = sum(counts) or 1
        MIN_W = 36
        col_widths = [max(MIN_W, usable_width * (cnt / tot)) for cnt in counts]
        scale = usable_width / sum(col_widths)
        col_widths = [w * scale for w in col_widths]

        t = Table(data, colWidths=col_widths, repeatRows=1, splitByRow=1, hAlign="RIGHT" if table_rtl else "LEFT")
        t.setStyle(tbl_style)
        t.setStyle(TableStyle([
            ("WORDWRAP", (0, 0), (-1, -1), True),
            ("ALIGN", (0, 0), (-1, -1), "RIGHT" if table_rtl else "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)
        story.append(Spacer(1, 6))

    def _emit_segments(label: str, text_block: str):
        # Canonicalize first so we detect/parse neatly
        text_block = _canonicalize_all_tables(text_block or "")
        story.append(Paragraph(f"<b>{label}</b>", styles["Normal"]))
        segments = _split_into_segments(text_block)

        for kind, payload in segments:
            if kind == "text":
                rtl = _contains_arabic(payload)
                prose = _md_to_html(payload, rtl=rtl)
                style = bot_style_rtl if (rtl and label.startswith("Assistant")) else user_style_rtl if rtl else bot_style if label.startswith("Assistant") else user_style
                story.append(Paragraph(prose, style))
            else:
                df: pd.DataFrame = payload
                _render_table_df(df)

    def _emit_tables(tables_payload: Any):
        for title, df in _tables_payload_to_dfs(tables_payload):
            _render_table_df(df, title=title)

    for n, chat in enumerate(history):
        meta = "Answer from Database • IFRS A/B/C (Refine Chain)"
        story.append(Paragraph(meta, meta_style))
        _emit_segments("User:", chat.get("question", ""))
        _emit_segments("Assistant:", chat.get("answer", ""))
        _emit_tables(chat.get("tables"))
        dur_text = _format_duration(chat.get("time_taken_sec"))
        story.append(Paragraph(f"<b>Time taken to answer : {dur_text}.</b>", meta_style))

        stage_ans = chat.get("stage_answers") or {}
        if stage_ans:
            story.append(Paragraph("<b>Intermediate Answers:</b>", header_style))
            for k in ["STAGE 1 (A)", "STAGE 2 (A+B)", "STAGE 3 (A+B+C)", "STAGE 4 (A+B+C+EY)"]:
                val = stage_ans.get(k, "")
                if isinstance(val, str) and val.strip():
                    story.append(Paragraph(f"<b>{k}</b>", styles["Normal"]))
                    _emit_segments("", val)

        sources = chat.get("sources") or []
        sources = filter_page_zero_references(sources)
        if (chat.get("answer") or "").strip().lower() != "sources not found." and sources:
            story.append(Paragraph("<b>References (with excerpts):</b>", header_style))
            for doc_i in sources:
                m = _unify_metadata(getattr(doc_i, "metadata", {}) or {})
                source_db = m.get('doc_name', 'Document')

                # Match UI format: different fields for IFRS vs EY/PwC
                if source_db in ["IFRS A", "IFRS B", "IFRS C"]:
                    ref_parts = [
                        f"Document: {sanitize_text(source_db)}",
                        f"Standard Name: {sanitize_text(m.get('chapter_name') or '—')}",
                        f"Paragraph Number: {sanitize_text(m.get('para_number') or '—')}",
                        f"Header: {sanitize_text(m.get('header') or '—')}",
                        f"Page: {m.get('page', 0)}",
                        f"Publisher: {sanitize_text(m.get('publisher') or '—')}",
                    ]
                else:
                    ref_parts = [
                        f"Document: {sanitize_text(source_db)}",
                        f"Chapter Name: {sanitize_text(m.get('chapter_name') or '—')}",
                        f"Paragraph Number: {sanitize_text(m.get('para_number') or '—')}",
                        f"Header: {sanitize_text(m.get('header') or '—')}",
                        f"Page: {m.get('page', 0)}",
                        f"Publisher: {sanitize_text(m.get('publisher') or '—')}",
                    ]
                story.append(Paragraph(" | ".join(ref_parts), meta_style))
                chunk_text = getattr(doc_i, "page_content", "") or ""
                story.append(Paragraph(_md_to_html(_make_excerpt(chunk_text)), excerpt_style))
                story.append(Spacer(1, 4))

        if n < len(history) - 1:
            story.append(PageBreak())

    doc.build(story)
    return buf.getvalue()


def _build_pdf_fpdf(history: list) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "IFRS Chat – Conversation Export", ln=1)
    pdf.ln(2)

    def _get_fpdf_arabic_font():
        font_candidates = [
            os.getenv("ARABIC_PDF_FONT_PATH", ""),
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        for path in font_candidates:
            if path and os.path.exists(path):
                try:
                    pdf.add_font("ArabicFont", "", path, uni=True)
                    return "ArabicFont"
                except Exception:
                    continue
        return ""

    arabic_fpdf_font = _get_fpdf_arabic_font()

    def write_wrapped(text, style="", rtl=False):
        text = sanitize_text(text)
        text = remove_citations(fix_citation_format(text))
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        if rtl:
            text = _shape_arabic_text(text)
        for line in text.split("\n"):
            if rtl and arabic_fpdf_font:
                pdf.set_font(arabic_fpdf_font, style, 11)
                pdf.multi_cell(0, 6, line, align="R")
            else:
                pdf.set_font("Arial", style, 11)
                pdf.multi_cell(0, 6, line)
        pdf.ln(1)

    def _compute_col_widths_fpdf(df, page_width):
        counts = []
        for c in df.columns:
            col_vals = [str(c)] + df[c].astype(str).tolist()
            counts.append(max(len(v) for v in col_vals))
        tot = sum(counts) or 1
        widths = [(page_width * (cnt / tot)) for cnt in counts]
        MIN_W = 18
        widths = [max(MIN_W, w) for w in widths]
        scale = page_width / sum(widths)
        return [w * scale for w in widths]

    def draw_table(df: pd.DataFrame, font_size=9, rtl=False):
        df = _drop_leading_empty_column(df)
        if df is None or df.empty:
            return
        page_width = pdf.w - pdf.l_margin - pdf.r_margin

        # Proportional widths with floor; normalize to fit
        counts = []
        for c in df.columns:
            col_vals = [str(c)] + df[c].astype(str).tolist()
            counts.append(max(len(v) for v in col_vals))
        tot = sum(counts) or 1
        widths = [(page_width * (cnt / tot)) for cnt in counts]
        MIN_W = 18
        widths = [max(MIN_W, w) for w in widths]
        scale = page_width / sum(widths)
        widths = [w * scale for w in widths]

        # Header
        if rtl and arabic_fpdf_font:
            pdf.set_font(arabic_fpdf_font, "B", font_size)
        else:
            pdf.set_font("Arial", "B", font_size)
        for j, col in enumerate(df.columns):
            cell_text = sanitize_text(str(col))
            if rtl:
                cell_text = _shape_arabic_text(cell_text)
            pdf.cell(widths[j], 6, cell_text, border=1, align="R" if rtl else "L")
        pdf.ln(6)

        # Rows
        if rtl and arabic_fpdf_font:
            pdf.set_font(arabic_fpdf_font, "", max(8, font_size - 1 if len(df.columns) >= 7 else font_size))
        else:
            pdf.set_font("Arial", "", max(8, font_size - 1 if len(df.columns) >= 7 else font_size))
        for _, row in df.iterrows():
            y_start = pdf.get_y()
            max_y = y_start
            x_start = pdf.get_x()
            for j, col in enumerate(df.columns):
                x = pdf.get_x()
                y = pdf.get_y()
                cell_text = sanitize_text(str(row[col]))
                if rtl:
                    cell_text = _shape_arabic_text(cell_text)
                pdf.multi_cell(widths[j], 6, cell_text, border=1, align="R" if rtl else "L")
                max_y = max(max_y, pdf.get_y())
                pdf.set_xy(x + widths[j], y)
            pdf.set_xy(x_start, max_y)

    def emit_segments(label: str, text_block: str):
        text_block = _canonicalize_all_tables(text_block or "")
        pdf.set_font("Arial", "B", 11)
        pdf.multi_cell(0, 6, label)
        segments = _split_into_segments(text_block)
        for kind, payload in segments:
            if kind == "text":
                rtl = _contains_arabic(payload)
                write_wrapped(payload, style="", rtl=rtl)
            else:
                table_rtl = _contains_arabic(" ".join([str(c) for c in payload.columns] + payload.astype(str).values.flatten().tolist()))
                draw_table(payload, font_size=9, rtl=table_rtl)

    def emit_tables(tables_payload: Any):
        for title, df in _tables_payload_to_dfs(tables_payload):
            if title:
                write_wrapped(title, style="B")
            table_rtl = _contains_arabic(" ".join([str(c) for c in df.columns] + df.astype(str).values.flatten().tolist()))
            draw_table(df, font_size=9, rtl=table_rtl)

    for chat in history:
        pdf.set_text_color(120, 120, 120)
        write_wrapped("Answer from Database • IFRS A/B/C (Refine Chain)")
        pdf.set_text_color(0, 0, 0)
        emit_segments("User:", chat.get("question", ""))
        emit_segments("Assistant:", chat.get("answer", ""))
        emit_tables(chat.get("tables"))
        dur_text = _format_duration(chat.get("time_taken_sec"))
        pdf.set_font("Arial", "B", 10)
        write_wrapped(f"Time taken to answer : {dur_text}.", style="B")

        stage_ans = chat.get("stage_answers") or {}
        if stage_ans:
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 6, "Intermediate Answers:", ln=1)
            pdf.set_font("Arial", "", 10)
            for k in ["STAGE 1 (A)", "STAGE 2 (A+B)", "STAGE 3 (A+B+C)", "STAGE 4 (A+B+C+EY)"]:
                val = stage_ans.get(k, "")
                if isinstance(val, str) and val.strip():
                    write_wrapped(f"{k}:", style="B")
                    segments = _split_into_segments(val)
                    for kind, payload in segments:
                        if kind == "text":
                            write_wrapped(payload)
                        else:
                            draw_table(payload, font_size=8)

        sources = chat.get("sources") or []
        #  ADD THIS: Filter page=0 from PDF export
        sources = filter_page_zero_references(sources)
        if (chat.get("answer") or "").strip().lower() != "sources not found." and sources:
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 6, "References (with excerpts):", ln=1)
            pdf.set_font("Arial", "", 10)
            for doc_i in sources:
                m = _unify_metadata(getattr(doc_i, "metadata", {}) or {})
                parts = [
                    f"Document: {sanitize_text(m.get('doc_name','Document'))}",
                    f"Chapter_OR_Standard: {sanitize_text(m.get('chapter') or '—')}",
                    f"Chapter Name: {sanitize_text(m.get('chapter_name') or '—')}",
                    f"Paragraph: {sanitize_text(m.get('para_number') or '—')}",
                    f"Header: {sanitize_text(m.get('header') or '—')}",
                    f"Page: {m.get('page', 0)}",
                    f"Publisher: {sanitize_text(m.get('publisher') or '—')}",
                ]
                write_wrapped(" | ".join(parts))
                chunk_text = getattr(doc_i, "page_content", "") or ""
                s = _make_excerpt(chunk_text)
                write_wrapped(s)

    out = BytesIO()
    out.write(pdf.output(dest="S").encode("latin1", errors="ignore"))
    return out.getvalue()


def _df_to_html_table(df: pd.DataFrame, rtl: bool = False) -> str:
    df = _drop_leading_empty_column(df)
    if df is None or df.empty:
        return ""
    ths = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        tds = "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in df.columns)
        rows.append(f"<tr>{tds}</tr>")
    dir_attr = ' dir="rtl"' if rtl else ""
    return f'<table{dir_attr}><thead><tr>{ths}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _build_html_export(history: list) -> str:
    css = """
    body { font-family: Arial, sans-serif; color: #111; }
    .message { margin: 18px 0; }
    .label { font-weight: 700; margin: 10px 0 6px; }
    .markdown { line-height: 1.6; }
    .rtl { direction: rtl; text-align: right; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0 14px; font-size: 13px; }
    th, td { border: 1px solid #999; padding: 6px 8px; text-align: left; vertical-align: top; }
    .rtl th, .rtl td { text-align: right; }
    """
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        f"<style>{css}</style></head><body>",
        "<h1>IFRS Chat – Conversation Export</h1>",
    ]

    for chat in history:
        parts.append("<div class='message'>")
        parts.append("<div class='label'>Answer from Database • IFRS A/B/C (Refine Chain)</div>")

        for label, text_block in [("User:", chat.get("question", "")), ("Assistant:", chat.get("answer", ""))]:
            text_block = _canonicalize_all_tables(text_block or "")
            segments = _split_into_segments(text_block)
            parts.append(f"<div class='label'>{html.escape(label)}</div>")
            for kind, payload in segments:
                if kind == "text":
                    rtl = _contains_arabic(payload)
                    prose = _md_to_html(payload, rtl=rtl)
                    cls = "markdown rtl" if rtl else "markdown"
                    parts.append(f"<div class='{cls}'>{prose}</div>")
                else:
                    df = payload
                    table_rtl = _contains_arabic(" ".join([str(c) for c in df.columns] + df.astype(str).values.flatten().tolist()))
                    cls = "markdown rtl" if table_rtl else "markdown"
                    parts.append(f"<div class='{cls}'>")
                    parts.append(_df_to_html_table(df, rtl=table_rtl))
                    parts.append("</div>")
            if label.startswith("Assistant"):
                for title, df in _tables_payload_to_dfs(chat.get("tables")):
                    table_rtl = _contains_arabic(" ".join([str(c) for c in df.columns] + df.astype(str).values.flatten().tolist()))
                    cls = "markdown rtl" if table_rtl else "markdown"
                    if title:
                        parts.append(f"<div class='label'>{html.escape(title)}</div>")
                    parts.append(f"<div class='{cls}'>")
                    parts.append(_df_to_html_table(df, rtl=table_rtl))
                    parts.append("</div>")

        dur_text = _format_duration(chat.get("time_taken_sec"))
        parts.append(f"<div class='label'>Time taken to answer : {html.escape(dur_text)}.</div>")
        parts.append("</div>")

    parts.append("</body></html>")
    return "".join(parts)

"""Formatting utilities separated from the main engine module."""

import os
import re
import unicodedata
from typing import Any, Dict

import pandas as pd

from rag_config import REPLACE_EXCEL_PATH

_SANITIZE_MAP = {
    "\u25a0": "-",
    "\u25aa": "-",
    "\u25cf": "-",
    "\u2022": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u00ad": "-",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2015": "-",
    "\ufe58": "-",
    "\ufe63": "-",
    "\u00a0": " ",
    "\u200b": "",
    "\ufeff": "",
}


def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = "".join(char for char in s if unicodedata.category(char) != "Mn")
    for k, v in _SANITIZE_MAP.items():
        s = s.replace(k, v)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r"[ \t]+", " ", s)
    s = s.replace("```", "")
    return s


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
        seen, out = set(), []
        for p in parts:
            key = p.lower()
            if key and key not in seen:
                seen.add(key)
                out.append(p)
        return "[" + "; ".join(out) + "]"

    text = re.sub(r"\[([^\]]+)\]", _clean_block, text)
    text = re.sub(r"(?:\s*\[[^\]]+\]){2,}", lambda m: " " + _merge_adjacent_brackets(m.group(0)), text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _merge_adjacent_brackets(s: str) -> str:
    contents = re.findall(r"\[([^\]]+)\]", s)
    parts = []
    for c in contents:
        parts.extend([p.strip() for p in c.split(";") if p.strip()])
    seen, out = set(), []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return "[" + "; ".join(out) + "]" if out else ""


def remove_citations(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s*\[[^\]]*\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_visible_answer(text: str) -> str:
    text = sanitize_text(text)
    text = replace_keywords(text)
    text = emphasize_headers(text)
    text = bold_standards(text)
    text = fix_citation_format(text)
    text = _dedupe_tokens_case_insensitive(text)
    return text


def _make_excerpt(txt: str, max_chars: int = 320) -> str:
    s = re.sub(r"\s+", " ", (txt or "")).strip()
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + " ..."


def _reference_title(meta: Dict[str, Any]) -> str:
    if not isinstance(meta, dict):
        return "Reference"
    db = meta.get("source_db") or meta.get("publisher") or "Source"
    chap = meta.get("chapter_name") or meta.get("chapter") or ""
    para = meta.get("para_number")
    head = meta.get("header")
    parts = [str(db)]
    if chap:
        parts.append(str(chap))
    if para and str(para).lower() not in {"none", "unnumbered", ""}:
        parts.append(f"para {para}")
    if head and str(head).strip() and str(head).strip() != "—":
        parts.append(str(head))
    return " — ".join(parts)


def _format_duration(sec: float) -> str:
    try:
        x = float(sec)
    except Exception:
        return "—"
    if x < 1:
        return f"{int(round(x*1000))} ms"
    if x < 60:
        return f"{x:.2f} s"
    m = int(x // 60)
    s = x % 60
    return f"{m}m {s:.1f}s"


def _unify_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(meta or {})
    source_db = m.get("source_db") or m.get("publisher") or ""
    chapter = m.get("chapter") or ""
    chapter_name = m.get("chapter_name") or chapter or ""
    para = m.get("para_number") or ""
    header = m.get("header") or ""
    page = m.get("page") or 0

    m["source_db"] = source_db
    m["publisher"] = m.get("publisher") or source_db
    m["chapter"] = chapter
    m["chapter_name"] = chapter_name
    m["para_number"] = para
    m["header"] = header
    m["page"] = page
    m["title"] = _reference_title(m)
    m["doc_name"] = m.get("doc_name") or chapter_name or chapter or "Document"
    return m

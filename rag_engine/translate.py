"""Translation utilities for answer text.

This module keeps translation logic separate from the main RAG engine pipeline.
"""

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_client import get_llm

_TRANSLATE_PROMPT = (
    "Translate the following content to formal Arabic.\n"
    "Rules:\n"
    "1) Preserve markdown headings, bullet lists, and table structure exactly.\n"
    "2) Do not add commentary.\n"
    "3) Keep numbers, IFRS/IAS standard codes, and citation tokens unchanged.\n"
    "4) Return only translated content.\n\n"
    "Content:\n{content}"
)


def _normalize_line_endings(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _protect_code_fences(text: str) -> tuple[str, dict[str, str]]:
    placeholders = {}
    idx = 0

    def repl(m):
        nonlocal idx
        key = f"__CODE_BLOCK_{idx}__"
        placeholders[key] = m.group(0)
        idx += 1
        return key

    protected = re.sub(r"```[\s\S]*?```", repl, text)
    return protected, placeholders


def _restore_code_fences(text: str, placeholders: dict[str, str]) -> str:
    out = text
    for key, val in placeholders.items():
        out = out.replace(key, val)
    return out


def translate_to_arabic(text: str, llm: Any = None) -> str:
    """Translate answer text to Arabic while preserving markdown structure."""
    if not isinstance(text, str) or not text.strip():
        return ""

    llm = llm or get_llm("mini")
    normalized = _normalize_line_endings(text)
    protected, placeholders = _protect_code_fences(normalized)

    prompt = _TRANSLATE_PROMPT.format(content=protected)
    resp = llm.invoke(
        [
            SystemMessage(content="You are an expert bilingual accounting translator."),
            HumanMessage(content=prompt),
        ]
    )
    translated = getattr(resp, "content", "") or ""
    translated = _restore_code_fences(translated, placeholders)
    return translated.strip()

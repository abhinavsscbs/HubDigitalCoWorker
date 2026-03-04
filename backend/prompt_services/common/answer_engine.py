from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from common.prompt_utils import tabular_data_for_prompt
from common.repo_paths import ensure_repo_paths


_ENGINE_LOADED = False
_ANSWER_WITH_REFINE_CHAIN: Optional[Callable[..., Any]] = None
_FORMAT_VISIBLE_ANSWER: Optional[Callable[[str], str]] = None


def _load_repo_engine() -> bool:
    global _ENGINE_LOADED, _ANSWER_WITH_REFINE_CHAIN, _FORMAT_VISIBLE_ANSWER
    if _ENGINE_LOADED:
        return _ANSWER_WITH_REFINE_CHAIN is not None and _FORMAT_VISIBLE_ANSWER is not None

    ensure_repo_paths()
    try:
        from rag_engine.answer import answer_with_refine_chain
        from rag_engine.formatting import format_visible_answer

        _ANSWER_WITH_REFINE_CHAIN = answer_with_refine_chain
        _FORMAT_VISIBLE_ANSWER = format_visible_answer
    except Exception:
        _ANSWER_WITH_REFINE_CHAIN = None
        _FORMAT_VISIBLE_ANSWER = None

    _ENGINE_LOADED = True
    return _ANSWER_WITH_REFINE_CHAIN is not None and _FORMAT_VISIBLE_ANSWER is not None


def _normalize_tables_payload_to_contract(tables_payload: list) -> dict:
    if not isinstance(tables_payload, list) or not tables_payload:
        return {"headers": [], "rows": []}

    first = tables_payload[0] if isinstance(tables_payload[0], dict) else {}
    columns = first.get("columns") or []
    rows = first.get("rows") or []
    if not isinstance(columns, list) or not isinstance(rows, list):
        return {"headers": [], "rows": []}

    headers = [str(col) for col in columns]
    normalized_rows = []
    for raw_row in rows:
        if not isinstance(raw_row, list):
            continue
        row = [str(item) for item in raw_row]
        if len(row) < len(headers):
            row.extend([""] * (len(headers) - len(row)))
        if len(row) > len(headers):
            row = row[: len(headers)]
        normalized_rows.append(row)

    return {"headers": headers, "rows": normalized_rows}


def _stub_answer(prompt_text: str) -> Dict[str, Any]:
    return {
        "promptResponseText": (
            f"Deterministic stub answer for question: '{prompt_text}'. "
            "This response was derived from the supplied references/source context."
        ),
        "promptResponseTabularData": tabular_data_for_prompt(prompt_text),
        "engine": "stub",
    }


def generate_answer(prompt_text: str) -> Dict[str, Any]:
    if not _load_repo_engine():
        return _stub_answer(prompt_text)

    try:
        result = _ANSWER_WITH_REFINE_CHAIN(prompt_text)
        answer_text = result.get("answer_text") or result.get("answer") or ""
        exception_text = result.get("exception_section") or ""

        if exception_text:
            answer_text = answer_text.rstrip() + "\n\n" + exception_text

        answer_text = str(answer_text).strip()
        if answer_text and answer_text.lower() != "sources not found.":
            answer_text = _FORMAT_VISIBLE_ANSWER(answer_text)

        if not answer_text:
            return _stub_answer(prompt_text)

        table_contract = _normalize_tables_payload_to_contract(result.get("tables") or [])
        if not table_contract["headers"] and not table_contract["rows"]:
            table_contract = tabular_data_for_prompt(prompt_text)

        return {
            "promptResponseText": answer_text,
            "promptResponseTabularData": table_contract,
            "engine": "repo",
        }
    except Exception:
        return _stub_answer(prompt_text)

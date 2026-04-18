"""Answer-level interfaces and confidence model.

The full generation pipeline remains in `engine.py`, while confidence modeling
and answer contracts live here for easier maintenance.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ConfidenceBreakdown:
    similarity_component: float
    citation_coverage_component: float
    source_diversity_component: float


@dataclass
class ConfidenceResult:
    score: float
    label: str
    components: ConfidenceBreakdown
    reason: str


def classify_confidence(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def build_confidence_result(
    score: float,
    similarity_component: float,
    citation_coverage_component: float,
    source_diversity_component: float,
    reason: str,
) -> Dict[str, Any]:
    score = max(0.0, min(1.0, float(score)))
    return {
        "score": round(score, 3),
        "label": classify_confidence(score),
        "components": {
            "similarity_component": round(float(similarity_component), 3),
            "citation_coverage_component": round(float(citation_coverage_component), 3),
            "source_diversity_component": round(float(source_diversity_component), 3),
        },
        "reason": reason,
    }


def answer_with_refine_chain(question: str, llm=None):
    # Lazy import avoids circular import between engine and answer modules.
    from .engine import answer_with_refine_chain as _impl

    return _impl(question, llm=llm)


def retrieve_and_generate_exceptions(question: str, main_answer: str, llm):
    from .engine import retrieve_and_generate_exceptions as _impl

    return _impl(question=question, main_answer=main_answer, llm=llm)


def generate_unified_reference_list(all_docs: List[Any], cited_doc_ids: List[str]) -> str:
    from .engine import generate_unified_reference_list as _impl

    return _impl(all_docs=all_docs, cited_doc_ids=cited_doc_ids)

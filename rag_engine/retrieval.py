"""Retrieval module for FAISS-backed IFRS corpora.

This module owns retrieval primitives and DB configuration. It is intentionally
independent from `engine.py` so retrieval behavior is visible in one place.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import uuid

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag_config import DB_PATHS, EMBEDDINGS_DEVICE, EMBEDDINGS_MODEL, STAGE_2_PERCENTILE


@dataclass
class DBConfig:
    path: str
    name: str
    score_threshold: float = 0.65
    top_percentile: float = STAGE_2_PERCENTILE


EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS_MODEL,
    model_kwargs={"device": EMBEDDINGS_DEVICE},
)


def _has_langchain_index(dir_path: str) -> bool:
    return os.path.exists(os.path.join(dir_path, "index.faiss")) or os.path.exists(
        os.path.join(dir_path, "index.pkl")
    )


def load_index(dir_path: str) -> FAISS:
    if not _has_langchain_index(dir_path):
        raise RuntimeError(f"{dir_path} is not a LangChain FAISS index (index.faiss/index.pkl missing).")
    return FAISS.load_local(dir_path, EMBEDDINGS, allow_dangerous_deserialization=True)


def build_retriever(dir_path: str, k: int = 5):
    vs = load_index(dir_path)
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k, "fetch_k": max(40, 5 * k)})


def retrieve_docs_with_score(
    dir_path: str,
    question: str,
    score_threshold: Optional[float] = None,
    top_percentile: Optional[float] = None,
    max_k: int = 50,
) -> List[Tuple[Document, float]]:
    """Retrieve docs and convert FAISS L2 distance to cosine-like similarity in [0,1]."""
    vs = load_index(dir_path)
    docs_with_scores = vs.similarity_search_with_score(question, k=max_k)

    docs_with_similarity: List[Tuple[Document, float]] = []
    for doc, l2_distance in docs_with_scores:
        cosine_similarity = 1 - (l2_distance**2 / 2)
        cosine_similarity = max(0.0, min(1.0, cosine_similarity))
        docs_with_similarity.append((doc, cosine_similarity))

    if top_percentile is None and score_threshold is None:
        top_percentile = STAGE_2_PERCENTILE

    if top_percentile is not None and score_threshold is None:
        sorted_docs = sorted(docs_with_similarity, key=lambda x: x[1], reverse=True)
        cutoff_index = max(1, int(len(sorted_docs) * top_percentile))
        return sorted_docs[:cutoff_index]

    if score_threshold is not None:
        return [(doc, score) for doc, score in docs_with_similarity if score >= score_threshold]

    return docs_with_similarity


def _source_tag(meta: dict, db_label: str) -> str:
    para_num = (meta or {}).get("para_number")
    if db_label in ["IFRS A", "IFRS B", "IFRS C"]:
        standard_name = (meta or {}).get("chapter_name") or (meta or {}).get("chapter") or (meta or {}).get("source") or "Document"
        if para_num and str(para_num).strip() and str(para_num).lower() not in {"none", "unnumbered"}:
            return f"({db_label} - {standard_name} - para {para_num})"
        return f"({db_label} - {standard_name})"

    chapter_name = (meta or {}).get("chapter_name") or (meta or {}).get("source") or "Document"
    header = (meta or {}).get("header", "")
    if header:
        if para_num and str(para_num).strip() and str(para_num).lower() not in {"none", "unnumbered"}:
            return f"({db_label} - {chapter_name} - {header} - para {para_num})"
        return f"({db_label} - {chapter_name} - {header})"
    if para_num and str(para_num).strip() and str(para_num).lower() not in {"none", "unnumbered"}:
        return f"({db_label} - {chapter_name} - para {para_num})"
    return f"({db_label} - {chapter_name})"


def fetch_docs(question: str, cfg: DBConfig) -> List[Document]:
    docs_with_scores = retrieve_docs_with_score(
        cfg.path,
        question,
        score_threshold=None,
        top_percentile=cfg.top_percentile,
        max_k=50,
    )

    cleaned: List[Document] = []
    for d, similarity_score in docs_with_scores:
        content = (d.page_content or "").strip()
        tag = _source_tag(d.metadata, cfg.name)
        visible_chunk = f"{tag} {content}"
        unique_id = str(uuid.uuid4())[:8]

        cleaned.append(
            Document(
                page_content=visible_chunk,
                metadata={
                    **(d.metadata or {}),
                    "source_db": cfg.name,
                    "_doc_id": unique_id,
                    "_similarity_score": similarity_score,
                },
            )
        )
    return cleaned


DBS: List[DBConfig] = [
    DBConfig(path=item["path"], name=item["name"], score_threshold=item["score_threshold"])
    for item in DB_PATHS
]

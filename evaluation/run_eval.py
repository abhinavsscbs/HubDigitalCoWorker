import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_engine.answer import answer_with_refine_chain


def _load_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _contains_all_terms(text: str, terms):
    t = (text or "").lower()
    return all(term.lower() in t for term in terms)


def evaluate(dataset_path: Path):
    rows = _load_jsonl(dataset_path)
    results = []

    for row in rows:
        qid = row.get("id") or "unknown"
        question = row["question"]
        must_include = row.get("must_include", [])
        expected_min_sources = int(row.get("expected_min_sources", 1))

        response = answer_with_refine_chain(question)

        answer_text = (response.get("answer_text") or response.get("answer") or "").strip()
        sources = response.get("sources") or []
        confidence = response.get("confidence") or {"score": 0.0, "label": "low"}

        lexical_pass = _contains_all_terms(answer_text, must_include)
        source_count_pass = len(sources) >= expected_min_sources

        results.append(
            {
                "id": qid,
                "question": question,
                "lexical_pass": lexical_pass,
                "source_count_pass": source_count_pass,
                "source_count": len(sources),
                "confidence_score": float(confidence.get("score", 0.0)),
                "confidence_label": confidence.get("label", "low"),
            }
        )

    lexical_rate = mean([1 if x["lexical_pass"] else 0 for x in results]) if results else 0.0
    source_rate = mean([1 if x["source_count_pass"] else 0 for x in results]) if results else 0.0
    avg_confidence = mean([x["confidence_score"] for x in results]) if results else 0.0

    summary = {
        "dataset_size": len(results),
        "lexical_completeness_rate": round(lexical_rate, 3),
        "source_coverage_rate": round(source_rate, 3),
        "avg_confidence_score": round(avg_confidence, 3),
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run lightweight evaluation for IFRS RAG answers.")
    parser.add_argument(
        "--dataset",
        default="evaluation/sample_eval_set.jsonl",
        help="Path to JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/eval_report.json",
        help="Path for JSON report output.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    summary = evaluate(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"Detailed report written to: {output_path}")


if __name__ == "__main__":
    main()

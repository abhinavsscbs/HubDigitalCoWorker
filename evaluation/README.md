# Evaluation Harness (Accuracy & Validation)

This folder adds a lightweight baseline to address section **2.4 Accuracy & Validation**.

## Files
- `sample_eval_set.jsonl`: starter evaluation dataset format.
- `run_eval.py`: runs the RAG pipeline for each question and computes baseline metrics.

## Metrics produced
- `lexical_completeness_rate`: % of answers containing required key terms per sample.
- `source_coverage_rate`: % of answers meeting minimum source count target.
- `avg_confidence_score`: average model confidence score from pipeline output.

## Usage
```bash
python evaluation/run_eval.py \
  --dataset evaluation/sample_eval_set.jsonl \
  --output evaluation/eval_report.json
```

## Dataset schema (JSONL)
Each line is one object:
```json
{
  "id": "q1",
  "question": "Your question",
  "must_include": ["term1", "term2"],
  "expected_min_sources": 2
}
```

## Notes
- This is a baseline harness, not a substitute for a full expert-reviewed gold standard.
- For production, extend with human scoring for factual correctness and groundedness.

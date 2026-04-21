# HubDigitalCoWorker — Technical Handover & Clarification Response

Prepared for: IT / Deployment Team  
Scope date: 2026-04-18  
Repository scope reviewed: `/workspace/HubDigitalCoWorker`

---

## 1) General Understanding Required

### 1.1 Architecture Overview

```text
┌─────────────┐      HTTPS/JSON       ┌──────────────────────┐
│ Frontend UI │  ───────────────────► │ Flask API (backend)  │
│ (external)  │                       │ backend/app.py        │
└─────────────┘                       └──────────┬───────────┘
                                                 │
                                                 │ calls
                                                 ▼
                                      ┌─────────────────────────┐
                                      │ RAG Orchestration       │
                                      │ rag_engine/engine.py    │
                                      └──────────┬──────────────┘
                                                 │
                              ┌──────────────────┼───────────────────┐
                              ▼                  ▼                   ▼
                 ┌────────────────────┐  ┌───────────────┐  ┌─────────────────────┐
                 │ FAISS indexes      │  │ LLM endpoint  │  │ Session persistence │
                 │ IFRS A/B/C, EY,PwC │  │ HTTP API      │  │ Redis + JSON file   │
                 └────────────────────┘  └───────────────┘  └─────────────────────┘
```

**End-to-end data flow**
1. Client calls one of: `POST /api/ask`, `POST /api/followup`, or `POST /api/translate`.  
2. Flask inserts a `Thinking` prompt entry and schedules background work via `ThreadPoolExecutor`.  
3. Worker pipeline executes (RAG generation for ask/followup, translation pipeline for translate).  
4. Prompt entry is updated to terminal state (`Completed` or `Failed`).  
5. Client polls `POST /api/updatestatus` using `promptId` until terminal (`isTerminal=true`).  
6. Export endpoints (`/api/export/pdf`, `/api/export/html`, `/api/export/csv`) operate on stored history payloads.

**Major components**
- **Flask backend**: request handling, session/history management, async prompt lifecycle, export endpoints.  
- **RAG engine**: retrieval, LLM prompting, source filtering, formatting, exception generation.  
- **FAISS DBs**: five independent indexes configured in YAML (`IFRS A/B/C`, `EY`, `PwC`).  
- **LLM endpoint**: Cohere-style HTTP endpoint abstracted through `CohereEndpointChatModel`.

---

### 1.2 Code Structure Explanation

## Root-level modules
- `rag_config.py`: compatibility re-export to `rag_engine.config`.
- `llm_client.py`: compatibility re-export to `rag_engine.llm_client`.

## `backend/`
- `app.py`: Flask API app, endpoints, async prompt execution, request/response schema adaptation, polling status contract (`/api/updatestatus`), history/export/translation orchestration.
- `session_store.py`: Redis-first + filesystem fallback session persistence.
- `config.yaml`: runtime configuration (RAG paths, thresholds, LLM settings, server/cors/session).
- `requirements.txt`: Python dependency list.

## `rag_engine/`
- `engine.py`: core business logic (retrieval, prompts, generation, exception flow, text formatting, PDF/HTML export, translation helpers).
- `answer.py`: answer contract helpers (confidence classification/shape) and lazy orchestration entry points.
- `retrieval.py`: concrete retrieval module (DB config, FAISS load, similarity conversion, percentile filtering, chunk packaging).
- `formatting.py`: concrete formatting module (sanitization, citation cleanup, visible-answer formatting, metadata normalization).
- `translate.py`: translation entry point.
- `exports.py`: export entry points (PDF/HTML backend integration).
- `tables.py`: markdown table extraction/normalization/parsing utilities.
- `llm_client.py`: LLM HTTP client and model factory.
- `config.py`: YAML + env config loader and typed constants.
- `__init__.py`: broad export convenience.

## Embedding folders
- `IFRS_A_embed_test/`, `IFRS_B_embed_test/...`, `IFRS_C_embed_test/`, `EY_embed_test/`, `PwC_embed_test/`: FAISS index artifacts (`index.faiss`, `index.pkl`) consumed by LangChain FAISS loader.

### File dependency highlights
- `backend/app.py` imports from `rag_engine.answer`, `rag_engine.exports`, `rag_engine.formatting`, `rag_engine.tables`, `rag_engine.translate`, `rag_engine.engine`.
- `rag_engine.answer` owns confidence-result contract helpers and delegates orchestration entry points.
- `rag_engine.retrieval` owns FAISS retrieval and document packaging concerns.
- `rag_engine.engine` depends on `rag_config`/`llm_client` compatibility modules which point back into `rag_engine.config` and `rag_engine.llm_client`.

---

### 1.3 Function-Level Documentation (Current State + Gap)

- **Current state**: many functions in `engine.py` already include docstrings (e.g., retrieval, filtering, citation mapping, answer pipeline), but coverage is inconsistent and the file remains large (~2.5K lines after modular extraction).
- **`answer.py`** now contains confidence contract helpers and orchestration entry points, but orchestration internals still primarily live in `engine.py`.
- **Gap**: there is no dedicated generated API/function reference document, and not all edge cases are documented consistently in code comments.

**Recommended immediate action**
1. Split `engine.py` into modules (`retrieval.py`, `prompting.py`, `exceptions.py`, `formatting.py`, `exports.py`, `translation.py`).
2. Add a strict function docstring template (Purpose/Inputs/Outputs/Edge cases/Raises).
3. Generate reference docs via `pydoc-markdown` or Sphinx autodoc in CI.

### 1.4 API Execution Contract (Important for Frontend Integration)

All interactive endpoints now follow the same async contract:

1. Call endpoint (`/api/ask`, `/api/followup`, `/api/translate`).
2. Receive success payload with `promptId` and `promptStatus=Thinking`.
3. Poll `/api/updatestatus` with the same `promptId`.
4. Stop polling when `isTerminal=true` (`promptStatus` is `Completed` or `Failed`).

**Operational note**: frontend should not assume synchronous completion for follow-up or translation; both are background worker tasks now.

---

## 2) RAG & AI Logic Clarification

### 2.1 Retrieval Logic

Current pipeline in `answer_with_refine_chain`:
1. **Relevance gate** per DB via `get_query_relevance_llm(..., score_threshold=STAGE_1_THRESHOLD)`.
2. If any DB is relevant, fetch docs per DB using `fetch_docs`.
3. `fetch_docs` calls `retrieve_docs_with_score` in percentile mode by default (`top_percentile=STAGE_2_PERCENTILE`, usually 0.30).
4. FAISS L2 distance is converted to cosine-like similarity in code.
5. Retrieved docs are tagged with metadata: `_doc_id`, `_similarity_score`, `source_db`.
6. Second LLM pass (`_extract_allowed_doc_ids_with_llm`) selects final doc IDs.
7. `_filter_docs_by_ids` restricts answer context strictly to selected docs.

**Ranking/selection notes**
- Stage 2 is percentile-based (top X%) not fixed `k` in final use.
- There is still a per-index initial fetch (`max_k=50`) before percentile filtering.
- Display-only filter removes page `0` references (`filter_page_zero_references`) after generation.

### 2.2 Prompt Engineering

Key prompt assets in `engine.py`:
- `_EXTRACTOR_PROMPT`: selects relevant `DOC_ID`s from multi-source snippets.
- `stuff_prompt_normal`: main answer composition prompt with formatting rules and table constraints.
- Exception workflow prompts:
  - `exception_identification_prompt`
  - `exception_generation_prompt`
  - `exception_filtering_prompt`
- `_build_strip_prompt`: optional post-processing to remove inline citations.

**Hallucination control currently implemented**
- “Use only provided context” rule in `stuff_prompt_normal`.
- Multi-stage filtering: relevance gate + doc-id selector.
- Structured source tagging (`(IFRS A - ... - para X)`) for traceability.
- Final source list attached back to response payload.

### 2.3 LLM Integration

- Client class: `CohereEndpointChatModel` in `rag_engine/llm_client.py`.
- Transport: HTTP `POST` to `COHERE_ENDPOINT_URL` with optional bearer token.
- Payload fields: `message`, `temperature`, `max_tokens`, `response_format`, optional `model`.
- Model keys in config: `mini`, `full`, `extractor`, `relevance`.
- Response parser supports multiple possible response schemas (`response`, `text`, `output`, nested `message`, `generations`).

### 2.4 Accuracy & Validation

- Lightweight evaluation harness now added under `evaluation/` with:
  - `evaluation/sample_eval_set.jsonl` starter dataset
  - `evaluation/run_eval.py` report generator
  - output metrics: lexical completeness rate, source coverage rate, and average confidence score.
- Evaluation harness is **optional** (offline QA/regression activity) and is **not** required for normal API request processing.
- Confidence score is now returned from the RAG pipeline as `confidence` with:
  - `score` (0.0–1.0),
  - `label` (`low|medium|high`),
  - component breakdown (similarity/citation coverage/source diversity).
- Validation remains partially logic-based (relevance threshold + LLM filtering + citation mapping), and should still be extended with expert-reviewed benchmarks.

**Recommendation**: create a gold QA set (100-300 curated IFRS questions) and compute retrieval precision@k, groundedness checks, and answer completeness before release.

---

## 3) Configuration & Environment

### 3.1 Environment Setup (as observed)

1. Create Python environment (3.10+ recommended).
2. Install `backend/requirements.txt`.
3. Configure `backend/config.yaml` and/or env var `RAG_ENGINE_CONFIG`.
4. Ensure all FAISS index folders exist and paths resolve.
5. Start backend with `python backend/app.py`.

**Critical config keys**
- `llm.endpoint_url`, `llm.api_key`, `llm.timeout_sec`, `llm.verify_ssl`
- `rag.embedder.model_name`, `rag.embedder.device`
- `rag.thresholds.stage_1`, `rag.thresholds.stage_2_percentile`
- `rag.db_paths[]`
- `session.redis.url`, `session.file.dir`

### 3.2 Paths & Hardcoding

- Config sample includes `/opt/ifrs/...` absolute paths for model/index/data.
- This is likely deployment-target-specific and should be parameterized.

**Recommended externalization**
- Set `RAG_ENGINE_CONFIG` per environment (dev/stage/prod).
- Use `${ENV_VAR:default}` placeholders already supported by config loader.
- Avoid embedding infrastructure paths in source; keep only relative defaults.

---

## 4) Security Requirements

### 4.1 API Security — current state

- No authentication/authorization middleware detected.
- CORS is enabled globally (`CORS(app)`), effectively open unless tightened externally.
- Endpoints rely on user-provided IDs and are not cryptographically bound to caller identity.

### 4.2 Required Security Implementation (recommended)

Minimum production baseline:
1. Put API behind API Gateway / reverse proxy (Nginx/Kong/Envoy).
2. Require JWT/OAuth2 access token validation at gateway or Flask middleware.
3. Restrict CORS to known frontend origin(s) only.
4. Add mTLS or internal network-only access for backend-to-LLM calls.
5. Enforce rate limiting and request size limits.
6. Add structured audit logs for auth failures and sensitive operations.

---

## 5) Performance & Capacity

### 5.1 Hardware Requirements (estimated guidance)

Because no benchmark data is included, treat as initial sizing:
- **CPU**: 8–16 vCPU (retrieval + parsing + concurrent Flask workers).
- **RAM**: 16–32 GB (embedder model + FAISS loads + request concurrency).
- **Disk**: 20+ GB SSD minimum (indexes, models, logs, history).

Actual need depends on index size growth and LLM endpoint latency.

### 5.2 Performance Metrics

- No baked-in metrics collection for average response/retrieval/LLM latency.
- `time_taken_sec` is returned per response but not aggregated.

### 5.3 Throughput

- No load/stress test artifacts found.
- Current bottlenecks likely:
  - LLM endpoint response time / availability
  - Python single-process limitations if deployed as-is
  - memory pressure when loading multiple FAISS indexes

**Recommendation**: run k6/Locust with representative workloads before go-live.

---

## 6) Deployment & DevOps

### 6.1 Deployment Model

Recommended path:
- **Initial**: single-node deployment (API + local FAISS storage), external managed Redis, external LLM endpoint.
- **Scale-out**: stateless Flask workers behind load balancer + shared Redis + replicated index storage.

### 6.2 Containerization

- No Dockerfile / docker-compose found in repository.
- Containerization is **not yet implemented** in current codebase.

**Required deliverable suggestion**
- Multi-stage Dockerfile for backend.
- Optional compose stack with Redis + backend + reverse proxy.
- Runtime env-configured mounts for FAISS/model directories.

### 6.3 CI/CD

No pipeline manifests found.

Recommended pipeline:
1. Lint + type checks + unit tests
2. Build image + vulnerability scan
3. Integration smoke test (mock LLM)
4. Push to registry
5. Deploy with rolling strategy

Versioning: semantic versioning (`vMAJOR.MINOR.PATCH`) tied to tagged releases.

---

## 7) Logging & Monitoring

### 7.1 Logging (current)

- Uses print statements and stack traces (`traceback.print_exc()`), mostly unstructured.
- No correlation IDs, no structured JSON logging, no centralized logger config.

**Required upgrade**
- Adopt structured logger (`structlog` or stdlib JSON formatter).
- Include fields: `request_id`, `user_id`, `prompt_id`, `latency_ms`, `status_code`, `error_type`.

### 7.2 Monitoring

- No metrics endpoint or instrumentation detected.

**Expected stack**
- Prometheus metrics + Grafana dashboard or equivalent.
- Track request count, error rate, p50/p95/p99 latency, LLM timeout rate, Redis errors.

---

## 8) Testing

- `pytest` is listed in requirements, but no tests directory/files found in repo snapshot.
- No explicit integration test scripts were found.

**Recommendation**
- Add unit tests for retrieval filters, citation mapping, config loading.
- Add integration tests for `/api/ask`, `/api/updatestatus`, `/api/export/*` with mocked LLM.

---

## 9) Maintainability Concerns

Observation confirmed: `rag_engine/engine.py` currently carries too many responsibilities.

**Refactor target structure**
- `rag_engine/retrieval_core.py`
- `rag_engine/prompt_templates.py`
- `rag_engine/generation_pipeline.py`
- `rag_engine/exceptions_pipeline.py`
- `rag_engine/formatting_core.py`
- `rag_engine/exporters.py`

Outcome: lower cognitive load, easier testing, safer modifications.

---

## 10) Data & Embeddings

### 10.1 Embedding Data

- Repo contains prebuilt FAISS indexes for IFRS/EY/PwC sources.
- Source document ingestion scripts are not included in this repository snapshot.

### 10.2 Rebuild Process

- No embedding rebuild pipeline/script was found.

**Required action**
- Deliver separate ingestion/indexing pipeline (chunking, metadata schema, embedding model lock, FAISS save process).
- Version indexes and metadata for deterministic rollbacks.

---

## 11) Risks & Dependencies

### External dependencies
- On-prem or remote LLM service availability and schema compatibility.
- Python packages (`langchain*`, `faiss-cpu`, `sentence-transformers`, `torch`, etc.).
- Redis availability for high-performance session handling.

### Failure scenarios
- **LLM down/timeout**: answer pipeline fails, prompts marked `Failed`.
- **FAISS index missing/corrupt**: load fails with runtime error.
- **Redis unavailable**: fallback to file history (reduced robustness/performance).

---

## 12) Upgrade & Improvement Plan

### Short Term (0–4 weeks)
- Implement authn/authz + restrict CORS.
- Replace print logging with structured logging.
- Clean config defaults and remove environment-specific absolute paths from source defaults.

### Mid Term (1–2 months)
- Modularize `engine.py` into focused modules.
- Add test suite + CI pipeline.
- Add latency/error monitoring and dashboards.
- Run baseline load tests and tune thresholds/concurrency.

### Long Term (2–4 months)
- Horizontal scaling architecture and caching strategy.
- Formal evaluation framework (gold set, groundedness, drift monitoring).
- Knowledge-base lifecycle automation (ingest → validate → deploy).

---

## 13) Final Notes

- The system has a workable RAG foundation and practical multi-source retrieval strategy.
- Production hardening is required before public exposure:
  - security controls,
  - observability,
  - modularization,
  - deployment standardization (containers + CI/CD).
- The highest immediate priorities are **security + logging + operational controls**.

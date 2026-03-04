# Prompt Services (Flask)

Production-style Flask implementations for:
- `POST /api/ask`
- `POST /api/followup`
- `POST /api/updatestatus`

Each service is an independent Flask app and only exposes its required endpoint plus `GET /health`.

## Non-functional notes
- Availability: document-level requirement only; no infrastructure included here.
- Request timeout: hard timeout guard at `60s` (configurable via `REQUEST_TIMEOUT_SECONDS`). Timeout returns `504` JSON.
- Rate limiting: in-memory and thread-safe, **single-instance only** (not distributed):
  - per-user: `10/min` by `userId` in request body
  - global: `5 req/s`
- Logging: structured JSON logs containing required fields.
- Ask/followup answer generation defaults to the repo's existing RAG capability (`rag_engine.answer.answer_with_refine_chain`), with deterministic stub fallback if unavailable.
- Prompt records are stored in shared local JSON files under `backend/prompt_services/data/prompt_history` so ask/followup/updatestatus can resolve the same prompt IDs across processes.
- If repo RAG dependencies/config are not loadable at runtime, APIs remain available via stub fallback (no crash).

## Environment variables
- `BASIC_AUTH_USER` (default `admin`)
- `BASIC_AUTH_PASS` (default `changeme`)
- `REQUEST_TIMEOUT_SECONDS` (default `60`)
- `PER_USER_RATE_LIMIT` (default `10`)
- `PER_USER_RATE_WINDOW_SECONDS` (default `60`)
- `GLOBAL_RATE_LIMIT` (default `5`)
- `GLOBAL_RATE_WINDOW_SECONDS` (default `1`)
- `MAX_HISTORY_PER_CONTEXT` (default `20`)
- `DEV_AUTO_COMPLETE` (default `true`) for status service
- `DEV_AUTO_COMPLETE_AFTER_SECONDS` (default `5`) for status service

Optional local demo seed vars:
- Follow-up service success demo:
  - `FOLLOWUP_DEMO_USER_ID`
  - `FOLLOWUP_DEMO_PROMPT_ID`
  - `FOLLOWUP_DEMO_PROMPT_TEXT`
- Status service success demo:
  - `STATUS_DEMO_USER_ID`
  - `STATUS_DEMO_PROMPT_ID`

## Run locally

1. `cd backend/prompt_services`
2. `python3 -m venv .venv`
3. `source .venv/bin/activate`
4. `pip install -r requirements.txt`
5. Set auth env vars:
   - `export BASIC_AUTH_USER=apiuser`
   - `export BASIC_AUTH_PASS=apipass`

Run each service (separately):
- Ask service: `python -m ask_service.app` (port `8001`)
- Follow-up service: `python -m followup_service.app` (port `8002`)
- Status service: `python -m status_service.app` (port `8003`)

## curl examples

### Ask service
Success:
```bash
curl -u apiuser:apipass -X POST http://localhost:8001/api/ask \
  -H "Content-Type: application/json" \
  -d '{"userId":"E101","promptRequestText":"Create a tabular summary for IFRS revenue"}'
```

401:
```bash
curl -u wrong:creds -X POST http://localhost:8001/api/ask \
  -H "Content-Type: application/json" \
  -d '{"userId":"E101","promptRequestText":"hello"}'
```

429 (send >10/min for same user):
```bash
for i in {1..11}; do
  curl -s -o /dev/null -w "%{http_code}\n" -u apiuser:apipass -X POST http://localhost:8001/api/ask \
    -H "Content-Type: application/json" \
    -d '{"userId":"E101","promptRequestText":"hello"}'
done
```

### Follow-up service
Success (use seeded promptId):
```bash
export FOLLOWUP_DEMO_USER_ID=E201
export FOLLOWUP_DEMO_PROMPT_ID=IFRS-20260302-AAAABBBB
python -m followup_service.app
```
Then:
```bash
curl -u apiuser:apipass -X POST http://localhost:8002/api/followup \
  -H "Content-Type: application/json" \
  -d '{"userId":"E201","promptId":"IFRS-20260302-AAAABBBB","promptRequestText":"Show this in table format"}'
```

401:
```bash
curl -u wrong:creds -X POST http://localhost:8002/api/followup \
  -H "Content-Type: application/json" \
  -d '{"userId":"E201","promptId":"IFRS-20260302-AAAABBBB","promptRequestText":"hello"}'
```

404 unknown promptId:
```bash
curl -u apiuser:apipass -X POST http://localhost:8002/api/followup \
  -H "Content-Type: application/json" \
  -d '{"userId":"E201","promptId":"IFRS-20260302-UNKNOWN00","promptRequestText":"hello"}'
```

429:
```bash
for i in {1..11}; do
  curl -s -o /dev/null -w "%{http_code}\n" -u apiuser:apipass -X POST http://localhost:8002/api/followup \
    -H "Content-Type: application/json" \
    -d '{"userId":"E201","promptId":"IFRS-20260302-AAAABBBB","promptRequestText":"hello"}'
done
```

### Status service
Success (use seeded promptId):
```bash
export STATUS_DEMO_USER_ID=E301
export STATUS_DEMO_PROMPT_ID=IFRS-20260302-CCCCDDDD
python -m status_service.app
```
Then:
```bash
curl -u apiuser:apipass -X POST http://localhost:8003/api/updatestatus \
  -H "Content-Type: application/json" \
  -d '{"userId":"E301","promptId":"IFRS-20260302-CCCCDDDD"}'
```

401:
```bash
curl -u wrong:creds -X POST http://localhost:8003/api/updatestatus \
  -H "Content-Type: application/json" \
  -d '{"userId":"E301","promptId":"IFRS-20260302-CCCCDDDD"}'
```

404 unknown promptId:
```bash
curl -u apiuser:apipass -X POST http://localhost:8003/api/updatestatus \
  -H "Content-Type: application/json" \
  -d '{"userId":"E301","promptId":"IFRS-20260302-UNKNOWN00"}'
```

429:
```bash
for i in {1..11}; do
  curl -s -o /dev/null -w "%{http_code}\n" -u apiuser:apipass -X POST http://localhost:8003/api/updatestatus \
    -H "Content-Type: application/json" \
    -d '{"userId":"E301","promptId":"IFRS-20260302-CCCCDDDD"}'
done
```

## Run tests

```bash
cd backend/prompt_services
pytest -q
```

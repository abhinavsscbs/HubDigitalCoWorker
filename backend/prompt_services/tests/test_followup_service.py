from datetime import datetime, timezone

from followup_service.app import create_app
from common.prompt_history import InMemoryPromptHistoryStore
from tests.conftest import basic_auth_headers


KNOWN_USER = "E200"
KNOWN_PROMPT_ID = "IFRS-20260101-ABCDEFGH"


def fake_answer_engine(prompt_text: str) -> dict:
    return {
        "promptResponseText": f"Repo-backed followup test answer for: {prompt_text}",
        "promptResponseTabularData": {"headers": ["col1", "col2"], "rows": [["x", "9"]]},
        "engine": "test",
    }


def build_test_app():
    prompt_store = InMemoryPromptHistoryStore()
    app = create_app(
        {
            "TESTING": True,
            "BASIC_AUTH_USER": "test-user",
            "BASIC_AUTH_PASS": "test-pass",
            "GLOBAL_RATE_LIMIT": 1000,
            "GLOBAL_RATE_WINDOW_SECONDS": 1,
            "PROMPT_HISTORY_STORE": prompt_store,
            "ANSWER_ENGINE_FUNC": fake_answer_engine,
        }
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    prompt_store.seed_prompt(
        KNOWN_USER,
        {
            "recordType": "prompt_service",
            "promptId": KNOWN_PROMPT_ID,
            "originalPromptId": KNOWN_PROMPT_ID,
            "referencedPromptId": None,
            "promptStatus": "Completed",
            "promptTitle": "Original seeded prompt",
            "promptRequestText": "Original seeded prompt",
            "promptResponseText": "Original seeded answer",
            "promptResponseTabularData": {"headers": [], "rows": []},
            "created_at": now_iso,
            "updated_at": now_iso,
        },
    )
    return app


def test_happy_path_with_seeded_prompt_context():
    app = build_test_app()
    client = app.test_client()

    response = client.post(
        "/api/followup",
        json={
            "userId": KNOWN_USER,
            "promptId": KNOWN_PROMPT_ID,
            "promptRequestText": "Can you provide this as tabular output?",
        },
        headers=basic_auth_headers("test-user", "test-pass"),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["promptId"].startswith("IFRS-")
    assert payload["promptId"] != KNOWN_PROMPT_ID
    assert f"Follow-up reference: {KNOWN_PROMPT_ID}" in payload["promptResponseText"]
    assert payload["promptResponseTabularData"]["headers"] == ["col1", "col2"]


def test_missing_fields_returns_400():
    app = build_test_app()
    client = app.test_client()

    response = client.post(
        "/api/followup",
        json={"userId": KNOWN_USER, "promptId": KNOWN_PROMPT_ID},
        headers=basic_auth_headers("test-user", "test-pass"),
    )

    assert response.status_code == 400


def test_bad_auth_returns_401():
    app = build_test_app()
    client = app.test_client()

    response = client.post(
        "/api/followup",
        json={
            "userId": KNOWN_USER,
            "promptId": KNOWN_PROMPT_ID,
            "promptRequestText": "follow up",
        },
        headers=basic_auth_headers("bad", "auth"),
    )

    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


def test_unknown_prompt_id_returns_404():
    app = build_test_app()
    client = app.test_client()

    response = client.post(
        "/api/followup",
        json={
            "userId": KNOWN_USER,
            "promptId": "IFRS-20260101-UNKNOWN0",
            "promptRequestText": "follow up",
        },
        headers=basic_auth_headers("test-user", "test-pass"),
    )

    assert response.status_code == 404
    assert response.get_json() == {"error": "Unknown promptId for userId"}


def test_per_user_rate_limit_returns_429():
    app = build_test_app()
    client = app.test_client()
    headers = basic_auth_headers("test-user", "test-pass")

    payload = {
        "userId": KNOWN_USER,
        "promptId": KNOWN_PROMPT_ID,
        "promptRequestText": "follow up",
    }

    for _ in range(10):
        response = client.post("/api/followup", json=payload, headers=headers)
        assert response.status_code == 200

    response = client.post("/api/followup", json=payload, headers=headers)
    assert response.status_code == 429

from datetime import datetime, timedelta, timezone

from status_service.app import create_app
from common.prompt_history import InMemoryPromptHistoryStore
from tests.conftest import basic_auth_headers


KNOWN_USER = "E300"
COMPLETED_PROMPT_ID = "IFRS-20260101-COMPLETE"
THINKING_PROMPT_ID = "IFRS-20260101-THINKING"


def build_test_app(auto_complete=False, auto_complete_after_seconds=5):
    prompt_store = InMemoryPromptHistoryStore()
    app = create_app(
        {
            "TESTING": True,
            "BASIC_AUTH_USER": "test-user",
            "BASIC_AUTH_PASS": "test-pass",
            "GLOBAL_RATE_LIMIT": 1000,
            "GLOBAL_RATE_WINDOW_SECONDS": 1,
            "DEV_AUTO_COMPLETE": auto_complete,
            "DEV_AUTO_COMPLETE_AFTER_SECONDS": auto_complete_after_seconds,
            "PROMPT_HISTORY_STORE": prompt_store,
        }
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    prompt_store.seed_prompt(
        KNOWN_USER,
        {
            "recordType": "prompt_service",
            "promptId": COMPLETED_PROMPT_ID,
            "originalPromptId": COMPLETED_PROMPT_ID,
            "referencedPromptId": None,
            "promptStatus": "Completed",
            "promptTitle": "Completed title",
            "promptRequestText": "Completed question",
            "promptResponseText": "Completed response text",
            "promptResponseTabularData": {"headers": [], "rows": []},
            "created_at": now_iso,
            "updated_at": now_iso,
        },
    )
    thinking_created = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    prompt_store.seed_prompt(
        KNOWN_USER,
        {
            "recordType": "prompt_service",
            "promptId": THINKING_PROMPT_ID,
            "originalPromptId": THINKING_PROMPT_ID,
            "referencedPromptId": None,
            "promptStatus": "Thinking",
            "promptTitle": "Thinking title",
            "promptRequestText": "Thinking question",
            "promptResponseText": "",
            "promptResponseTabularData": {"headers": [], "rows": []},
            "created_at": thinking_created,
            "updated_at": thinking_created,
        },
    )
    return app


def test_happy_path_completed():
    app = build_test_app(auto_complete=False)
    client = app.test_client()

    response = client.post(
        "/api/updatestatus",
        json={"userId": KNOWN_USER, "promptId": COMPLETED_PROMPT_ID},
        headers=basic_auth_headers("test-user", "test-pass"),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["promptId"] == COMPLETED_PROMPT_ID
    assert payload["promptStatus"] == "Completed"
    assert payload["promptResponseText"] == "Completed response text"


def test_happy_path_thinking_auto_complete_enabled():
    app = build_test_app(auto_complete=True, auto_complete_after_seconds=0)
    client = app.test_client()

    response = client.post(
        "/api/updatestatus",
        json={"userId": KNOWN_USER, "promptId": THINKING_PROMPT_ID},
        headers=basic_auth_headers("test-user", "test-pass"),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["promptId"] == THINKING_PROMPT_ID
    assert payload["promptStatus"] == "Completed"
    assert payload["promptResponseText"]


def test_missing_fields_returns_400():
    app = build_test_app()
    client = app.test_client()

    response = client.post(
        "/api/updatestatus",
        json={"userId": KNOWN_USER},
        headers=basic_auth_headers("test-user", "test-pass"),
    )

    assert response.status_code == 400


def test_bad_auth_returns_401():
    app = build_test_app()
    client = app.test_client()

    response = client.post(
        "/api/updatestatus",
        json={"userId": KNOWN_USER, "promptId": COMPLETED_PROMPT_ID},
        headers=basic_auth_headers("bad", "auth"),
    )

    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


def test_unknown_prompt_id_returns_404():
    app = build_test_app()
    client = app.test_client()

    response = client.post(
        "/api/updatestatus",
        json={"userId": KNOWN_USER, "promptId": "IFRS-20260101-UNKNOWN0"},
        headers=basic_auth_headers("test-user", "test-pass"),
    )

    assert response.status_code == 404
    assert response.get_json() == {"error": "Unknown promptId for userId"}


def test_per_user_rate_limit_returns_429():
    app = build_test_app()
    client = app.test_client()
    headers = basic_auth_headers("test-user", "test-pass")

    payload = {"userId": KNOWN_USER, "promptId": COMPLETED_PROMPT_ID}

    for _ in range(10):
        response = client.post("/api/updatestatus", json=payload, headers=headers)
        assert response.status_code == 200

    response = client.post("/api/updatestatus", json=payload, headers=headers)
    assert response.status_code == 429

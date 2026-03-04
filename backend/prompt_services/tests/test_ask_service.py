from ask_service.app import create_app
from common.prompt_history import InMemoryPromptHistoryStore
from tests.conftest import basic_auth_headers


def fake_answer_engine(prompt_text: str) -> dict:
    return {
        "promptResponseText": f"Repo-backed test answer for: {prompt_text}",
        "promptResponseTabularData": {"headers": ["col1", "col2"], "rows": [["a", "1"]]},
        "engine": "test",
    }


def build_test_app():
    return create_app(
        {
            "TESTING": True,
            "BASIC_AUTH_USER": "test-user",
            "BASIC_AUTH_PASS": "test-pass",
            "GLOBAL_RATE_LIMIT": 1000,
            "GLOBAL_RATE_WINDOW_SECONDS": 1,
            "PROMPT_HISTORY_STORE": InMemoryPromptHistoryStore(),
            "ANSWER_ENGINE_FUNC": fake_answer_engine,
        }
    )


def test_happy_path():
    app = build_test_app()
    client = app.test_client()

    response = client.post(
        "/api/ask",
        json={
            "userId": "E100",
            "promptRequestText": "Please provide a table of key IFRS metrics",
        },
        headers=basic_auth_headers("test-user", "test-pass"),
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["promptId"].startswith("IFRS-")
    assert payload["promptStatus"] in {"Completed", "Thinking"}
    assert payload["promptTitle"]
    assert "repo-backed test answer" in payload["promptResponseText"].lower()
    assert payload["promptResponseTabularData"]["headers"] == ["col1", "col2"]


def test_missing_fields_returns_400():
    app = build_test_app()
    client = app.test_client()

    response = client.post(
        "/api/ask",
        json={"userId": "E100"},
        headers=basic_auth_headers("test-user", "test-pass"),
    )

    assert response.status_code == 400


def test_bad_auth_returns_401():
    app = build_test_app()
    client = app.test_client()

    response = client.post(
        "/api/ask",
        json={"userId": "E100", "promptRequestText": "hello"},
        headers=basic_auth_headers("wrong", "creds"),
    )

    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


def test_per_user_rate_limit_returns_429():
    app = build_test_app()
    client = app.test_client()

    headers = basic_auth_headers("test-user", "test-pass")
    payload = {"userId": "E100", "promptRequestText": "hello"}

    for _ in range(10):
        response = client.post("/api/ask", json=payload, headers=headers)
        assert response.status_code == 200

    response = client.post("/api/ask", json=payload, headers=headers)
    assert response.status_code == 429

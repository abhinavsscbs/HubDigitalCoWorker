import os
import time
from datetime import datetime, timezone

from flask import Flask, jsonify

from common.answer_engine import generate_answer
from common.auth import require_basic_auth
from common.config import load_common_config
from common.errors import ApiError, ValidationError
from common.json_logging import get_json_logger
from common.prompt_history import JsonFilePromptHistoryStore
from common.prompt_utils import derive_prompt_title, generate_prompt_id
from common.rate_limiter import InMemoryRateLimiter
from common.timeout import TimeoutExceeded, run_with_timeout
from common.validation import parse_json_payload, require_non_empty_string_fields


def _seed_demo_context_from_env(store):
    user_id = os.getenv("FOLLOWUP_DEMO_USER_ID")
    prompt_id = os.getenv("FOLLOWUP_DEMO_PROMPT_ID")
    prompt_text = os.getenv("FOLLOWUP_DEMO_PROMPT_TEXT", "Sample original question")
    if user_id and prompt_id:
        now_iso = datetime.now(timezone.utc).isoformat()
        store.seed_prompt(
            user_id,
            {
                "recordType": "prompt_service",
                "promptId": prompt_id,
                "originalPromptId": prompt_id,
                "referencedPromptId": None,
                "promptStatus": "Completed",
                "promptTitle": derive_prompt_title(prompt_text),
                "promptRequestText": prompt_text,
                "promptResponseText": "Seeded answer",
                "promptResponseTabularData": {"headers": [], "rows": []},
                "created_at": now_iso,
                "updated_at": now_iso,
            },
        )


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(load_common_config())
    if test_config:
        app.config.update(test_config)

    app.config["RATE_LIMITER"] = InMemoryRateLimiter(
        per_user_limit=app.config["PER_USER_RATE_LIMIT"],
        per_user_window_seconds=app.config["PER_USER_RATE_WINDOW_SECONDS"],
        global_limit=app.config["GLOBAL_RATE_LIMIT"],
        global_window_seconds=app.config["GLOBAL_RATE_WINDOW_SECONDS"],
    )
    app.config["PROMPT_HISTORY_STORE"] = app.config.get(
        "PROMPT_HISTORY_STORE",
        JsonFilePromptHistoryStore(max_entries_per_user=app.config["MAX_HISTORY_PER_CONTEXT"]),
    )
    _seed_demo_context_from_env(app.config["PROMPT_HISTORY_STORE"])
    app.config["ANSWER_ENGINE_FUNC"] = app.config.get("ANSWER_ENGINE_FUNC", generate_answer)
    app.config["LOGGER"] = get_json_logger("followup-service")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"}), 200

    @app.route("/api/followup", methods=["POST"])
    @require_basic_auth("Follow-up Service")
    def followup():
        started = time.perf_counter()
        status_code = 500
        user_id = None
        referenced_prompt_id = None
        prompt_id = None
        logger = app.config["LOGGER"]

        try:
            payload = parse_json_payload()
            data = require_non_empty_string_fields(
                payload,
                ["userId", "promptId", "promptRequestText"],
            )
            user_id = data["userId"]
            referenced_prompt_id = data["promptId"]
            followup_text = data["promptRequestText"]

            allowed, _ = app.config["RATE_LIMITER"].allow(user_id)
            if not allowed:
                raise ApiError(429, "Rate limit exceeded")

            prompt_store = app.config["PROMPT_HISTORY_STORE"]
            referenced_record = prompt_store.get_prompt_record(user_id, referenced_prompt_id)
            if not referenced_record:
                raise ApiError(404, "Unknown promptId for userId")

            def answer_engine():
                nonlocal prompt_id
                prompt_id = generate_prompt_id()
                root_prompt_id = referenced_record.get("originalPromptId") or referenced_prompt_id
                latest_thread_record = prompt_store.get_latest_thread_record(user_id, root_prompt_id)
                previous_question = (
                    latest_thread_record.get("promptRequestText")
                    if latest_thread_record
                    else referenced_record.get("promptRequestText")
                )
                context_summary = (
                    f" Context summary: last related question was '{previous_question}'."
                    if previous_question
                    else ""
                )
                combined_prompt = (
                    f"Original context: {previous_question}\nFollow-up question: {followup_text}"
                    if previous_question
                    else followup_text
                )
                engine_output = app.config["ANSWER_ENGINE_FUNC"](combined_prompt)
                response_text = engine_output["promptResponseText"]

                response_text = (
                    f"{response_text}\n\n"
                    f"Follow-up reference: {referenced_prompt_id}. "
                    "Response derived from referenced source/query context."
                    f"{context_summary}"
                )
                now_iso = datetime.now(timezone.utc).isoformat()
                record = {
                    "recordType": "prompt_service",
                    "promptId": prompt_id,
                    "originalPromptId": root_prompt_id,
                    "referencedPromptId": referenced_prompt_id,
                    "promptStatus": "Completed",
                    "promptTitle": derive_prompt_title(followup_text),
                    "promptRequestText": followup_text,
                    "promptResponseText": response_text,
                    "promptResponseTabularData": engine_output["promptResponseTabularData"],
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
                prompt_store.add_prompt_record(user_id, record)

                return {
                    "promptId": prompt_id,
                    "promptStatus": record["promptStatus"],
                    "promptTitle": record["promptTitle"],
                    "promptResponseText": record["promptResponseText"],
                    "promptResponseTabularData": record["promptResponseTabularData"],
                }

            response_payload = run_with_timeout(
                answer_engine,
                timeout_seconds=app.config["REQUEST_TIMEOUT_SECONDS"],
            )
            status_code = 200
            return jsonify(response_payload), 200

        except ValidationError as exc:
            status_code = 400
            return jsonify({"error": exc.message}), 400
        except TimeoutExceeded:
            status_code = 504
            return jsonify({"error": "Request timed out"}), 504
        except ApiError as exc:
            status_code = exc.status_code
            return jsonify({"error": exc.message}), exc.status_code
        except Exception:
            status_code = 500
            logger.exception("Unhandled error in /api/followup")
            return jsonify({"error": "Internal server error"}), 500
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "request_complete",
                extra={
                    "promptId": prompt_id,
                    "referencedPromptId": referenced_prompt_id,
                    "userId": user_id,
                    "latency_ms": latency_ms,
                    "status_code": status_code,
                },
            )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8002")))

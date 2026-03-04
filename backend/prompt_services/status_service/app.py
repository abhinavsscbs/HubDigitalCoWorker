import os
import time
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify

from common.auth import require_basic_auth
from common.config import load_common_config
from common.errors import ApiError, ValidationError
from common.json_logging import get_json_logger
from common.prompt_history import JsonFilePromptHistoryStore
from common.prompt_utils import derive_prompt_title
from common.rate_limiter import InMemoryRateLimiter
from common.timeout import TimeoutExceeded, run_with_timeout
from common.validation import parse_json_payload, require_non_empty_string_fields


def _seed_demo_data(store):
    demo_user = os.getenv("STATUS_DEMO_USER_ID")
    demo_prompt_id = os.getenv("STATUS_DEMO_PROMPT_ID")
    if not demo_user or not demo_prompt_id:
        return

    created_at = datetime.now(timezone.utc) - timedelta(seconds=10)
    store.seed_prompt(
        demo_user,
        {
            "recordType": "prompt_service",
            "promptId": demo_prompt_id,
            "originalPromptId": demo_prompt_id,
            "referencedPromptId": None,
            "promptStatus": "Thinking",
            "promptTitle": derive_prompt_title("Demo seeded title"),
            "promptRequestText": "Demo seeded question",
            "promptResponseText": "",
            "promptResponseTabularData": {"headers": [], "rows": []},
            "created_at": created_at.isoformat(),
            "updated_at": created_at.isoformat(),
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
    _seed_demo_data(app.config["PROMPT_HISTORY_STORE"])
    app.config["LOGGER"] = get_json_logger("status-service")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"}), 200

    @app.route("/api/updatestatus", methods=["POST"])
    @require_basic_auth("Check Previous Prompt Status Service")
    def updatestatus():
        started = time.perf_counter()
        status_code = 500
        user_id = None
        prompt_id = None
        returned_status = None
        logger = app.config["LOGGER"]

        try:
            payload = parse_json_payload()
            data = require_non_empty_string_fields(payload, ["userId", "promptId"])
            user_id = data["userId"]
            prompt_id = data["promptId"]

            allowed, _ = app.config["RATE_LIMITER"].allow(user_id)
            if not allowed:
                raise ApiError(429, "Rate limit exceeded")

            prompt_store = app.config["PROMPT_HISTORY_STORE"]

            def status_lookup():
                record = prompt_store.get_prompt_record(user_id, prompt_id)
                if not record:
                    raise ApiError(404, "Unknown promptId for userId")

                if (
                    app.config["DEV_AUTO_COMPLETE"]
                    and record["promptStatus"] == "Thinking"
                ):
                    created_at_raw = record.get("created_at")
                    created_at = datetime.now(timezone.utc)
                    if isinstance(created_at_raw, str):
                        try:
                            created_at = datetime.fromisoformat(created_at_raw)
                        except Exception:
                            created_at = datetime.now(timezone.utc)
                    elapsed = (
                        datetime.now(timezone.utc) - created_at
                    ).total_seconds()
                    if elapsed >= app.config["DEV_AUTO_COMPLETE_AFTER_SECONDS"]:
                        completed_response = record["promptResponseText"] or (
                            f"Auto-completed stub response for promptId {prompt_id}."
                        )
                        prompt_store.update_prompt_record(
                            user_id,
                            prompt_id,
                            {
                                "promptStatus": "Completed",
                                "promptResponseText": completed_response,
                            },
                        )
                        record = prompt_store.get_prompt_record(user_id, prompt_id)

                return {
                    "promptId": prompt_id,
                    "promptStatus": record["promptStatus"],
                    "promptTitle": record["promptTitle"],
                    "promptResponseText": record["promptResponseText"],
                    "promptResponseTabularData": record["promptResponseTabularData"],
                }

            response_payload = run_with_timeout(
                status_lookup,
                timeout_seconds=app.config["REQUEST_TIMEOUT_SECONDS"],
            )
            returned_status = response_payload["promptStatus"]
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
            logger.exception("Unhandled error in /api/updatestatus")
            return jsonify({"error": "Internal server error"}), 500
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "request_complete",
                extra={
                    "userId": user_id,
                    "promptId": prompt_id,
                    "promptStatus": returned_status,
                    "latency_ms": latency_ms,
                    "status_code": status_code,
                },
            )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8003")))

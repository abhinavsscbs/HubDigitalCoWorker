"""
Minimal API smoke test for async prompt endpoints.

Behavior:
- Calls /api/ask, /api/followup, and /api/translate.
- Polls /api/updatestatus until terminal status (Completed/Failed).
- Intentionally does NOT enforce max poll count or timeout.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict

import requests


TERMINAL_STATUSES = {"Completed", "Failed"}


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def _extract_prompt_payload(resp: Dict[str, Any]) -> Dict[str, Any]:
    payload = resp.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"Missing payload in response: {json.dumps(resp)}")
    return payload


def _poll_until_terminal(base_url: str, user_id: str, prompt_id: str, poll_interval_sec: float) -> Dict[str, Any]:
    status_url = f"{base_url}/api/updatestatus"
    while True:
        status_resp = _post_json(status_url, {"userId": user_id, "promptId": prompt_id})
        prompt_payload = _extract_prompt_payload(status_resp)
        status = prompt_payload.get("promptStatus")
        print(f"[poll] promptId={prompt_id} status={status}")
        if status in TERMINAL_STATUSES:
            return status_resp
        time.sleep(poll_interval_sec)


def _start_and_poll(base_url: str, endpoint: str, body: Dict[str, Any], poll_interval_sec: float) -> Dict[str, Any]:
    start_url = f"{base_url}{endpoint}"
    start_resp = _post_json(start_url, body)
    start_payload = _extract_prompt_payload(start_resp)
    prompt_id = start_payload.get("promptId")
    if not prompt_id:
        raise ValueError(f"{endpoint} did not return promptId: {json.dumps(start_resp)}")
    print(f"[start] endpoint={endpoint} promptId={prompt_id}")
    return _poll_until_terminal(base_url, body["userId"], prompt_id, poll_interval_sec)


def run_smoke(base_url: str, user_id: str, ask_q: str, followup_q: str, poll_interval_sec: float) -> None:
    ask_final = _start_and_poll(
        base_url,
        "/api/ask",
        {"userId": user_id, "question": ask_q},
        poll_interval_sec,
    )
    ask_payload = _extract_prompt_payload(ask_final)
    ask_prompt_id = ask_payload.get("promptId")
    print(f"[done] ask promptId={ask_prompt_id} status={ask_payload.get('promptStatus')}")

    followup_final = _start_and_poll(
        base_url,
        "/api/followup",
        {"userId": user_id, "question": followup_q, "promptId": ask_prompt_id},
        poll_interval_sec,
    )
    followup_payload = _extract_prompt_payload(followup_final)
    print(
        "[done] followup "
        f"promptId={followup_payload.get('promptId')} "
        f"status={followup_payload.get('promptStatus')}"
    )

    translate_final = _start_and_poll(
        base_url,
        "/api/translate",
        {"userId": user_id},
        poll_interval_sec,
    )
    translate_payload = _extract_prompt_payload(translate_final)
    print(
        "[done] translate "
        f"promptId={translate_payload.get('promptId')} "
        f"status={translate_payload.get('promptStatus')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Async API smoke test without poll timeout.")
    parser.add_argument("--base-url", default="http://localhost:3000", help="Backend base URL")
    parser.add_argument("--user-id", default="smoke-test-user", help="Prompt-service userId")
    parser.add_argument("--ask", default="What is IFRS 9?", help="Ask question")
    parser.add_argument("--followup", default="Summarize in 3 bullets.", help="Follow-up question")
    parser.add_argument("--poll-interval-sec", type=float, default=1.0, help="Status polling interval")
    args = parser.parse_args()

    run_smoke(args.base_url, args.user_id, args.ask, args.followup, args.poll_interval_sec)


if __name__ == "__main__":
    main()

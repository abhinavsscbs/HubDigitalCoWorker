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
from pathlib import Path
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _poll_until_terminal(
    base_url: str,
    user_id: str,
    prompt_id: str,
    poll_interval_sec: float,
    trace_file: Path,
) -> Dict[str, Any]:
    status_url = f"{base_url}/api/updatestatus"
    poll_history = []
    while True:
        status_resp = _post_json(status_url, {"userId": user_id, "promptId": prompt_id})
        prompt_payload = _extract_prompt_payload(status_resp)
        status = prompt_payload.get("promptStatus")
        poll_history.append(status_resp)
        _write_json(trace_file, {"promptId": prompt_id, "pollHistory": poll_history})
        if status in TERMINAL_STATUSES:
            return status_resp
        time.sleep(poll_interval_sec)


def _start_and_poll(
    base_url: str,
    endpoint: str,
    body: Dict[str, Any],
    poll_interval_sec: float,
    results_dir: Path,
    step_name: str,
) -> Dict[str, Any]:
    start_url = f"{base_url}{endpoint}"
    start_resp = _post_json(start_url, body)
    _write_json(results_dir / f"{step_name}_start.json", start_resp)
    start_payload = _extract_prompt_payload(start_resp)
    prompt_id = start_payload.get("promptId")
    if not prompt_id:
        raise ValueError(f"{endpoint} did not return promptId: {json.dumps(start_resp)}")
    final_resp = _poll_until_terminal(
        base_url,
        body["userId"],
        prompt_id,
        poll_interval_sec,
        results_dir / f"{step_name}_poll_history.json",
    )
    _write_json(results_dir / f"{step_name}_final.json", final_resp)
    return final_resp


def run_smoke(
    base_url: str,
    user_id: str,
    ask_q: str,
    followup_q: str,
    poll_interval_sec: float,
    results_dir: Path,
) -> None:
    ask_final = _start_and_poll(
        base_url,
        "/api/ask",
        {"userId": user_id, "question": ask_q},
        poll_interval_sec,
        results_dir,
        "ask",
    )
    ask_payload = _extract_prompt_payload(ask_final)
    ask_prompt_id = ask_payload.get("promptId")

    followup_final = _start_and_poll(
        base_url,
        "/api/followup",
        {"userId": user_id, "question": followup_q, "promptId": ask_prompt_id},
        poll_interval_sec,
        results_dir,
        "followup",
    )
    _extract_prompt_payload(followup_final)

    _start_and_poll(
        base_url,
        "/api/translate",
        {"userId": user_id},
        poll_interval_sec,
        results_dir,
        "translate",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Async API smoke test without poll timeout.")
    parser.add_argument("--base-url", default="http://localhost:3000", help="Backend base URL")
    parser.add_argument("--user-id", default="smoke-test-user", help="Prompt-service userId")
    parser.add_argument("--ask", default="What is IFRS 9?", help="Ask question")
    parser.add_argument("--followup", default="Summarize in 3 bullets.", help="Follow-up question")
    parser.add_argument("--poll-interval-sec", type=float, default=1.0, help="Status polling interval")
    parser.add_argument(
        "--results-dir",
        default="api_test_results",
        help="Directory where request/poll payloads are stored as JSON",
    )
    args = parser.parse_args()

    run_smoke(
        args.base_url,
        args.user_id,
        args.ask,
        args.followup,
        args.poll_interval_sec,
        Path(args.results_dir),
    )


if __name__ == "__main__":
    main()

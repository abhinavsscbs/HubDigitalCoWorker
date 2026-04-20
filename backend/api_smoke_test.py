"""Simple API smoke test helpers for HubDigitalCoWorker backend.

The key behavior here is that all JSON artifacts are written with UTF-8
encoding to avoid platform-default encoding issues (notably cp1252 on Windows).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


TERMINAL_STATES = {"completed", "failed", "error", "cancelled"}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _poll_until_terminal(
    base_url: str,
    prompt_id: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
    trace_file: Path,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    poll_history: List[Dict[str, Any]] = []

    while time.time() < deadline:
        response = requests.get(f"{base_url}/api/prompt/{prompt_id}", timeout=30)
        response.raise_for_status()
        payload = response.json()
        poll_history.append(payload)

        status = str(payload.get("status", "")).lower()
        if status in TERMINAL_STATES:
            _write_json(trace_file, {"promptId": prompt_id, "pollHistory": poll_history})
            return payload

        time.sleep(poll_interval_seconds)

    _write_json(trace_file, {"promptId": prompt_id, "pollHistory": poll_history})
    raise TimeoutError(
        f"Prompt {prompt_id} did not reach a terminal state in {timeout_seconds}s"
    )


def _start_and_poll(
    base_url: str,
    route: str,
    request_payload: Dict[str, Any],
    timeout_seconds: int,
    poll_interval_seconds: float,
    results_dir: Path,
    step_name: str,
) -> Dict[str, Any]:
    start_resp = requests.post(f"{base_url}{route}", json=request_payload, timeout=30)
    start_resp.raise_for_status()
    start_payload = start_resp.json()

    prompt_id = start_payload.get("promptId") or start_payload.get("id")
    if not prompt_id:
        _write_json(results_dir / f"{step_name}_start_response.json", start_payload)
        raise RuntimeError(f"No promptId/id found in response for step '{step_name}'.")

    _write_json(results_dir / f"{step_name}_start_response.json", start_payload)

    final_resp = _poll_until_terminal(
        base_url,
        str(prompt_id),
        timeout_seconds,
        poll_interval_seconds,
        results_dir / f"{step_name}_poll_history.json",
    )
    _write_json(results_dir / f"{step_name}_final_response.json", final_resp)
    return final_resp


def run_smoke(
    base_url: str,
    user_query: str,
    timeout_seconds: int,
    poll_interval_seconds: float,
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    _start_and_poll(
        base_url=base_url,
        route="/api/translate",
        request_payload={"query": user_query, "targetLanguage": "ar"},
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        results_dir=results_dir,
        step_name="translate",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run API smoke test flow")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--query", default="Summarize IFRS 15 core principles.")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--results-dir", default="smoke_results")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_smoke(
        args.base_url,
        args.query,
        args.timeout_seconds,
        args.poll_interval_seconds,
        Path(args.results_dir),
    )


if __name__ == "__main__":
    main()

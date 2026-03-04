from typing import Iterable

from flask import request

from .errors import ValidationError


def parse_json_payload() -> dict:
    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, dict):
        raise ValidationError("Invalid JSON payload")
    return payload


def require_non_empty_string_fields(payload: dict, fields: Iterable[str]) -> dict:
    cleaned = {}
    for field in fields:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"Field '{field}' is required and must be a non-empty string")
        cleaned[field] = value.strip()
    return cleaned

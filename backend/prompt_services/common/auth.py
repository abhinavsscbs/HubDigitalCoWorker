import hmac
from functools import wraps

from flask import current_app, jsonify, request


def unauthorized_response(realm: str):
    response = jsonify({"error": "Unauthorized"})
    response.status_code = 401
    response.headers["WWW-Authenticate"] = f'Basic realm="{realm}"'
    return response


def require_basic_auth(realm: str):
    """Flask decorator enforcing Basic Authentication."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            auth = request.authorization
            expected_user = str(current_app.config.get("BASIC_AUTH_USER", ""))
            expected_pass = str(current_app.config.get("BASIC_AUTH_PASS", ""))

            if not auth or (auth.type or "").lower() != "basic":
                return unauthorized_response(realm)

            user_ok = hmac.compare_digest(auth.username or "", expected_user)
            pass_ok = hmac.compare_digest(auth.password or "", expected_pass)
            if not (user_ok and pass_ok):
                return unauthorized_response(realm)

            return func(*args, **kwargs)

        return wrapper

    return decorator

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_common_config() -> dict:
    """Load shared config from environment variables."""
    return {
        "BASIC_AUTH_USER": os.getenv("BASIC_AUTH_USER", "admin"),
        "BASIC_AUTH_PASS": os.getenv("BASIC_AUTH_PASS", "changeme"),
        "REQUEST_TIMEOUT_SECONDS": int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60")),
        "PER_USER_RATE_LIMIT": int(os.getenv("PER_USER_RATE_LIMIT", "10")),
        "PER_USER_RATE_WINDOW_SECONDS": int(os.getenv("PER_USER_RATE_WINDOW_SECONDS", "60")),
        "GLOBAL_RATE_LIMIT": int(os.getenv("GLOBAL_RATE_LIMIT", "5")),
        "GLOBAL_RATE_WINDOW_SECONDS": float(os.getenv("GLOBAL_RATE_WINDOW_SECONDS", "1")),
        "MAX_HISTORY_PER_CONTEXT": int(os.getenv("MAX_HISTORY_PER_CONTEXT", "20")),
        "DEV_AUTO_COMPLETE": _env_bool("DEV_AUTO_COMPLETE", True),
        "DEV_AUTO_COMPLETE_AFTER_SECONDS": int(os.getenv("DEV_AUTO_COMPLETE_AFTER_SECONDS", "5")),
    }

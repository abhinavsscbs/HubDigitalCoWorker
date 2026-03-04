import random
import string
from datetime import datetime, timezone


def generate_prompt_id() -> str:
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    return f"IFRS-{date_part}-{suffix}"


def derive_prompt_title(prompt_text: str, words: int = 10) -> str:
    tokens = prompt_text.split()
    return " ".join(tokens[:words])


def tabular_data_for_prompt(prompt_text: str) -> dict:
    lowered = prompt_text.lower()
    if "table" in lowered or "tabular" in lowered:
        return {
            "headers": ["col1", "col2"],
            "rows": [
                ["row1", "value1"],
                ["row2", "value2"],
                ["row3", "value3"],
            ],
        }
    return {"headers": [], "rows": []}

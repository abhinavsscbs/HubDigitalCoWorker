from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


class BasePromptHistoryStore:
    def add_prompt_record(self, user_id: str, record: dict) -> None:
        raise NotImplementedError

    def get_prompt_record(self, user_id: str, prompt_id: str) -> Optional[dict]:
        raise NotImplementedError

    def get_latest_thread_record(self, user_id: str, original_prompt_id: str) -> Optional[dict]:
        raise NotImplementedError

    def update_prompt_record(self, user_id: str, prompt_id: str, updates: dict) -> bool:
        raise NotImplementedError

    def seed_prompt(self, user_id: str, record: dict) -> None:
        self.add_prompt_record(user_id, record)

    def reset(self) -> None:
        raise NotImplementedError


class InMemoryPromptHistoryStore(BasePromptHistoryStore):
    def __init__(self, max_entries_per_user: int = 500):
        self.max_entries_per_user = max_entries_per_user
        self._lock = threading.Lock()
        self._data: Dict[str, List[dict]] = {}

    def add_prompt_record(self, user_id: str, record: dict) -> None:
        with self._lock:
            items = self._data.setdefault(user_id, [])
            items.insert(0, dict(record))
            if len(items) > self.max_entries_per_user:
                del items[self.max_entries_per_user :]

    def get_prompt_record(self, user_id: str, prompt_id: str) -> Optional[dict]:
        with self._lock:
            for item in self._data.get(user_id, []):
                if item.get("promptId") == prompt_id:
                    return dict(item)
        return None

    def get_latest_thread_record(self, user_id: str, original_prompt_id: str) -> Optional[dict]:
        with self._lock:
            for item in self._data.get(user_id, []):
                if item.get("originalPromptId") == original_prompt_id:
                    return dict(item)
        return None

    def update_prompt_record(self, user_id: str, prompt_id: str, updates: dict) -> bool:
        with self._lock:
            items = self._data.get(user_id, [])
            for index, item in enumerate(items):
                if item.get("promptId") == prompt_id:
                    merged = dict(item)
                    merged.update(updates)
                    merged["updated_at"] = datetime.now(timezone.utc).isoformat()
                    items[index] = merged
                    return True
        return False

    def reset(self) -> None:
        with self._lock:
            self._data.clear()


class JsonFilePromptHistoryStore(BasePromptHistoryStore):
    """
    Thread-safe local JSON-file store for prompt records.
    Shared by all prompt services when they use the same directory.
    """

    def __init__(self, max_entries_per_user: int = 500, base_dir: Optional[str] = None):
        self.max_entries_per_user = max_entries_per_user
        self._lock = threading.Lock()
        if base_dir:
            self._base_dir = Path(base_dir).resolve()
        else:
            self._base_dir = Path(__file__).resolve().parents[1] / "data" / "prompt_history"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_user_id(user_id: str) -> str:
        if not user_id:
            return "unknown"
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", user_id.strip().lower())
        return safe.strip("_") or "unknown"

    def _file_path(self, user_id: str) -> Path:
        return self._base_dir / f"{self._safe_user_id(user_id)}.json"

    def _load(self, user_id: str) -> List[dict]:
        path = self._file_path(user_id)
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            return raw if isinstance(raw, list) else []
        except Exception:
            return []

    def _save(self, user_id: str, records: List[dict]) -> None:
        path = self._file_path(user_id)
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(records, handle, ensure_ascii=True)
        except Exception:
            return None

    def add_prompt_record(self, user_id: str, record: dict) -> None:
        with self._lock:
            items = self._load(user_id)
            items.insert(0, dict(record))
            if len(items) > self.max_entries_per_user:
                items = items[: self.max_entries_per_user]
            self._save(user_id, items)

    def get_prompt_record(self, user_id: str, prompt_id: str) -> Optional[dict]:
        with self._lock:
            for item in self._load(user_id):
                if isinstance(item, dict) and item.get("promptId") == prompt_id:
                    return dict(item)
        return None

    def get_latest_thread_record(self, user_id: str, original_prompt_id: str) -> Optional[dict]:
        with self._lock:
            for item in self._load(user_id):
                if isinstance(item, dict) and item.get("originalPromptId") == original_prompt_id:
                    return dict(item)
        return None

    def update_prompt_record(self, user_id: str, prompt_id: str, updates: dict) -> bool:
        with self._lock:
            items = self._load(user_id)
            for index, item in enumerate(items):
                if isinstance(item, dict) and item.get("promptId") == prompt_id:
                    merged = dict(item)
                    merged.update(updates)
                    merged["updated_at"] = datetime.now(timezone.utc).isoformat()
                    items[index] = merged
                    self._save(user_id, items)
                    return True
        return False

    def reset(self) -> None:
        with self._lock:
            for path in self._base_dir.glob("*.json"):
                try:
                    path.unlink()
                except Exception:
                    continue

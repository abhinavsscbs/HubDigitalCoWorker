import time
from collections import defaultdict, deque
from threading import Lock


class InMemoryRateLimiter:
    """Thread-safe in-memory limiter for single-instance deployments only."""

    def __init__(
        self,
        per_user_limit: int,
        per_user_window_seconds: float,
        global_limit: int,
        global_window_seconds: float,
    ):
        self.per_user_limit = per_user_limit
        self.per_user_window_seconds = per_user_window_seconds
        self.global_limit = global_limit
        self.global_window_seconds = global_window_seconds

        self._user_hits = defaultdict(deque)
        self._global_hits = deque()
        self._lock = Lock()

    @staticmethod
    def _prune(events: deque, now: float, window: float) -> None:
        cutoff = now - window
        while events and events[0] <= cutoff:
            events.popleft()

    def allow(self, user_id: str):
        now = time.monotonic()

        with self._lock:
            self._prune(self._global_hits, now, self.global_window_seconds)
            if len(self._global_hits) >= self.global_limit:
                return False, "overall"

            user_events = self._user_hits[user_id]
            self._prune(user_events, now, self.per_user_window_seconds)
            if len(user_events) >= self.per_user_limit:
                return False, "per-user"

            self._global_hits.append(now)
            user_events.append(now)
            return True, None

    def reset(self) -> None:
        with self._lock:
            self._user_hits.clear()
            self._global_hits.clear()

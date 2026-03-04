from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError


_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="prompt-service")


class TimeoutExceeded(Exception):
    pass


def run_with_timeout(func, timeout_seconds: int):
    future = _EXECUTOR.submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutExceeded(f"Request exceeded {timeout_seconds} seconds") from exc

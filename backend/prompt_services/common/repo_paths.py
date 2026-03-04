from pathlib import Path
import sys


def ensure_repo_paths() -> None:
    """
    Ensure imports can access:
    - repo root (for rag_engine/*, rag_config.py, llm_client.py)
    - backend dir (for session_store.py)
    """
    current = Path(__file__).resolve()
    backend_dir = current.parents[2]
    repo_root = current.parents[3]

    backend_str = str(backend_dir)
    repo_str = str(repo_root)

    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

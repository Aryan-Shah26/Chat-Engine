import json
from pathlib import Path

DATASET_PATH = "data/eval_qa.json"


def load_qa_dataset(path: str = DATASET_PATH) -> list[dict]:
    """
    Each entry: {"question": str, "relevant": [{"source": str, "page": int, "chunk": int}, ...]}
    "relevant" = ground-truth chunk keys that should be retrieved.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No eval dataset at {path}. Create one first (see schema in docstring).")
    return json.loads(p.read_text())
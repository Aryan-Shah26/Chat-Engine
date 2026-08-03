from src.core.llm_client import LLMClient
from src.retrieval.hybrid_retriever import load_hybrid_index

_retriever_cache: dict[str, tuple] = {}


def get_llm_client() -> LLMClient:
    return LLMClient()


def get_collection_index(collection_id: str, top_k: int = 5, refresh: bool = False):
    """
    Lazily loads and caches (dense_retriever, bm25_retriever, vectorstore) for
    a collection so repeated chat requests don't re-embed/reload from disk.
    """
    if refresh or collection_id not in _retriever_cache:
        _retriever_cache[collection_id] = load_hybrid_index(collection_id, top_k=top_k)
    return _retriever_cache[collection_id]


def invalidate_collection_cache(collection_id: str):
    _retriever_cache.pop(collection_id, None)

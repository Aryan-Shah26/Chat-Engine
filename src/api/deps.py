from src.core.llm_client import LLMClient
from src.retrieval.hybrid_retriever import load_hybrid_index
from src.retrieval.retriever import load_chroma_retriever

_retriever_cache: dict[str, tuple] = {}


_llm_client: LLMClient | None = None

def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client

def get_collection_index(collection_id: str, top_k: int = 5):
    """Lazily loads and caches (dense_retriever, bm25_retriever, vectorstore) for a collection."""
    if collection_id not in _retriever_cache:
        _retriever_cache[collection_id] = load_hybrid_index(collection_id, top_k=top_k)
    return _retriever_cache[collection_id]


def get_collection_vectorstore(collection_id: str, top_k: int = 5, refresh: bool = False):
    """
    Lazily loads and caches (dense_retriever, bm25_retriever, vectorstore) for
    a collection so repeated chat requests don't re-embed/reload from disk.
    """
    if collection_id in _retriever_cache:
        _, _, vectorstore = _retriever_cache[collection_id]
        return vectorstore
    _, vectorstore = load_chroma_retriever(collection_id)
    return vectorstore

def invalidate_collection_cache(collection_id: str):
    _retriever_cache.pop(collection_id, None)

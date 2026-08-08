from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.core.config import settings

_embedder = None
_reranker = None


def _chroma_path(collection_id: str) -> str:
    return str(Path(settings.data_dir) / collection_id / "chroma_db")


def get_embedder() -> HuggingFaceEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _embedder


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def build_chroma_retriever(chunks: list[dict], collection_id: str, top_k: int = 5):
    """Builds a Chroma retriever scoped to a single collection_id."""
    embedder = get_embedder()
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    vectorstore = Chroma.from_texts(
        texts=texts, embedding=embedder, metadatas=metadatas,
        persist_directory=_chroma_path(collection_id),
    )
    return vectorstore.as_retriever(search_kwargs={"k": top_k * 4}), vectorstore


def load_chroma_retriever(collection_id: str, top_k: int = 5):
    """Loads a persisted Chroma retriever for a collection_id."""
    embedder = get_embedder()
    vectorstore = Chroma(persist_directory=_chroma_path(collection_id), embedding_function=embedder)
    return vectorstore.as_retriever(search_kwargs={"k": top_k * 4}), vectorstore


def add_chroma_retriever(chunks: list[dict], vectorstore: Chroma):
    """Adds new chunks to an existing Chroma retriever."""
    vectorstore.add_texts(
        texts=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def rerank(query: str, docs: list, top_k: int = 5) -> list:
    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


def rerank_with_scores(query: str, docs: list, top_k: int = 5) -> tuple[list, list[float]]:
    """Like rerank() but also returns the cross-encoder scores of the top-k docs."""
    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [doc for doc, _ in scored], [round(float(s), 4) for _, s in scored]

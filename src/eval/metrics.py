def _doc_key(doc) -> tuple:
    meta = doc.metadata
    return (meta.get("source"), meta.get("page"), meta.get("chunk"))


def recall_at_k(retrieved: list, relevant: list[dict], k: int) -> float:
    """Fraction of relevant chunks present in top-k retrieved docs."""
    if not relevant:
        return 0.0
    retrieved_keys = {_doc_key(d) for d in retrieved[:k]}
    relevant_keys = {(r["source"], r["page"], r["chunk"]) for r in relevant}
    hits = retrieved_keys & relevant_keys
    return len(hits) / len(relevant_keys)


def mrr(retrieved: list, relevant: list[dict]) -> float:
    """Reciprocal rank of the first relevant doc found; 0 if none found."""
    relevant_keys = {(r["source"], r["page"], r["chunk"]) for r in relevant}
    for rank, doc in enumerate(retrieved, start=1):
        if _doc_key(doc) in relevant_keys:
            return 1.0 / rank
    return 0.0


def context_precision(retrieved: list, relevant: list[dict], k: int) -> float:
    """Fraction of top-k retrieved docs that are actually relevant."""
    if not retrieved[:k]:
        return 0.0
    relevant_keys = {(r["source"], r["page"], r["chunk"]) for r in relevant}
    hits = sum(1 for d in retrieved[:k] if _doc_key(d) in relevant_keys)
    return hits / len(retrieved[:k])
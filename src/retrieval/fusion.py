import hashlib


def _content_key(doc) -> str:
    """Returns a deterministic normalized content key for deduplication."""
    norm_text = " ".join(doc.page_content.strip().split())
    return hashlib.md5(norm_text.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(*ranked_lists: list, k: int = 60) -> list:
    """
    Fuses multiple ranked lists of LangChain Documents via Reciprocal
    Rank Fusion. score(doc) = sum(1 / (k + rank)) across every list the
    doc appears in. Returns docs sorted by fused score, deduplicated by content.
    """
    scores: dict[str, float] = {}
    doc_lookup: dict[str, object] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = _content_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_lookup.setdefault(key, doc)

    fused_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_lookup[key] for key in fused_keys]
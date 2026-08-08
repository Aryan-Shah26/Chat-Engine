from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

from src.core.config import settings
from src.core.llm_client import LLMClient

VERIFY_PROMPT = """Does the CONTEXT support the CLAIM? Answer only "yes" or "no".
CONTEXT: {context}
CLAIM: {claim}
Answer:"""

CITATION_PATTERN = re.compile(
    r"[\[【]([^\[\]【】]+?)(?:\s*[-–—:,]\s*(?:[Pp]age|[Pp]\.?|[Pp])?\s*(\d+)|\s*[-–—]\s*(\d+))?[\]】]"
)


def extract_cited_claims(answer: str) -> list[dict]:
    """
    Splits answer into sentences/items and pairs each with every inline citation found in it.
    Supports markdown bullets, multi-line answers, multiple citations per sentence,
    and various citation formats (hyphens, en-dashes, em-dashes, colons).
    """
    if not answer or not answer.strip():
        return []

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    claims = []

    for line in lines:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
        for sentence in sentences:
            matches = list(CITATION_PATTERN.finditer(sentence))
            for match in matches:
                source = match.group(1).strip()
                if source.lower() in ("unverified", ""):
                    continue

                page_str = match.group(2) or match.group(3)
                page = int(page_str) if page_str and page_str.isdigit() else None

                # Clean claim text for accurate LLM judge evaluation
                clean_text = CITATION_PATTERN.sub("", sentence).strip()
                clean_text = re.sub(r"\s+", " ", clean_text).strip()
                if not clean_text:
                    clean_text = sentence

                claims.append({
                    "text": clean_text,
                    "source": source,
                    "page": page,
                    "raw_citation": match.group(0),
                })
    return claims


def verify_citations(client: LLMClient, claims: list[dict], retrieved_docs: list) -> list[dict]:
    """For each claim, asks the LLM whether the matching source doc actually supports it."""
    if not claims:
        return []

    with ThreadPoolExecutor(max_workers=min(len(claims), 8)) as pool:
        return list(pool.map(lambda c: _verify_one(client, c, retrieved_docs), claims))


def _verify_one(client: LLMClient, claim: dict, retrieved_docs: list) -> dict:
    claim_source = Path(claim["source"]).name.lower().strip()
    claim_page = claim.get("page")
    claim_page = str(claim_page).strip() if claim_page is not None else None

    matching_docs = []
    for doc in retrieved_docs:
        doc_source = Path(str(doc.metadata.get("source", ""))).name.lower().strip()
        doc_page = str(doc.metadata.get("page", "")).strip()

        source_matches = (
            doc_source == claim_source
            or (claim_source and claim_source in doc_source)
            or (doc_source and doc_source in claim_source)
        )

        page_matches = (
            claim_page is None  # no page specified in citation → match any page
            or doc_page == claim_page
            or (not doc_page and claim_page == "1")
            or (doc_page == "1" and not claim_page)
        )

        if source_matches and page_matches:
            matching_docs.append(doc)

    if not matching_docs:
        return {**claim, "verified": False}

    context = "\n\n".join(doc.page_content for doc in matching_docs)
    try:
        verdict = client.complete(
            [{"role": "user", "content": VERIFY_PROMPT.format(context=context, claim=claim["text"])}],
            max_tokens=20,
            temperature=0.0,
            model=settings.critic_model,
        )
        is_verified = verdict.strip().lower().startswith("yes") or "yes" in verdict.strip().lower().split()[:3]
    except Exception:
        is_verified = False

    return {**claim, "verified": is_verified}


def filter_hallucinated_citations(answer: str, verified_claims: list[dict]) -> str:
    """Strips the citation tag (keeps the sentence) for any claim that failed verification."""
    for claim in verified_claims:
        if not claim.get("verified", False):
            raw = claim.get("raw_citation")
            if raw and raw in answer:
                answer = answer.replace(raw, "[unverified]", 1)
            else:
                source_esc = re.escape(claim.get("source", ""))
                page = claim.get("page")
                pattern = re.compile(
                    rf"\[\s*{source_esc}\s*(?:[-–—:,]\s*(?:[Pp]age|[Pp]\.?|[Pp])?\s*{page}|\s*[-–—]\s*{page})?\s*\]",
                    re.IGNORECASE,
                )
                answer = pattern.sub("[unverified]", answer)
    return answer

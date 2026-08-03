from src.core.config import settings
from src.core.llm_client import LLMClient

REWRITE_PROMPT = """Rewrite the user's question to be specific and unambiguous, \
using context clues about the kind of document collection being searched. \
Return ONLY the rewritten question, nothing else.
Question: {question}
Rewritten:"""

MULTI_QUERY_PROMPT = """Generate {n} different search queries that would help answer \
the user's question, each focusing on a different angle or phrasing. \
Return ONLY the queries, one per line, no numbering.
Question: {question}
Queries:"""


def rewrite_query(client: LLMClient, question: str) -> str:
    return client.complete(
        [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
        max_tokens=128, temperature=0.0, model=settings.critic_model,
    )


def generate_multi_queries(client: LLMClient, question: str, n: int = 4) -> list[str]:
    raw = client.complete(
        [{"role": "user", "content": MULTI_QUERY_PROMPT.format(question=question, n=n)}],
        max_tokens=256, temperature=0.3, model=settings.critic_model,
    )
    queries = [line.strip("- ").strip() for line in raw.split("\n") if line.strip()]
    return [question] + queries  # keep original as one of the variants

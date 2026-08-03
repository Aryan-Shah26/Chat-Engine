from src.core.config import settings
from src.core.llm_client import LLMClient

FAITHFULNESS_PROMPT = """Does the ANSWER contain any claim NOT supported by the CONTEXT? \
Answer only "yes" (unsupported claims exist) or "no" (fully supported).
CONTEXT: {context}
ANSWER: {answer}
Answer:"""


def faithfulness(client: LLMClient, answer: str, retrieved_docs: list) -> float:
    """
    LLM-judge faithfulness score in [0, 1]: 1.0 if every claim in the answer
    is grounded in the retrieved context, 0.0 if the judge flags unsupported
    content.
    """
    if not retrieved_docs:
        return 0.0
    context = "\n\n".join(d.page_content for d in retrieved_docs)
    verdict = client.complete(
        [{"role": "user", "content": FAITHFULNESS_PROMPT.format(context=context, answer=answer)}],
        max_tokens=5, temperature=0.0, model=settings.critic_model,
    )
    return 0.0 if verdict.strip().lower().startswith("yes") else 1.0

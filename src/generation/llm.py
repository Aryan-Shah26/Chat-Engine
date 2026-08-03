from src.core.llm_client import LLMClient
from src.generation.prompts import get_messages


def generate_answer(client: LLMClient, retrieved_docs: list, question: str) -> dict:
    if not retrieved_docs:
        return {
            "answer": "I could not find relevant information in the indexed documents.",
            "sources": [],
        }

    context = "\n\n".join(
        f"[{doc.metadata['source']} - Page {doc.metadata['page']}]: {doc.page_content}"
        for doc in retrieved_docs
    )
    answer = client.complete(get_messages(context, question), max_tokens=1024, temperature=0.2)
    return {
        "answer": answer,
        "sources": [f"{doc.metadata['source']} - Page {doc.metadata['page']}" for doc in retrieved_docs],
    }

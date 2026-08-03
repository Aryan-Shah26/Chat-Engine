def get_messages(context: str, question: str) -> list[dict]:
    system = (
        "You are an assistant that answers questions using ONLY the provided context. "
        "If the answer is not in the context, say 'I cannot find this in the provided documents.' "
        "Always cite the source document and page your answer is based on, in the form [source - Page N]."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

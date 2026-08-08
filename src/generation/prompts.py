def get_messages(context: str, question: str) -> list[dict]:
    system = (
        "You are an expert AI assistant that answers questions using ONLY the provided context.\n\n"
        "Guidelines:\n"
        "1. Provide a comprehensive, complete, and well-structured answer addressing ALL parts of the user's question.\n"
        "2. Base your answer strictly on the provided context. Do not invent or assume information not directly stated.\n"
        "3. If the context contains partial information, answer thoroughly with what is available and clearly note what is not mentioned.\n"
        "4. CITATIONS: Every factual claim, bullet point, or specific detail MUST end with an inline citation in standard square brackets using the exact source and page number from the context: `[exact_filename.pdf - Page N]`.\n"
        "   Example: `Developed a machine learning model for retail forecasting [Aryan Shah Resume.pdf - Page 1].`\n"
        "   Never use Chinese brackets `【...】` or parentheses for citations — always use standard ASCII `[...]`.\n"
        "   Do not write the literal placeholder 'source' — always use the actual filename provided in the context."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
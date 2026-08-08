from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk(documents: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    if not documents:
        raise ValueError("No documents to chunk. The document appears to be empty or unreadable.")

    separators = ["\n\n", "\n", "• ", ". ", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    all_chunks = []

    for document in documents:
        text = document["text"]
        chunks = splitter.split_text(text)

        for index, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    **document["metadata"],
                    "chunk": index,
                },
            })
    return all_chunks
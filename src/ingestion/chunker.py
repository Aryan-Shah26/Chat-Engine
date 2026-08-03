from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk(documents : list[dict]) -> list[dict] :

    if not documents :
        raise ValueError("No documents to chunk. The document appears to be empty or unreadable.")
    chunk_size = 512
    chunk_overlap = 20
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    all_chunks = []

    for document in documents:
        text = document["text"]
        chunks = splitter.split_text(text)

        for index, chunk in enumerate(chunks) :
            all_chunks.append({
                "text" : chunk,
                "metadata" : {
                    **document["metadata"],
                    "chunk" : index
                }
            })
    return all_chunks
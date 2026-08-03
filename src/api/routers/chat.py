from fastapi import APIRouter, HTTPException

from src.agent.graph import build_agent_graph, run_agent
from src.api.deps import get_collection_index, get_llm_client
from src.generation.citation_check import extract_cited_claims, filter_hallucinated_citations, verify_citations
from src.generation.llm import generate_answer
from src.retrieval.hybrid_retriever import hybrid_search, multi_query_hybrid_search, multi_source_search
from src.retrieval.query_transform import generate_multi_queries
from src.store import db
from src.store.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/collections", tags=["chat"])


@router.post("/{collection_id}/chat", response_model=ChatResponse)
def chat(collection_id: str, payload: ChatRequest):
    if not db.get_collection(collection_id):
        raise HTTPException(404, "Collection not found")

    dense_r, bm25_r, _ = get_collection_index(collection_id, top_k=payload.top_k)
    if dense_r is None:
        raise HTTPException(400, "No documents indexed in this collection yet")

    client = get_llm_client()

    try:
        if payload.mode == "agentic":
            graph_app = build_agent_graph(client, dense_r, bm25_r, top_k=payload.top_k)
            result = run_agent(graph_app, payload.question)
            answer, sources, docs = result["answer"], result["sources"], result["retrieved"]
        else:
            if payload.mode == "multi_query":
                queries = generate_multi_queries(client, payload.question, n=4)
                docs = multi_query_hybrid_search(queries, dense_r, bm25_r, top_k=payload.top_k)
            elif payload.mode == "multi_source":
                sources_list = [d["filename"] for d in db.list_documents(collection_id)]
                docs = multi_source_search(payload.question, dense_r, bm25_r, sources_list, top_k_per_source=3)
            else:
                docs = hybrid_search(payload.question, dense_r, bm25_r, top_k=payload.top_k)
            result = generate_answer(client, docs, payload.question)
            answer, sources = result["answer"], result["sources"]
    except RuntimeError as e:
        raise HTTPException(502, str(e))

    if payload.verify_citations:
        claims = extract_cited_claims(answer)
        if claims:
            verified = verify_citations(client, claims, docs)
            answer = filter_hallucinated_citations(answer, verified)

    return ChatResponse(answer=answer, sources=sources)

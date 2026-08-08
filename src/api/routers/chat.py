from fastapi import APIRouter, HTTPException

from src.agent.graph import build_agent_graph, run_agent
from src.api.deps import get_collection_index, get_llm_client
from src.core.metrics import MetricsCollector
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
    mc = MetricsCollector()
    mc.start_timer("total")
    mc.record("query_mode", payload.mode)

    try:
        if payload.mode == "agentic":
            mc.start_timer("agentic_loop")
            graph_app = build_agent_graph(client, dense_r, bm25_r, top_k=payload.top_k)
            result = run_agent(graph_app, payload.question)
            mc.stop_timer("agentic_loop")
            answer, sources, docs = result["answer"], result["sources"], result["retrieved"]
            # Merge agentic metrics into the collector
            agentic_metrics = result.get("metrics", {})
            for k, v in agentic_metrics.items():
                mc.record(k, v)
            mc.record("chunks_retrieved", len(docs))
            mc.record("unique_sources", len({d.metadata.get("source") for d in docs}))
        else:
            mc.start_timer("retrieval")
            if payload.mode == "multi_query":
                queries = generate_multi_queries(client, payload.question, n=4)
                mc.record("planner_queries_generated", len(queries))
                docs = multi_query_hybrid_search(queries, dense_r, bm25_r, top_k=payload.top_k, metrics=mc)
            elif payload.mode == "multi_source":
                sources_list = [d["filename"] for d in db.list_documents(collection_id)]
                docs = multi_source_search(payload.question, dense_r, bm25_r, sources_list, top_k_per_source=3, metrics=mc)
            else:
                docs = hybrid_search(payload.question, dense_r, bm25_r, top_k=payload.top_k, metrics=mc)
            mc.stop_timer("retrieval")

            mc.start_timer("generation")
            result = generate_answer(client, docs, payload.question)
            mc.stop_timer("generation")
            answer, sources = result["answer"], result["sources"]
    except RuntimeError as e:
        raise HTTPException(502, str(e))

    # Answer-level metrics
    mc.record("answer_length_chars", len(answer))
    mc.record("answer_length_words", len(answer.split()))

    if payload.verify_citations:
        mc.start_timer("citation_verification")
        claims = extract_cited_claims(answer)
        mc.record("citations_extracted", len(claims))
        if claims:
            verified = verify_citations(client, claims, docs)
            num_verified = sum(1 for c in verified if (isinstance(c, dict) and c.get("verified", False)))
            mc.record("citations_verified", num_verified)
            answer = filter_hallucinated_citations(answer, verified)
        else:
            mc.record("citations_verified", 0)
        mc.stop_timer("citation_verification")
    else:
        mc.record("citations_extracted", 0)
        mc.record("citations_verified", 0)

    mc.stop_timer("total")

    return ChatResponse(answer=answer, sources=sources, metrics=mc.to_dict())

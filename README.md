# RAG Chat Engine

A generalized, deployable Retrieval-Augmented Generation chatbot. Ingests any
mix of PDF / HTML / TXT / MD documents into isolated **collections**, and
answers questions over them via hybrid retrieval, multi-query expansion,
cross-document reasoning, or a self-critiquing agentic loop — all behind a
single FastAPI service.

Originally built as a research-paper Q&A tool; generalized here into a
domain-agnostic document chat engine with a proper service boundary.

---

## Architecture

```
Client (Streamlit UI / curl / any HTTP caller)
        |
        v
FastAPI service  (src/api)
  ├── /collections            create/list collections
  ├── /collections/{id}/documents   ingest a file into a collection
  └── /collections/{id}/chat        ask a question (4 modes)
        |
        v
src/ingestion   parse (pdf/html/txt/md) -> chunk -> extract tables
src/retrieval   Chroma (dense) + BM25 (sparse) -> RRF fusion -> cross-encoder rerank
src/agent       LangGraph planner -> retriever -> critic -> answer (Reflexion loop)
src/generation  Groq-backed answer generation + citation verification
src/eval        Recall@K / MRR / Context Precision + LLM-judge Faithfulness
src/store       SQLite registry (collections, documents) — no in-memory state
```

Every module talks to the LLM through `src/core/llm_client.py`. Swapping
providers means editing one file, not five.

### Chat modes

| Mode | Behavior |
|---|---|
| `hybrid` | Dense + sparse retrieval, RRF fusion, rerank |
| `multi_query` | LLM expands the question into N variants before hybrid search |
| `multi_source` | Guarantees representation from every document in the collection (cross-document comparison) |
| `agentic` | LangGraph loop: plan → retrieve → critic scores confidence → retries or answers |

All modes optionally run citation verification, which strips any `[source -
Page N]` tag the LLM couldn't actually justify against the retrieved context.

---

## What changed from the original research-paper bot

- Wired `extract_tables()` into the ingestion pipeline — it existed before but was never called, so table content was never actually searchable.
- Replaced the "Ragas metrics" claim with a real from-scratch Faithfulness metric (LLM-as-judge), alongside the existing Recall@K/MRR/Context Precision.
- Fixed a real bug: the original code indexed documents under their random temp-file name, not the uploaded filename, so citations pointed at gibberish.
- Swapped HuggingFace Inference API for Groq.
- Added multi-collection isolation via SQLite (was single global in-memory session).
- Split UI from logic: Streamlit is now a thin HTTP client, not the orchestrator.
- Added Docker + Render deployment config.

---

## Setup

```bash
git clone <this-repo>
cd rag-chat-engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
```

## Run locally

```bash
uvicorn src.api.main:app --reload          # API on :8000
streamlit run ui/app.py                    # UI on :8501
```

## Run with Docker

```bash
docker compose up --build
```

## Deploy to Render

1. Push to GitHub.
2. Render → New → Blueprint → point at this repo (`render.yaml` is picked up automatically).
3. Set `GROQ_API_KEY` in the Render dashboard (marked `sync: false` so it's not committed).

---

## API quick reference

```bash
# Create a collection
curl -X POST localhost:8000/collections -H "Content-Type: application/json" -d '{"name": "contracts"}'

# Ingest a document
curl -X POST localhost:8000/collections/{id}/documents -F "file=@doc.pdf"

# Ask a question
curl -X POST localhost:8000/collections/{id}/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the termination clause?", "mode": "hybrid"}'
```

---

## Tech Stack

FastAPI · LangGraph · LangChain · ChromaDB · BM25 (rank_bm25) · sentence-transformers
(embeddings + cross-encoder reranking) · Groq (Llama 3.3 / 3.1) · SQLite · Streamlit · Docker

## License

MIT

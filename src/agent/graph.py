import time

from langgraph.graph import END, StateGraph

from src.agent.state import AgentState
from src.core.config import settings
from src.core.llm_client import LLMClient
from src.generation.llm import generate_answer
from src.retrieval.hybrid_retriever import multi_query_hybrid_search
from src.retrieval.query_transform import generate_multi_queries

MAX_ATTEMPTS = 2
CONFIDENCE_THRESHOLD = 0.6
CRITIC_PROMPT = """Rate 0-1 how well the CONTEXT answers the QUESTION. Reply with only a number.
QUESTION: {question}
CONTEXT: {context}
Score:"""


def build_agent_graph(client: LLMClient, dense_retriever, bm25_retriever, top_k: int = 5):
    """
    Wires planner -> retriever -> critic -> answer into a LangGraph. If
    critic confidence is below threshold and attempts remain, loops back
    to the planner with the original question (re-triggers multi-query).
    """

    def planner(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        queries = generate_multi_queries(client, state["question"], n=4)
        elapsed = round(time.perf_counter() - t0, 4)
        m = dict(state.get("metrics", {}))
        m["planner_sec"] = m.get("planner_sec", 0) + elapsed
        m["planner_queries_generated"] = len(queries)
        return {**state, "plan": queries, "attempts": state["attempts"] + 1, "metrics": m}

    def retriever(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        docs = multi_query_hybrid_search(state["plan"], dense_retriever, bm25_retriever, top_k=top_k)
        elapsed = round(time.perf_counter() - t0, 4)
        m = dict(state.get("metrics", {}))
        m["retriever_sec"] = m.get("retriever_sec", 0) + elapsed
        return {**state, "retrieved": docs, "metrics": m}

    def critic(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        if not state["retrieved"]:
            elapsed = round(time.perf_counter() - t0, 4)
            m = dict(state.get("metrics", {}))
            m["critic_sec"] = m.get("critic_sec", 0) + elapsed
            return {**state, "confidence": 0.0, "metrics": m}
        context = "\n\n".join(doc.page_content for doc in state["retrieved"])
        score_str = client.complete(
            [{"role": "user", "content": CRITIC_PROMPT.format(question=state["question"], context=context)}],
            max_tokens=5, temperature=0.0, model=settings.critic_model,
        )
        try:
            score = float(score_str)
        except ValueError:
            score = 0.0
        elapsed = round(time.perf_counter() - t0, 4)
        m = dict(state.get("metrics", {}))
        m["critic_sec"] = m.get("critic_sec", 0) + elapsed
        m["critic_confidence"] = round(score, 4)
        # Track confidence history for delta computation
        history = m.get("confidence_history", [])
        history = list(history) + [round(score, 4)]
        m["confidence_history"] = history
        return {**state, "confidence": score, "metrics": m}

    def answer(state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        result = generate_answer(client, state["retrieved"], state["question"])
        elapsed = round(time.perf_counter() - t0, 4)
        m = dict(state.get("metrics", {}))
        m["answer_generation_sec"] = elapsed
        m["retry_count"] = max(0, state["attempts"] - 1)
        # Compute confidence improvement
        history = m.get("confidence_history", [])
        if len(history) >= 2:
            m["confidence_improvement"] = round(history[-1] - history[0], 4)
        else:
            m["confidence_improvement"] = 0.0
        return {**state, "answer": result["answer"], "sources": result["sources"], "metrics": m}

    def should_retry(state: AgentState) -> str:
        if state["confidence"] >= CONFIDENCE_THRESHOLD or state["attempts"] >= MAX_ATTEMPTS:
            return "answer"
        return "planner"

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("retriever", retriever)
    graph.add_node("critic", critic)
    graph.add_node("answer", answer)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "critic")
    graph.add_conditional_edges("critic", should_retry, {"planner": "planner", "answer": "answer"})
    graph.add_edge("answer", END)
    return graph.compile()


def run_agent(app, question: str) -> dict:
    initial_state: AgentState = {
        "question": question, "plan": [], "retrieved": [],
        "confidence": 0.0, "attempts": 0, "answer": "", "sources": [],
        "metrics": {},
    }
    return app.invoke(initial_state)

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
        queries = generate_multi_queries(client, state["question"], n=4)
        return {**state, "plan": queries, "attempts": state["attempts"] + 1}

    def retriever(state: AgentState) -> AgentState:
        docs = multi_query_hybrid_search(state["plan"], dense_retriever, bm25_retriever, top_k=top_k)
        return {**state, "retrieved": docs}

    def critic(state: AgentState) -> AgentState:
        if not state["retrieved"]:
            return {**state, "confidence": 0.0}
        context = "\n\n".join(doc.page_content for doc in state["retrieved"])
        score_str = client.complete(
            [{"role": "user", "content": CRITIC_PROMPT.format(question=state["question"], context=context)}],
            max_tokens=5, temperature=0.0, model=settings.critic_model,
        )
        try:
            score = float(score_str)
        except ValueError:
            score = 0.0
        return {**state, "confidence": score}

    def answer(state: AgentState) -> AgentState:
        result = generate_answer(client, state["retrieved"], state["question"])
        return {**state, "answer": result["answer"], "sources": result["sources"]}

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
    }
    return app.invoke(initial_state)

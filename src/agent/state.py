from typing import TypedDict


class AgentState(TypedDict):
    question: str
    plan: list[str]          # sub-queries from planner
    retrieved: list          # Documents from latest retrieval round
    confidence: float        # critic's confidence score (0-1)
    attempts: int
    answer: str
    sources: list[str]
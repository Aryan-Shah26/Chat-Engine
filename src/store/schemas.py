from pydantic import BaseModel, Field


class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class CollectionOut(BaseModel):
    id: str
    name: str
    created_at: float


class DocumentOut(BaseModel):
    id: str
    collection_id: str
    filename: str
    chunk_count: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: str = Field("hybrid", pattern="^(hybrid|multi_query|multi_source|agentic)$")
    verify_citations: bool = True
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    metrics: dict = {}

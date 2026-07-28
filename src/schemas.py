# Task 5 - pydantic models so every function boundary passes validated data, not raw dicts
from pydantic import BaseModel, Field


# one retrieved chunk plus its similarity distance, passed from retrieval into prompt building
class RetrievedChunk(BaseModel):
    chunk_id: int
    text: str
    distance: float


# what the pipeline (and later the API) takes in for a single question
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="The user's question about the Contract Act, 1872")


# the LLM's own scope decision, parsed and validated from its structured JSON output (Task 7)
class ScopeClassification(BaseModel):
    is_scope: bool
    reason: str


# what the pipeline (and later the API) hands back for a single question
class ChatbotResponse(BaseModel):
    question: str
    answer: str
    sources: list[int] = Field(default_factory=list, description="chunk_ids the answer was grounded in")
    is_scope: bool

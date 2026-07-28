# Task 6 - the full pipeline: retrieve -> ground -> generate -> validated ChatbotResponse
import llm
import retrieval
import settings
from schemas import ChatbotResponse, QueryRequest


# is_scope is hardcoded True here - Task 7 adds a real scope-check step before this function
# runs at all, and short-circuits out-of-scope questions before they ever reach retrieval
def answer_question(store, request: QueryRequest) -> ChatbotResponse:
    chunks = retrieval.retrieve(store, request.question)

    # nothing relevant came back - say so instead of asking the LLM to guess from bad context
    if not chunks or chunks[0].distance > settings.LOW_CONFIDENCE_DISTANCE:
        return ChatbotResponse(
            question=request.question,
            answer=f"I don't have enough information in {settings.LAW_NAME} to answer that.",
            sources=[],
            is_scope=True,
        )

    prompt = retrieval.build_prompt(chunks, request.question)
    model = llm.get_llm()
    response = model.invoke(prompt)

    return ChatbotResponse(
        question=request.question,
        answer=response.content,
        sources=[c.chunk_id for c in chunks],
        is_scope=True,
    )

# Task 6/7 - the full pipeline: scope-check -> retrieve -> ground -> generate -> ChatbotResponse
import guardrails
import llm
import retrieval
import settings
from schemas import ChatbotResponse, QueryRequest


# scope is checked first and short-circuits everything else: no point spending an
# embedding + retrieval + generation call on a question we're going to decline anyway
def answer_question(store, request: QueryRequest) -> ChatbotResponse:
    scope = guardrails.check_scope(request.question)
    if not scope.is_scope:
        return ChatbotResponse(
            question=request.question,
            answer=guardrails.DECLINE_MESSAGE,
            sources=[],
            is_scope=False,
        )

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

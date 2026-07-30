# Task 6/7 - the full pipeline as a LangGraph StateGraph, not a linear function:
# condense -> retrieve -> check_scope -[conditional]-> generate
#                                      \-> low_confidence
#                                      \-> meta
#                                      \-> decline
# a checkpointer gives it multi-turn memory: the "condense" node rewrites a follow-up
# question (e.g. "what about for minors?") into a standalone one using prior turns, so
# retrieval/scope-check/generation all operate on a question that makes sense on its own.
#
# retrieval runs BEFORE the scope check (not after) so its result can feed the scope
# decision: a strong retrieval match overrides the LLM classifier's verdict. This exists
# because the classifier alone repeatedly misjudged genuinely in-scope questions about
# common-sounding Act terms ("what are voidable agreements", "what is an agreement") as
# too vague/off-topic - retrieval evidence is a more reliable signal for those than the
# classifier's guess at whether a question "sounds specific enough."
import operator
import uuid
from typing import Annotated, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

import guardrails
import llm
import retrieval
import settings
from schemas import ChatbotResponse, QueryRequest, RetrievedChunk

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and a new follow-up question, rewrite the follow-up as "
     "a standalone question that makes sense without the history - pull in whatever specific "
     "topic/entity the follow-up implicitly refers to. If the follow-up is already standalone "
     "(doesn't depend on prior context), return it completely unchanged. "
     "Output ONLY the resulting question, nothing else - no preamble, no quotes."),
    ("human", "History:\n{history}\n\nFollow-up question: {question}\n\nStandalone question:"),
])


# checkpointed state must stay JSON-safe (the MemorySaver serializes it with msgpack), so
# pydantic objects (ScopeClassification, RetrievedChunk) are kept as plain dict/str here and
# only reconstructed as typed objects locally inside whichever node needs them
class GraphState(TypedDict):
    question: str  # the raw question exactly as the user typed it this turn
    standalone_question: str  # question rewritten to stand alone, using history if any
    history: Annotated[list[tuple[str, str]], operator.add]  # (question, answer) pairs, appended each turn
    scope_category: str  # "in_scope" | "meta" | "out_of_scope"
    chunks: list[dict]  # RetrievedChunk.model_dump() per chunk
    answer: str
    sources: list[int]
    source_labels: list[str]
    is_scope: bool


# builds and compiles the graph once per vector store; the store and chat model are captured
# by closure since graph nodes only receive/return GraphState, not arbitrary dependencies
def build_graph(store) -> CompiledStateGraph:
    model = llm.get_llm()

    # first turn (no history) skips the LLM call entirely and passes the question through
    def condense_node(state: GraphState) -> dict:
        history = state.get("history") or []
        question = state["question"]
        if not history:
            return {"standalone_question": question}

        history_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in history[-3:])
        prompt = CONDENSE_PROMPT.invoke({"history": history_text, "question": question})
        response = model.invoke(prompt)
        return {"standalone_question": response.content.strip()}

    def retrieve_node(state: GraphState) -> dict:
        chunks = retrieval.retrieve(store, state["standalone_question"])
        return {"chunks": [c.model_dump() for c in chunks]}

    # a strong retrieval match (distance comfortably below STRONG_MATCH_DISTANCE) short-
    # circuits straight to "in_scope" without even calling the classifier - cheaper and more
    # reliable than the LLM's judgment for questions that already have clear evidence
    def check_scope_node(state: GraphState) -> dict:
        chunks = state["chunks"]
        if chunks and chunks[0]["distance"] < settings.STRONG_MATCH_DISTANCE:
            return {"scope_category": "in_scope"}

        scope = guardrails.check_scope(state["standalone_question"])
        return {"scope_category": scope.category}

    def route_after_scope(state: GraphState) -> str:
        category = state["scope_category"]
        if category == "meta":
            return "meta"
        if category == "out_of_scope":
            return "decline"

        chunks = state["chunks"]
        if not chunks or chunks[0]["distance"] > settings.LOW_CONFIDENCE_DISTANCE:
            return "low_confidence"
        return "generate"

    def generate_node(state: GraphState) -> dict:
        chunks = [RetrievedChunk(**c) for c in state["chunks"]]
        prompt = retrieval.build_prompt(chunks, state["standalone_question"])
        answer = model.invoke(prompt).content
        return {
            "answer": answer,
            "sources": [c.chunk_id for c in chunks],
            "source_labels": retrieval.build_source_labels(chunks),
            "is_scope": True,
            "history": [(state["question"], answer)],
        }

    def decline_node(state: GraphState) -> dict:
        return {
            "answer": guardrails.DECLINE_MESSAGE,
            "sources": [],
            "source_labels": [],
            "is_scope": False,
            "history": [(state["question"], guardrails.DECLINE_MESSAGE)],
        }

    def meta_node(state: GraphState) -> dict:
        return {
            "answer": guardrails.META_MESSAGE,
            "sources": [],
            "source_labels": [],
            "is_scope": False,
            "history": [(state["question"], guardrails.META_MESSAGE)],
        }

    def low_confidence_node(state: GraphState) -> dict:
        answer = f"I don't have enough information in {settings.LAW_NAME} to answer that."
        return {
            "answer": answer,
            "sources": [],
            "source_labels": [],
            "is_scope": True,
            "history": [(state["question"], answer)],
        }

    graph = StateGraph(GraphState)
    graph.add_node("condense", condense_node)
    graph.add_node("check_scope", check_scope_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("decline", decline_node)
    graph.add_node("meta", meta_node)
    graph.add_node("low_confidence", low_confidence_node)

    graph.add_edge(START, "condense")
    graph.add_edge("condense", "retrieve")
    graph.add_edge("retrieve", "check_scope")
    graph.add_conditional_edges("check_scope", route_after_scope, {
        "generate": "generate",
        "low_confidence": "low_confidence",
        "meta": "meta",
        "decline": "decline",
    })
    graph.add_edge("generate", END)
    graph.add_edge("decline", END)
    graph.add_edge("meta", END)
    graph.add_edge("low_confidence", END)

    # in-memory checkpointer: keeps conversation state per thread_id for the life of the
    # process. Good enough for this assignment's scope - swap for a SqliteSaver/
    # PostgresSaver if history needs to survive a restart
    return graph.compile(checkpointer=MemorySaver())


# thread_id groups turns into one conversation. None means a one-off/stateless call
# (evaluation, tests) - each gets its own fresh uuid so questions never leak context into
# each other; the live API passes a real session_id instead so follow-ups actually work
def answer_question(graph: CompiledStateGraph, request: QueryRequest, thread_id: str | None = None) -> ChatbotResponse:
    thread_id = thread_id or request.session_id or f"stateless-{uuid.uuid4()}"

    result = graph.invoke(
        {"question": request.question, "history": []},
        config={"configurable": {"thread_id": thread_id}},
    )

    return ChatbotResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        source_labels=result["source_labels"],
        is_scope=result["is_scope"],
    )

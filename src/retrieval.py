# Task 4 - retrieves relevant chunks for a question and builds a grounded prompt
from langchain_core.prompts import ChatPromptTemplate

import settings
from schemas import RetrievedChunk

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     f"You are a legal assistant that answers questions about {settings.LAW_NAME} only. "
     "Answer using ONLY the context below, nothing from outside it. "
     "If the context does not contain the answer, say explicitly that you don't have "
     "enough information, instead of guessing."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


# returns the top-k chunks most similar to the question, as validated RetrievedChunk objects
# (not raw LangChain Document/score tuples) so every caller downstream gets checked data
def retrieve(store, question: str, k: int = settings.RETRIEVAL_K) -> list[RetrievedChunk]:
    results = store.similarity_search_with_score(question, k=k)
    return [
        RetrievedChunk(
            chunk_id=doc.metadata["chunk_id"],
            text=doc.page_content,
            distance=score,
            section_label=doc.metadata.get("section_label") or None,
        )
        for doc, score in results
    ]


# joins retrieved chunks into one labeled block, so the LLM can point back at a specific chunk
def build_context(chunks: list[RetrievedChunk]) -> str:
    parts = [f"[Chunk {c.chunk_id}]\n{c.text}" for c in chunks]
    return "\n\n".join(parts)


# fills the prompt template with the retrieved context and the question, ready for the LLM
def build_prompt(chunks: list[RetrievedChunk], question: str):
    context = build_context(chunks)
    return PROMPT_TEMPLATE.invoke({"context": context, "question": question})


# builds the honest, human-readable citation for each chunk: chunk_id is the reliable part,
# the section number is a best-effort hint (see loader.extract_section_label) and is marked
# with "~" so the UI never implies it's a confirmed section reference
def build_source_labels(chunks: list[RetrievedChunk]) -> list[str]:
    return [
        f"Chunk {c.chunk_id} (~Sec. {c.section_label})" if c.section_label else f"Chunk {c.chunk_id}"
        for c in chunks
    ]

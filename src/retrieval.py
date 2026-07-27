# Task 4 - retrieves relevant chunks for a question and builds a grounded prompt
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

import settings

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     f"You are a legal assistant that answers questions about {settings.LAW_NAME} only. "
     "Answer using ONLY the context below, nothing from outside it. "
     "If the context does not contain the answer, say explicitly that you don't have "
     "enough information, instead of guessing."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


# returns the top-k chunks most similar to the question, each paired with its distance score
def retrieve(store, question: str, k: int = settings.RETRIEVAL_K) -> list[tuple[Document, float]]:
    return store.similarity_search_with_score(question, k=k)


# joins retrieved chunks into one labeled block, so the LLM can point back at a specific chunk
def build_context(results: list[tuple[Document, float]]) -> str:
    parts = [f"[Chunk {doc.metadata['chunk_id']}]\n{doc.page_content}" for doc, _ in results]
    return "\n\n".join(parts)


# fills the prompt template with the retrieved context and the question, ready for the LLM
def build_prompt(results: list[tuple[Document, float]], question: str):
    context = build_context(results)
    return PROMPT_TEMPLATE.invoke({"context": context, "question": question})

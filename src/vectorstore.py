# Task 3 - embeds chunks and stores/retrieves them from a local Chroma vector store
from langchain_chroma import Chroma
from langchain_core.documents import Document

import embeddings
import settings


# wraps each chunk as a LangChain Document, embeds it, and persists the collection to disk
def build_vectorstore(chunks: list[str]) -> Chroma:
    documents = [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings.get_embeddings(),
        collection_name=settings.COLLECTION_NAME,
        persist_directory=str(settings.CHROMA_DIR),
    )


# reopens the already-built collection instead of re-embedding every chunk again
def load_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings.get_embeddings(),
        persist_directory=str(settings.CHROMA_DIR),
    )

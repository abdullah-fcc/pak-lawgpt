# Task 3 - embeds chunks and stores/retrieves them from a local Chroma vector store
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document

import embeddings
import loader
import settings


# wraps each chunk as a LangChain Document, embeds it, and persists the collection to disk
# wipes any existing collection first, so re-running this doesn't pile up duplicate chunks
def build_vectorstore(chunks: list[str]) -> Chroma:
    if settings.CHROMA_DIR.exists():
        shutil.rmtree(settings.CHROMA_DIR)

    # Chroma metadata values can't be None, so a missing section label is stored as "" and
    # translated back to None when read (see retrieval.retrieve)
    documents = [
        Document(
            page_content=chunk,
            metadata={"chunk_id": i, "section_label": loader.extract_section_label(chunk) or ""},
        )
        for i, chunk in enumerate(chunks)
    ]
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

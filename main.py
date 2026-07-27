# orchestrates the RAG pipeline, task by task
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import loader
import settings
import utils
import vectorstore

logger = utils.setup_logger(__name__)


# Task 2 - extract, clean, chunk, and print evidence it all worked
def run_loading_and_chunking() -> list[str]:
    cleaned_text, chunks = loader.load_and_chunk()

    word_count = len(cleaned_text.split())
    logger.info(f"Extracted {len(cleaned_text)} characters, {word_count} words from {settings.PDF_PATH}")
    logger.info(f"First 500 characters:\n{cleaned_text[:500]}")

    logger.info(f"Split into {len(chunks)} chunks (chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})")
    for i, chunk in enumerate(chunks[:3]):
        logger.info(f"--- example chunk {i} ({len(chunk)} chars) ---\n{chunk}")

    return chunks


# Task 3 - embed all chunks into Chroma, then sanity-check retrieval with a sample query
def run_embedding_and_vectorstore(chunks: list[str]) -> None:
    logger.info(f"Embedding {len(chunks)} chunks with provider={settings.LLM_PROVIDER}...")
    store = vectorstore.build_vectorstore(chunks)
    logger.info(f"Vector store persisted to {settings.CHROMA_DIR}")

    sample_query = "What is consideration in a contract?"
    results = store.similarity_search_with_score(sample_query, k=3)

    logger.info(f"Sample query: {sample_query!r}")
    for rank, (doc, score) in enumerate(results, start=1):
        logger.info(f"--- result {rank} (distance={score:.4f}) ---\n{doc.page_content[:300]}")


def main() -> None:
    chunks = run_loading_and_chunking()
    run_embedding_and_vectorstore(chunks)


if __name__ == "__main__":
    main()

# orchestrates the RAG pipeline, task by task
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pydantic import ValidationError

import loader
import retrieval
import schemas
import settings
import utils
import vectorstore

logger = utils.setup_logger(__name__)

# Task 4 - questions with known answers (verified by hand against the PDF), used to
# confirm retrieval actually surfaces the section that contains the answer
KNOWN_ANSWER_QUESTIONS = [
    "What is consideration in a contract?",
    "Who is competent to contract?",
    "Is an agreement made without consideration void?",
]


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
def run_embedding_and_vectorstore(chunks: list[str]):
    logger.info(f"Embedding {len(chunks)} chunks with provider={settings.LLM_PROVIDER}...")
    store = vectorstore.build_vectorstore(chunks)
    logger.info(f"Vector store persisted to {settings.CHROMA_DIR}")

    sample_query = "What is consideration in a contract?"
    results = store.similarity_search_with_score(sample_query, k=3)

    logger.info(f"Sample query: {sample_query!r}")
    for rank, (doc, score) in enumerate(results, start=1):
        logger.info(f"--- result {rank} (distance={score:.4f}) ---\n{doc.page_content[:300]}")

    return store


# Task 4/5 - retrieve top-k chunks per question, build the grounded prompt, and print both
# so we can manually confirm the retrieved chunks actually contain the known answer.
# every boundary here passes a validated pydantic object (QueryRequest, RetrievedChunk),
# never a raw dict/tuple
def run_retrieval_pipeline(store) -> None:
    for question_text in KNOWN_ANSWER_QUESTIONS:
        request = schemas.QueryRequest(question=question_text)
        chunks = retrieval.retrieve(store, request.question)
        prompt = retrieval.build_prompt(chunks, request.question)

        logger.info(f"Question: {request.question!r}")
        for rank, chunk in enumerate(chunks, start=1):
            logger.info(f"  retrieved chunk {rank} (distance={chunk.distance:.4f}): {chunk.text[:200]}")
        logger.info(f"  prompt sent to LLM would be:\n{prompt.to_string()[:600]}")


# Task 5 - proves a validation error is caught gracefully instead of crashing the app
def demo_schema_validation() -> None:
    try:
        schemas.QueryRequest(question="")
    except ValidationError as e:
        logger.info(f"QueryRequest correctly rejected an empty question: {e.errors()[0]['msg']}")


def main() -> None:
    demo_schema_validation()
    chunks = run_loading_and_chunking()
    store = run_embedding_and_vectorstore(chunks)
    run_retrieval_pipeline(store)


if __name__ == "__main__":
    main()

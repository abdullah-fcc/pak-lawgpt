# orchestrates the RAG pipeline, task by task
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import loader
import settings
import utils

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


def main() -> None:
    run_loading_and_chunking()


if __name__ == "__main__":
    main()

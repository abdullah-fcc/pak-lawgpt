# all the constants used across the project, so nothing is hardcoded in the pipeline files
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# the law this whole chatbot is scoped to
LAW_NAME = "The Contract Act, 1872"
LAW_YEAR = 1872

# where things get saved/loaded from
DATA_DIR = Path("data")
PDF_PATH = DATA_DIR / "contract_act_1872.pdf"

# Task 2 - chunking
# 1000 chars is roughly enough to hold one full section + its illustrations without
# splitting mid-clause for most sections; 150 char overlap (~15%) keeps the boundary
# sentence from getting orphaned on the rare section that does get split
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Task 3 - embeddings & vector store
# which provider powers both embeddings and the LLM later, set in .env, defaults to openai
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "contract_act_1872"
RETRIEVAL_K = 3

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
# 5, not 3: short definitional queries ("what is an agreement") for the Act's most common
# defined terms sometimes rank the actual definition chunk 4th-5th rather than top-3, since
# the surrounding legalese doesn't repeat the query word as prominently as rarer terms do -
# k=5 gives those a real chance without pulling in truly irrelevant chunks
RETRIEVAL_K = 5

# Task 6 - LLM
OPENAI_CHAT_MODEL = "gpt-4o-mini"
GEMINI_CHAT_MODEL = "gemini-2.5-flash"
# cosine distance beyond which the top chunk isn't a real match - answer "I don't know"
# instead of asking the LLM to make something up from irrelevant context.
# Calibrated from real measurements: genuinely in-scope queries (even weakly-phrased ones
# like "what is an agreement") score 0.53-1.13; genuinely off-topic queries score 1.29+.
# 1.2 sits in that gap with margin on both sides.
LOW_CONFIDENCE_DISTANCE = 1.2

# Task 7 - scope guardrails: below this distance, retrieval found a strong enough match that
# the question is treated as in-scope regardless of what the LLM classifier says. This is the
# assignment's "combine both approaches" (retrieval-score gating + classification) - added
# after the classifier alone repeatedly misjudged questions about common-sounding but
# actually-defined Act terms (e.g. "what are voidable agreements") as too vague/off-topic.
# Same 0.53-1.13 vs 1.29+ gap as above, set conservatively below it so it only fires on
# genuinely confident matches, not on every borderline case
STRONG_MATCH_DISTANCE = 0.9

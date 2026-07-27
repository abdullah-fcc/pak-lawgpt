# pak-lawgpt

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about **The Contract
Act, 1872** — and only that Act. Built for the AI Summer Internship 2026 RAG assignment.

Given a question, it retrieves the most relevant sections of the Act from a vector store,
builds a prompt grounded in those sections, and asks an LLM to answer using only that
context — citing which section(s) it drew from. Questions outside the scope of the Contract
Act (general knowledge, other laws, casual chat) are declined rather than answered.

See `reports/report.md` for the full task-by-task write-up (chunking decisions, sanity
checks, eval results, sample outputs).

## Tech stack

- **LangChain + LangGraph** — retrieval and generation pipeline
- **OpenAI** (primary) / **Gemini** (secondary, for comparison) — LLM + embeddings, switchable
  via `.env`
- **Chroma** — local vector store
- **Pydantic** — request/response/data validation throughout the pipeline
- **FastAPI** — API layer

## Setup

Requires Python 3.13+.

```bash
python3.13 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install langchain==1.3.14 langgraph==1.2.9 langchain-openai==1.4.1 \
    langchain-google-genai==4.3.1 langchain-community==0.4.2 langchain-text-splitters==1.1.2 \
    langchain-chroma==1.1.0 chromadb==1.5.9 pypdf==6.14.2 fastapi==0.140.0 \
    "uvicorn[standard]==0.51.0" python-multipart==0.0.32 pydantic==2.13.4 python-dotenv==1.2.2
```

(These are the same packages pinned in `pyproject.toml`.)

Create a `.env` file in the project root (never committed, see `.gitignore`):

```
LLM_PROVIDER=openai        # or "gemini"
OPENAI_API_KEY=xxxxxxxxxxxxxxxx
GEMINI_API_KEY=xxxxxxxxxxxxxxxx
```

## Project structure

```
data/                    the source PDF (The Contract Act, 1872)
src/settings.py           all constants: paths, law metadata, chunking/model config
src/utils.py               shared helpers (logger)
src/loader.py               Task 2: PDF extraction, OCR cleanup, section-aware chunking
src/embeddings.py            Task 3: OpenAI/Gemini embedding model factory
src/vectorstore.py            Task 3: build/load the local Chroma vector store
chroma_db/                     persisted vector store (gitignored, rebuilt from data/)
main.py                         orchestrates the pipeline, task by task
reports/report.md                full task-by-task write-up
```

Run the pipeline so far:

```bash
python main.py
```

(Grows as later tasks are added — embeddings, vector store, retrieval, API, etc.)

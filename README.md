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
src/retrieval.py                Task 4: retrieve top-k chunks, build the grounded prompt
src/schemas.py                   Task 5: pydantic models used at every pipeline boundary
src/llm.py                        Task 6: OpenAI/Gemini chat model factory
src/pipeline.py                    Task 6/7: scope-check -> retrieve -> ground -> generate
src/guardrails.py                   Task 7: scope classification (in_scope/meta/out_of_scope)
chroma_db/                     persisted vector store (gitignored, rebuilt from data/)
main.py                         orchestrates the pipeline, task by task
evaluate.py                      Task 8: runs the 15-question eval set, reports pass rate
api/main.py                       Task 9: FastAPI app (POST /ask, GET /health, GET /)
api/static/                        Task 9: chat frontend (index.html, style.css, app.js)
reports/report.md                    full task-by-task write-up
```

Run the pipeline so far:

```bash
python main.py
```

Run the Task 8 evaluation set (after `main.py` has built `chroma_db/` at least once):

```bash
python evaluate.py
```

## Running the API

Once `main.py` has been run at least once (so `chroma_db/` exists):

```bash
uvicorn api.main:app --reload
```

Then open `http://127.0.0.1:8000/` for the chat frontend, or use the API directly:

- `POST /ask` — body: `{"question": "..."}` (a `QueryRequest`), returns a `ChatbotResponse`
  (`answer`, `sources` as chunk IDs, `is_scope`)
- `GET /health` — `{"status": "ok", "store_ready": true}`
- `GET /docs` — interactive Swagger UI, auto-generated from the pydantic models

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
    -d '{"question": "What is consideration in a contract?"}'
```

(Grows as later tasks are added — embeddings, vector store, retrieval, API, etc.)

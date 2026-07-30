# Task 9 - FastAPI service that exposes the RAG pipeline over HTTP, plus a small chat frontend
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import pipeline
import settings
import utils
import vectorstore
from schemas import ChatbotResponse, HealthResponse, QueryRequest

logger = utils.setup_logger(__name__)

app = FastAPI(title="pak-lawgpt", description=f"RAG chatbot for {settings.LAW_NAME}")

STATIC_DIR = Path(__file__).parent / "static"
FRONTEND_PATH = STATIC_DIR / "index.html"

# serves style.css/app.js under /static/... ; index.html itself is returned by the route
# below instead of through this mount, since it needs to live at "/" not "/static/"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# loaded/built once at import time (module-level), not on every request - reloading per
# request tanks latency, and rebuilding the graph would also throw away the in-memory
# checkpointer, breaking conversation memory between requests
STORE = None
GRAPH = None
STORE_READY = False
if settings.CHROMA_DIR.exists():
    try:
        STORE = vectorstore.load_vectorstore()
        GRAPH = pipeline.build_graph(STORE)
        STORE_READY = True
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
else:
    logger.error(f"{settings.CHROMA_DIR} not found - run main.py first to build it")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return FRONTEND_PATH.read_text()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok" if STORE_READY else "not_ready", store_ready=STORE_READY)


@app.post("/ask", response_model=ChatbotResponse)
def ask(request: QueryRequest) -> ChatbotResponse:
    if not STORE_READY:
        raise HTTPException(status_code=503, detail="Vector store not built yet, run main.py first")
    return pipeline.answer_question(GRAPH, request)

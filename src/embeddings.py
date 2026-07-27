# Task 3 - picks the embedding model based on LLM_PROVIDER, so switching provider is one env var
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

import settings


def get_embeddings() -> Embeddings:
    if settings.LLM_PROVIDER == "gemini":
        return GoogleGenerativeAIEmbeddings(model=settings.GEMINI_EMBEDDING_MODEL)
    return OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL)

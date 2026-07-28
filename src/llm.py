# Task 6 - picks the chat model based on LLM_PROVIDER, same switch as embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

import settings


def get_llm() -> BaseChatModel:
    if settings.LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(model=settings.GEMINI_CHAT_MODEL, temperature=0)
    return ChatOpenAI(model=settings.OPENAI_CHAT_MODEL, temperature=0)

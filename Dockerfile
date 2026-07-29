FROM python:3.13-slim

WORKDIR /app

# dependencies in their own layer, so this only re-runs when the package list actually changes
RUN pip install --no-cache-dir \
    langchain==1.3.14 langgraph==1.2.9 langchain-openai==1.4.1 \
    langchain-google-genai==4.3.1 langchain-community==0.4.2 langchain-text-splitters==1.1.2 \
    langchain-chroma==1.1.0 chromadb==1.5.9 pypdf==6.14.2 fastapi==0.140.0 \
    "uvicorn[standard]==0.51.0" python-multipart==0.0.32 pydantic==2.13.4 python-dotenv==1.2.2

# app code only - not data/ or chroma_db/, see .dockerignore: the vector store is a build
# artifact, not application code, so it's mounted as a volume at `docker run` time instead
# of baked into the image
COPY src/ src/
COPY api/ api/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

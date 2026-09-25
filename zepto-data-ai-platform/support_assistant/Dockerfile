# Zepto Support Assistant — FastAPI + LangGraph + ChromaDB + all-MiniLM-L6-v2
# Build:  docker build -t zepto-support-assistant .
# Run:    docker run --rm -p 7860:7860 zepto-support-assistant        (MOCK_LLM defaults to 1)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MOCK_LLM=1 \
    HF_HOME=/app/.cache/huggingface

# Hugging Face Spaces runs containers as uid 1000; creating that user keeps the image portable.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# CPU-only torch first (~200 MB instead of the multi-GB CUDA build), then the app dependencies.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the container starts without network access.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=appuser:appuser . .
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

# syntax=docker/dockerfile:1.7

FROM node:18-alpine AS node-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.10-slim AS final
WORKDIR /app

RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    HF_HOME=/home/user/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user rag_backend.py .
COPY --chown=user api/ ./api/
COPY --chown=user --from=node-builder /frontend/dist ./static/

RUN mkdir -p /app/logs && chown user:user /app/logs

USER user

RUN mkdir -p /home/user/.cache/huggingface

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
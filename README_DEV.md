# OmniDoc RAG Development

## Option A - Run services separately

Use this workflow for active development.

```bash
# Terminal 1: Start FastAPI backend
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start React frontend
cd frontend
npm install
npm run dev
```

The app is available at:

```text
http://localhost:5173
```

The Vite dev server proxies `/upload`, `/chat`, and `/health` to:

```text
http://localhost:8000
```

## Option B - Run via Docker Compose

Use this workflow to test the production build locally before pushing to Hugging Face Spaces.

```bash
docker-compose up --build
```

The app is available at:

```text
http://localhost:7860
```

This uses the same Docker image shape that runs on Hugging Face Spaces: FastAPI serves the API and the built React app from one process on port `7860`.

## Environment

Create a root `.env` file for local runs and Docker Compose:

```env
GROQ_API_KEY=your_groq_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX=omnidoc-rag
HF_TOKEN=optional_huggingface_token
```

Note: the current RAG backend uses the hard-coded Pinecone index name `omnidoc-rag`.

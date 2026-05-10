import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from rag_backend import get_context_and_answer, process_documents


LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, format=LOG_FORMAT, enqueue=True)
logger.add(
    LOG_DIR / "app.log",
    format=LOG_FORMAT,
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    enqueue=True,
)
logging.getLogger("uvicorn.access").disabled = True

app = FastAPI(title="OmniDoc RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    active_files: List[str] = Field(default_factory=list)


@app.on_event("startup")
async def startup_event():
    logger.info("OmniDoc RAG API started — ready to serve")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "{} {} from {} failed after {:.2f}ms",
            request.method,
            request.url.path,
            client_ip,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "{} {} from {} completed in {:.2f}ms",
        request.method,
        request.url.path,
        client_ip,
        elapsed_ms,
    )
    return response


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    allowed_extensions = {".pdf", ".docx", ".txt"}
    filenames = [file.filename or "unnamed" for file in files]
    logger.info("Upload received: {}", filenames)

    with tempfile.TemporaryDirectory(prefix="omnidoc_upload_") as temp_dir:
        temp_paths = []
        try:
            for file in files:
                original_name = os.path.basename(file.filename or "upload")
                suffix = Path(original_name).suffix.lower()

                if suffix not in allowed_extensions:
                    logger.error("Unsupported upload type for {}", original_name)
                    return JSONResponse(
                        status_code=500,
                        content={"error": f"Unsupported file type: {suffix}", "filename": original_name},
                    )

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as buffer:
                    shutil.copyfileobj(file.file, buffer)
                    destination = Path(buffer.name)

                temp_paths.append((str(destination), original_name))
                logger.info("Saved upload {} to {}", original_name, destination)

            logger.info("Processing start for {}", filenames)
            processed = process_documents(temp_paths)
            missing = [name for _, name in temp_paths if name not in processed]

            if missing:
                logger.error("Processing failed or skipped for {}", missing[0])
                return JSONResponse(
                    status_code=500,
                    content={"error": "Document could not be processed", "filename": missing[0]},
                )

            logger.info("Processing end for {}; processed {}", filenames, processed)
            return {"processed": processed, "count": len(processed)}
        except HTTPException:
            raise
        except Exception as exc:
            failed_name = filenames[0] if filenames else "unknown"
            logger.exception("Upload failed for {}", failed_name)
            return JSONResponse(status_code=500, content={"error": str(exc), "filename": failed_name})
        finally:
            logger.info("Temporary upload directory cleaned: {}", temp_dir)


def sse_event(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/chat")
async def chat(payload: ChatRequest):
    query_preview = payload.query[:100]
    logger.info(
        "Chat request received: query='{}', active_files={}",
        query_preview,
        len(payload.active_files),
    )

    async def stream():
        token_count = 0
        try:
            retrieved_docs, response_stream = get_context_and_answer(payload.query, payload.active_files)
            logger.info("Chat retrieved {} documents", len(retrieved_docs))
            logger.info("Chat stream start")

            for chunk in response_stream:
                if chunk is None:
                    continue
                token = str(chunk)
                token_count += 1
                yield sse_event({"token": token, "done": False})

            sources = [
                {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source_name", "Unknown"),
                }
                for doc in retrieved_docs
            ]
            logger.info("Chat stream end; total tokens streamed={}", token_count)
            yield sse_event({"sources": sources, "done": True})
        except Exception as exc:
            logger.exception("Chat stream failed")
            yield sse_event({"error": str(exc), "done": True})

    return StreamingResponse(stream(), media_type="text/event-stream")


static_path = BASE_DIR / "static"
assets_path = static_path / "assets"

if static_path.exists():
    if assets_path.exists():
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @app.get("/")
    async def serve_spa():
        return FileResponse(static_path / "index.html")

    @app.get("/{full_path:path}")
    async def serve_spa_routes(full_path: str):
        if not full_path.startswith(("upload", "chat", "health", "assets")):
            return FileResponse(static_path / "index.html")
        raise HTTPException(status_code=404, detail="Not found")

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

import src.chat.model  # noqa: F401

# Import all models so SQLAlchemy registers them with Base.metadata
import src.rag.model  # noqa: F401
from src.chat import chat_router
from src.core import Base, engine
from src.knowledge_base import kb_router
from src.rag import get_vector_store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all database tables (including parent_chunks, ingestion_tasks)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Initialize Qdrant collections (idempotent — skips if exists)
    from src.chat import get_session_vector_store
    vector_store = get_vector_store()
    session_vector_store = get_session_vector_store()
    try:
        await vector_store.ensure_collection()
        await session_vector_store.ensure_collection()
        logger.info("Qdrant collections initialized")
    except Exception as e:
        # Qdrant may not be running; log warning but don't block startup
        logger.warning(
            "Could not initialize Qdrant collection: %s. "
            "RAG features will be unavailable until Qdrant is running.",
            str(e),
        )

    yield

app = FastAPI(
    title="Chat Application with PDF Knowledge Base",
    description="An intelligent chat system that leverages Large Language Models (LLMs) and PDF document management to provide context-aware responses. Users can upload PDF documents which are processed and indexed for semantic search, enabling the LLM to reference relevant content in its responses.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "Support Team",
        "url": "https://example.com/support",
    },
    license_info={
        "name": "MIT",
    },
)

app.include_router(chat_router)
app.include_router(kb_router)

if __name__ == "__main__":
    # Passing the app as an import string ("main:app") enables hot-reloading
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
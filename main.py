import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from src.core.database import engine, Base
from src.chat import chat_router
from src.knowledge_base import kb_router
from src.rag.dependency import get_vector_store

# Import RAG models so SQLAlchemy registers them with Base.metadata
import src.rag.model  # noqa: F401

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all database tables (including parent_chunks, ingestion_tasks)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Initialize Qdrant collection (idempotent — skips if exists)
    vector_store = get_vector_store()
    try:
        await vector_store.ensure_collection()
        logger.info("Qdrant collection initialized")
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
"""Main application entrypoint."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI

from app.shared.config import get_app_settings
from app.shared.logging import setup_logging
from app.shared.middleware import CorrelationIdMiddleware
from app.shared.db import engine, Base

# Import all models so metadata knows about them
import app.kb.domain.models  # noqa
import app.chat.domain.models  # noqa

from app.kb.api import router as kb_router
from app.chat.api import router as chat_router

from app.kb.config import get_qdrant_settings
from app.kb.infra.qdrant_store import QdrantStore

setup_logging()
logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    logger.info("application_startup", status="started")
    
    # Initialize DB schema
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Initialize Qdrant Collection
    kb_config = get_qdrant_settings()
    qdrant_store = QdrantStore(
        host=kb_config.host,
        port=kb_config.port,
        collection_name=kb_config.collection_name,
    )
    await qdrant_store.ensure_collection()
    await qdrant_store.close()

    yield
    logger.info("application_shutdown", status="stopped")

app_settings = get_app_settings()

fastapi_app = FastAPI(
    title=app_settings.title,
    description="Refactored RAG Chatbot using DDD architecture.",
    lifespan=lifespan,
    version=app_settings.version,
)

# Add Middleware
fastapi_app.add_middleware(CorrelationIdMiddleware)

from app.frontend import router as frontend_router

# Include API routes
fastapi_app.include_router(kb_router)
fastapi_app.include_router(chat_router)
fastapi_app.include_router(frontend_router)

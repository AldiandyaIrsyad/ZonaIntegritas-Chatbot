"""Main application entrypoint."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.knowledge_base.api import router as kb_api_router
from app.knowledge_base.presentation import router as kb_presentation_router
from app.chat.api import router as chat_api_router
from app.chat.presentation import router as chat_presentation_router
from app.core.config import get_app_settings, get_qdrant_settings
from app.core.logging import setup_logging

setup_logging()
logger = structlog.get_logger(__name__)

from app.infra.db import engine, Base
import app.knowledge_base.model
import app.chat.model
import app.rag.model
from app.infra.vector_store import QdrantStore

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events.
    
    Args:
        app (FastAPI): The FastAPI application instance.
        
    Yields:
        None
    """
    logger.info("application_startup", status="started")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    qdrant_settings = get_qdrant_settings()
    qdrant_store = QdrantStore(
        host=qdrant_settings.host,
        port=qdrant_settings.port,
        collection_name=qdrant_settings.collection_name,
    )
    await qdrant_store.ensure_collection()
    yield
    logger.info("application_shutdown", status="stopped")


app_settings = get_app_settings()
app = FastAPI(
    title=app_settings.title,
    description="FastAPI chatbot service with structured logging and Jinja2 templates.",
    lifespan=lifespan,
    version=app_settings.version,
)

# Include API routes
app.include_router(router)
app.include_router(kb_api_router)
app.include_router(kb_presentation_router)
app.include_router(chat_api_router, prefix="/api")
app.include_router(chat_presentation_router)

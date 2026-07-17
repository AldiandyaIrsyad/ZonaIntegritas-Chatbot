"""Main application entrypoint."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from sqlalchemy import text

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
from app.kb.dependency import get_vector_store, get_document_parser, get_reranker, get_text_embedder

setup_logging()
logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    logger.info("application_startup", status="started")
    
    # Initialize DB schema
    async with engine.begin() as conn:
        # Create ltree extension on Postgres (no-op on SQLite)
        if engine.dialect.name == "postgresql":
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS ltree"))

        await conn.run_sync(Base.metadata.create_all)

        # Guarded ALTER TABLE: add new ltree columns to existing parent_chunks table
        # (create_all won't alter existing tables)
        if engine.dialect.name == "postgresql":
            result = await conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'parent_chunks'"
            ))
            existing_cols = {row[0] for row in result}
            new_cols = {
                "parent_id": "VARCHAR REFERENCES parent_chunks(id) ON DELETE SET NULL",
                "ordinal": "INTEGER DEFAULT 0",
                "path": "VARCHAR DEFAULT ''",
                "depth": "INTEGER DEFAULT 0",
            }
            for col_name, col_def in new_cols.items():
                if col_name not in existing_cols:
                    await conn.execute(text(
                        f"ALTER TABLE parent_chunks ADD COLUMN {col_name} {col_def}"
                    ))
                    logger.info("db.alter_table", table="parent_chunks", column=col_name)

            # Guarded ALTER TABLE: add RAG context/sources columns to existing
            # messages table. JSONB here matches how SQLAlchemy's generic
            # JSON model column type maps under the Postgres dialect, so
            # fresh-DB create_all and this guarded ALTER produce the same
            # schema.
            result = await conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'messages'"
            ))
            existing_msg_cols = {row[0] for row in result}
            new_msg_cols = {
                "context": "TEXT",
                "sources": "JSONB DEFAULT NULL",
            }
            for col_name, col_def in new_msg_cols.items():
                if col_name not in existing_msg_cols:
                    await conn.execute(text(
                        f"ALTER TABLE messages ADD COLUMN {col_name} {col_def}"
                    ))
                    logger.info("db.alter_table", table="messages", column=col_name)

    # Initialize Qdrant collection (main retrieval)
    kb_config = get_qdrant_settings()
    qdrant_store = QdrantStore(
        host=kb_config.host,
        port=kb_config.port,
        collection_name=kb_config.collection_name,
    )
    await qdrant_store.ensure_collection()
    await qdrant_store.close()

    yield

    # Close the process-lifetime singleton clients from app.kb.dependency
    # (each wraps an httpx.AsyncClient or in-process model that would
    # otherwise leak). Only close getters that were actually called at least
    # once during this app's lifetime (cache_info().currsize > 0) — calling
    # an unused getter here would instantiate it for the first time just to
    # immediately close it, which for get_text_embedder() means loading the
    # entire BGE-M3 model on the way out. Independent try/except per client
    # so one failure doesn't block the others from closing.
    for getter in (get_document_parser, get_reranker, get_text_embedder, get_vector_store):
        if getter.cache_info().currsize == 0:
            continue
        closeable = getter()
        if closeable is None:
            continue
        try:
            await closeable.close()
        except Exception as exc:
            logger.warning("shutdown.close_failed", target=type(closeable).__name__, error=str(exc))

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

"""Database connection and session management.

Provides the async SQLAlchemy engine and session factory for PostgreSQL.
This is **shared infrastructure** used by both bounded contexts:
    - ``app/chat/infra/postgres_chat_repo.py`` (chat persistence)
    - ``app/kb/infra/postgres_repo.py``      (KB document/chunk persistence)

Both contexts import :class:`Base` as their ORM declarative base, so all
SQLAlchemy models share one ``metadata`` object and one engine.
"""

from typing import AsyncGenerator
import structlog
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base, DeclarativeBase

from app.shared.config import get_db_settings

logger = structlog.get_logger(__name__)


class Base(DeclarativeBase):
    """Shared SQLAlchemy declarative base for all ORM models.

    Both ``app/chat/domain/models.py`` and ``app/kb/domain/models.py``
    subclass this so ``Base.metadata`` collects every table and
    ``init_db()`` can create them in one pass.
    """
    pass

from sqlalchemy.ext.asyncio import AsyncEngine


def get_engine() -> AsyncEngine:
    """Create and return the SQLAlchemy async engine.

    ``pool_pre_ping=True`` emits a lightweight ``SELECT 1`` before each
    checkout so connections dropped by the server (idle timeout, restart)
    are replaced instead of raising ``OperationalError`` mid-request.
    """
    settings = get_db_settings()
    engine = create_async_engine(
        settings.async_database_url,
        echo=False,
        future=True,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
    return engine


# Module-level singleton engine — created once at import time and reused
# for the lifetime of the process. Closed in ``app/main.py``'s lifespan.
engine = get_engine()

# Global session maker tied to the engine. ``expire_on_commit=False`` keeps
# attributes accessible after ``commit()`` (needed because the chat pipeline
# commits mid-request and then reads ORM objects). ``autoflush=False`` avoids
# implicit flushes triggering before each query.
async_session_maker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency provider for an :class:`AsyncSession`.

    Used by both chat and kb ``dependency.py`` composition roots to inject
    a session into their respective Postgres repositories. Commits on
    success, rolls back and re-raises on exception, always closes.

    Yields:
        AsyncSession: The database session for the request.
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            logger.error("database.session.error", error=str(exc))
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize the database by creating all defined tables.

    Idempotent: ``create_all`` only creates tables that don't already exist.
    Called during application startup in ``app/main.py``'s lifespan.
    """
    logger.info("Initializing database tables...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully.")
    except Exception as exc:
        logger.error("database.initialization.failed", error=str(exc))
        raise

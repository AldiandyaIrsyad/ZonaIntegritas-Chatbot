"""Database connection and session management.

Provides the async SQLAlchemy engine and session factory for PostgreSQL.
"""

from typing import AsyncGenerator
import structlog
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

from app.core.config import get_db_settings

logger = structlog.get_logger(__name__)

from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

def get_engine():
    """Create and return the SQLAlchemy async engine."""
    settings = get_db_settings()
    # Setting pool_pre_ping to True to gracefully handle disconnected connections
    engine = create_async_engine(
        settings.async_database_url,
        echo=False,
        future=True,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
    return engine

engine = get_engine()

# Global session maker tied to the engine
async_session_maker = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency provider for database sessions.
    
    Yields:
        AsyncSession: The database session.
    """
    async with async_session_maker() as session:
        try:
            yield session
        except Exception as exc:
            logger.error("database.session.error", error=str(exc))
            await session.rollback()
            raise
        finally:
            await session.close()

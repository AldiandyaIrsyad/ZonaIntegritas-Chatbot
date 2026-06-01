"""
Database connection and session management.

Provides the async SQLAlchemy engine and session dependency.
"""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from .config import get_db_settings

db_settings = get_db_settings()
engine = create_async_engine(db_settings.database_url, echo=False, pool_pre_ping=True)
async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()

async def get_db():
    """FastAPI dependency that provides an async database session.
    
    Yields:
        AsyncSession: The database session.
    """
    async with async_session() as session:
        yield session
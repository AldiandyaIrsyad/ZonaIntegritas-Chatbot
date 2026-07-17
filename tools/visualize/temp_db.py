"""Temporary SQLite database factory for visualization runs.

Creates a file-based SQLite database that reuses the production
``Base.metadata`` (from :mod:`app.shared.db`) so that the real
``ParentChunk`` / ``PDFDocument`` / ``IngestionTask`` ORM models can be
persisted and inspected with a DB browser after the run.

The SQLite file is written to ``viz_output/`` and named
``viz_<timestamp>.sqlite``.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.shared.db import Base

# Import all models so Base.metadata discovers them (same as app/main.py lifespan).
import app.kb.domain.models  # noqa: F401
import app.chat.domain.models  # noqa: F401


def create_temp_engine(sqlite_path: str) -> AsyncEngine:
    """Create an async SQLite engine for a file-based database.

    Args:
        sqlite_path: Absolute path to the ``.sqlite`` file.

    Returns:
        An :class:`AsyncEngine` backed by ``aiosqlite``.
    """
    return create_async_engine(
        f"sqlite+aiosqlite:///{sqlite_path}",
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
    )


async def init_temp_db(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create all tables and return a session factory.

    Args:
        engine: The temp SQLite engine.

    Returns:
        An ``async_sessionmaker`` bound to the temp engine.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )

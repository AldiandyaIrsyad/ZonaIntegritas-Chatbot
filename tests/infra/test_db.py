"""Tests for database infrastructure module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.infra.db import get_engine, get_db_session
from app.core.config import DatabaseSettings

@patch("app.infra.db.get_db_settings")
@patch("app.infra.db.create_async_engine")
def test_get_engine(mock_create_engine, mock_get_settings):
    """Test engine creation with settings."""
    mock_settings = MagicMock()
    mock_settings.async_database_url = "postgresql+asyncpg://test_user:test_password@localhost:5432/test_db"
    mock_get_settings.return_value = mock_settings
    
    engine = get_engine()
    
    mock_create_engine.assert_called_once()
    args, kwargs = mock_create_engine.call_args
    assert "postgresql+asyncpg://test_user:test_password@localhost:5432/test_db" in args[0]
    assert kwargs.get("pool_pre_ping") is True

@patch("app.infra.db.async_session_maker")
async def test_get_db_session_success(mock_session_maker):
    """Test successful database session yield."""
    mock_session = AsyncMock()
    # mock_session_maker returns a context manager that yields mock_session
    mock_session_maker.return_value.__aenter__.return_value = mock_session
    
    generator = get_db_session()
    session = await generator.__anext__()
    
    assert session is mock_session
    
    # Finish the generator
    with pytest.raises(StopAsyncIteration):
        await generator.__anext__()
        
    mock_session.close.assert_awaited_once()

@patch("app.infra.db.async_session_maker")
async def test_get_db_session_exception(mock_session_maker):
    """Test session rollback and exception propagation."""
    mock_session = AsyncMock()
    mock_session_maker.return_value.__aenter__.return_value = mock_session
    
    generator = get_db_session()
    session = await generator.__anext__()
    
    # Inject exception into the generator
    with pytest.raises(ValueError):
        await generator.athrow(ValueError("Test error"))
        
    mock_session.rollback.assert_awaited_once()
    mock_session.close.assert_awaited_once()

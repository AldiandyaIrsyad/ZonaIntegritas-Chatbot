import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from src.core.config import LLMSettings, DatabaseSettings, StorageSettings

def test_llm_settings_defaults():
    settings = LLMSettings()
    assert settings.base_url == "https://openrouter.ai/api/v1"
    assert settings.model == "google/gemini-2.5-flash"

def test_database_settings():
    settings = DatabaseSettings(_env_file=None, password="test_password")
    assert "postgresql+asyncpg://postgres:test_password" in settings.database_url

def test_storage_settings():
    settings = StorageSettings()
    assert settings.upload_dir == "user_upload"

@pytest.mark.asyncio
async def test_get_db_yields_session():
    # Import here to avoid env loading issues during collection if env is missing
    from src.core.database import get_db
    
    dummy_session = AsyncMock()
    mock_async_session = MagicMock()
    mock_async_session.return_value.__aenter__.return_value = dummy_session
    mock_async_session.return_value.__aexit__.return_value = None

    with patch("src.core.database.async_session", mock_async_session):
        async for session in get_db():
            assert session == dummy_session

"""Tests for LLM connection infrastructure module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from openai import APIConnectionError, APIError

from app.infra.llm_connection import LLMConnection

@pytest.fixture
def mock_openai_client():
    """Fixture providing a mock AsyncOpenAI client."""
    with patch("app.infra.llm_connection.AsyncOpenAI") as mock_client_class:
        mock_instance = AsyncMock()
        mock_client_class.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def connection(mock_openai_client):
    """Fixture providing an LLMConnection instance."""
    return LLMConnection(
        base_url="http://localhost:11434/v1",
        api_key=None,
        default_headers={"X-Test": "1"}
    )

async def test_stream_chat_success(connection, mock_openai_client):
    """Test successful streaming of chat completion."""
    # Setup mock stream
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta.content = "Hello "
    
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta.content = "world!"
    
    # Async generator for the stream
    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2
        
    mock_openai_client.chat.completions.create.return_value = mock_stream()
    
    messages = [{"role": "user", "content": "Hi"}]
    chunks = []
    
    async for chunk in connection.stream_chat("test-model", messages, max_tokens=10):
        chunks.append(chunk)
        
    assert chunks == ["Hello ", "world!"]
    
    mock_openai_client.chat.completions.create.assert_awaited_once_with(
        model="test-model",
        messages=messages,
        max_tokens=10,
        stream=True
    )

async def test_stream_chat_empty_chunks(connection, mock_openai_client):
    """Test that empty delta contents are skipped."""
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta.content = "Test"
    
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta.content = ""  # Empty content
    
    mock_chunk3 = MagicMock()
    mock_chunk3.choices = []  # No choices
    
    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2
        yield mock_chunk3
        
    mock_openai_client.chat.completions.create.return_value = mock_stream()
    
    chunks = []
    async for chunk in connection.stream_chat("test-model", [], 10):
        chunks.append(chunk)
        
    assert chunks == ["Test"]

async def test_stream_chat_connection_error(connection, mock_openai_client):
    """Test handling of APIConnectionError."""
    # The `create` method itself might raise before returning the stream
    mock_openai_client.chat.completions.create.side_effect = APIConnectionError(
        request=MagicMock()
    )
    
    with pytest.raises(APIConnectionError):
        # We need to iterate or start the async generator to trigger the error
        stream = connection.stream_chat("test-model", [], 10)
        await stream.__anext__()

async def test_stream_chat_api_error(connection, mock_openai_client):
    """Test handling of APIError."""
    error = APIError(
        request=MagicMock(),
        message="API error",
        body={}
    )
    error.status_code = 500
    mock_openai_client.chat.completions.create.side_effect = error
    
    with pytest.raises(APIError):
        stream = connection.stream_chat("test-model", [], 10)
        await stream.__anext__()

async def test_close(connection, mock_openai_client):
    """Test closing the client."""
    await connection.close()
    mock_openai_client.close.assert_awaited_once()

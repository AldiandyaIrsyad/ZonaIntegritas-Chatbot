import pytest
from unittest.mock import AsyncMock, patch
from typing import AsyncGenerator

from src.infra.llm_provider import LLM 


# --- Mock Classes for OpenAI Stream Simulation ---

class MockDelta:
    def __init__(self, content: str):
        self.content = content

class MockChoice:
    def __init__(self, content: str):
        self.delta = MockDelta(content)

class MockChunk:
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]

async def mock_stream_generator() -> AsyncGenerator[MockChunk, None]:
    """Simulates the AsyncGenerator returned by openai.AsyncStream"""
    yield MockChunk("Mocked ")
    yield MockChunk("API ")
    yield MockChunk("Response")

# --- Test Suite ---

@pytest.mark.asyncio
async def test_api_backend_initialization_and_stream():
    """
    Validates OpenRouter backend routing, header injection, and stream parsing.
    Mocked to prevent external network calls and API key exposure.
    """
    with patch("LLMs.LLM.AsyncOpenAI") as MockOpenAI:
        mock_client_instance = MockOpenAI.return_value
        mock_client_instance.chat.completions.create = AsyncMock()
        mock_client_instance.chat.completions.create.return_value = mock_stream_generator()


        # Initialize the API backend
        llm = LLM(
            backend="api", 
            model="openai/gpt-3.5-turbo", 
            api_key="test_dummy_key"
        )

        # Verify correct initialization parameters
        MockOpenAI.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1",
            api_key="test_dummy_key",
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Local-Dev-App"
            }
        )

        # Execute stream and collect chunks
        result_text = ""
        async for chunk in llm.input("Test prompt"):
            result_text += chunk

        # Assert generation logic
        assert result_text == "Mocked API Response"
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test prompt"}],
            stream=True
        )

@pytest.mark.asyncio
async def test_local_backend_live_integration():
    """
    Live integration test against the local Ollama instance.
    Assumes `ollama serve` is running on http://localhost:11434.
    """
    target_model = "qwen2.5:0.5b" 
    
    try:
        llm = LLM(backend="local", model=target_model)
        
        chunks_received = 0
        final_output = ""
        
        # Test stream connectivity
        async for chunk in llm.input("Acknowledge this message with a single word."):
            assert isinstance(chunk, str)
            final_output += chunk
            chunks_received += 1
            
            # Short-circuit the stream to minimize test execution time
            if chunks_received >= 3:
                break
                
        assert chunks_received > 0
        assert len(final_output) > 0

    except Exception as e:
        pytest.fail(f"Local backend integration failed. Ensure Ollama is running and '{target_model}' is pulled. Error: {e}")

def test_invalid_backend_rejection():
    """Validates the constructor guard clauses."""
    with pytest.raises(ValueError, match="Invalid backend specified"):
        LLM(backend="unsupported", model="test-model")
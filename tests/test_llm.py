import pytest
from app.llm.service import LLMService

class MockLLMConnection:
    async def stream_chat(self, model: str, messages: list[dict[str, str]], max_tokens: int):
        yield "Hello"
        yield " World"

    async def close(self):
        pass

@pytest.mark.asyncio
async def test_llm_service():
    connection = MockLLMConnection()
    service = LLMService(
        connection=connection,
        model="gpt-3.5-turbo",
        max_tokens=1000,
        max_completion_tokens=200
    )
    
    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"}
    ]
    
    chunks = []
    async for chunk in service.stream_response(history):
        chunks.append(chunk)
        
    assert "".join(chunks) == "Hello World"
    assert len(service.last_context_payload) == 2

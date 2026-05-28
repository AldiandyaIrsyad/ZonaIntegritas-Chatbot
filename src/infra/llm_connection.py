from typing import Optional

from openai import AsyncOpenAI
from pydantic import SecretStr


class LLMConnection:
    """
    Handles the low-level infrastructure connection to the LLM Gateway (OpenRouter)
    or local containers (Ollama).
    """
    def __init__(
        self, 
        base_url: Optional[str] = None,
        api_key: Optional[SecretStr] = None,
        default_headers: Optional[dict] = None,
    ):
        resolved_key = api_key.get_secret_value() if api_key else "ollama-dummy-token"
        
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=resolved_key,
            default_headers=default_headers
        )


    async def stream_chat(self, model: str, messages: list, max_tokens: int):
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
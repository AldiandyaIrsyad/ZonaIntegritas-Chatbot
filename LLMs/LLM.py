import os
from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI

class LLM:
    """
    Unified LLM Gateway interfacing with OpenRouter (API) 
    and OpenAI-compliant local containers (Ollama).
    """
    def __init__(
        self, 
        backend: str, 
        model: str, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None
    ):
        self.backend = backend.lower()
        self.model = model

        if self.backend == "api":
            # OpenRouter configuration
            self.client = AsyncOpenAI(
                base_url=base_url or "https://openrouter.ai/api/v1",
                api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
                default_headers={
                    "HTTP-Referer": "http://localhost:3000", # Required by OpenRouter
                    "X-Title": "Local-Dev-App"
                }
            )
        elif self.backend == "local":
            self.client = AsyncOpenAI(
                base_url=base_url or "http://localhost:11434/v1",
                api_key="ollama-dummy-token"
            )
        else:
            raise ValueError("Invalid backend specified; use 'api' or 'local'.")

    async def input(self, text: str) -> AsyncGenerator[str, None]:
        """Streams raw text chunks from the activated backend wrapper."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
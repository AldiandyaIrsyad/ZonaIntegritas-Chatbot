from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI
from pydantic import SecretStr

class LLM:
    """
    Unified LLM Gateway interfacing with OpenRouter (API) 
    and OpenAI-compliant local containers (Ollama).
    """
    def __init__(
        self, 
        backend: str, 
        model: str, 
        api_key: Optional[SecretStr] = None, 
        base_url: Optional[str] = None
    ):
        self.backend = backend.lower()
        self.model = model

        resolved_key = api_key.get_secret_value() if api_key else "ollama-dummy-token"
        

        headers = {
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Local-Dev-App"
        } if self.backend == "api" else {}

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=resolved_key,
            default_headers=headers
        )

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
import tiktoken
from typing import AsyncGenerator, Optional, List, Dict
from openai import AsyncOpenAI
from pydantic import SecretStr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, trim_messages

class LLM:
    """
    Unified LLM Gateway interfacing with OpenRouter (API) 
    and OpenAI-compliant local containers (Ollama).
    Includes internal token management and context window truncation.
    """
    def __init__(
        self, 
        model: str, 
        api_key: Optional[SecretStr] = None, 
        base_url: Optional[str] = None
    ):
        self.model = model
        resolved_key = api_key.get_secret_value() if api_key else "ollama-dummy-token"
        
        headers = {
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Local-Dev-App"
        }

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=resolved_key,
            default_headers=headers
        )

    def _count_tokens(self, messages: List[BaseMessage]) -> int:
        """Calculates total token utilization using the appropriate BPE encoding."""
        try:
            encoder = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoder = tiktoken.get_encoding("cl100k_base")
            
        total = 0
        for msg in messages:
            # Padding accounts for OpenAI's IM_START/IM_END boundary tokens
            total += len(encoder.encode(msg.content)) + 4 
        return total

    def _prune_context(self, raw_history: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Converts raw dicts to LangChain primitives, applies token limits, and reverts format."""
        lc_messages = []
        for msg in raw_history:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))

        trimmed_lc_messages = trim_messages(
            lc_messages,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=self._count_tokens,
            include_system=True, 
            allow_partial=True,
            # start_on="human" # Optional: Ensure we always start with the most recent user message if possible
        )

        final_payload = []
        for msg in trimmed_lc_messages:
            if isinstance(msg, HumanMessage):
                final_payload.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                final_payload.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                final_payload.append({"role": "system", "content": msg.content})
                
        return final_payload

    async def input(
            self, 
            raw_history: List[Dict[str, str]], 
            max_tokens: int = 4000,
            max_completion_tokens: int = 1000
            ) -> AsyncGenerator[str, None]:
        context_payload = self._prune_context(raw_history, max_tokens)
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=context_payload,
            max_completion_tokens=max_completion_tokens,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
import tiktoken
from typing import AsyncGenerator, List, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, trim_messages

from src.infra.llm_connection import LLMConnection

class LLMService:
    """
    Application Service handling domain-level LLM logic: 
    internal token management, context window truncation, and generation wrapper.
    """
    def __init__(
        self, 
        connection: LLMConnection,
        model: str, 
        max_tokens: int,
        max_completion_tokens: int,
    ):
        self.connection = connection
        self.model = model
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens

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
        system_msgs = [m for m in raw_history if m["role"] == "system"]
        chat_msgs = [m for m in raw_history if m["role"] != "system"]
        
        system_tokens = self._count_tokens(system_msgs)
        available_tokens = max_tokens - system_tokens
        
        retained_chat_msgs = []
        current_tokens = 0
        
        for msg in reversed(chat_msgs):
            msg_tokens = self._count_tokens([msg])
            if current_tokens + msg_tokens > available_tokens:
                break
            retained_chat_msgs.insert(0, msg)
            current_tokens += msg_tokens
            
        return system_msgs + retained_chat_msgs


    async def stream_response(
            self, 
            raw_history: List[Dict[str, str]]
            ) -> AsyncGenerator[str, None]:
        context_payload = self._prune_context(raw_history, self.max_tokens)

        async for chunk in self.connection.stream_chat(self.model, context_payload, self.max_completion_tokens):
            yield chunk

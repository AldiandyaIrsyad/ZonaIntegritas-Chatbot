import tiktoken
import copy
from typing import AsyncGenerator, List, Dict

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
        
        try:
            self.encoder = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            total += 3 
            for key, value in msg.items():
                total += len(self.encoder.encode(str(value)))
        total += 3
        return total

    def _prune_context(self, raw_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        system_msgs = [copy.deepcopy(m) for m in raw_history if m["role"] == "system"]
        chat_msgs = [m for m in raw_history if m["role"] != "system"]
        
        # Reserve minimum tokens for chat (200) + completion + safety buffer (50)
        max_system_tokens = self.max_tokens - self.max_completion_tokens - 250
        
        if max_system_tokens <= 0:
            raise ValueError("max_tokens is too small to accommodate completion and basic context.")
        
        system_tokens = self._count_tokens(system_msgs)
        
        # Safely truncate system messages if they exceed the allowed limit
        while system_tokens > max_system_tokens and system_msgs:
            # Find the largest system message to truncate
            largest_idx = -1
            max_len = -1
            for i, msg in enumerate(system_msgs):
                content_len = len(str(msg.get("content", "")))
                if content_len > max_len:
                    max_len = content_len
                    largest_idx = i
                    
            if largest_idx == -1 or max_len == 0:
                break
                
            content = str(system_msgs[largest_idx].get("content", ""))
            tokens = self.encoder.encode(content)
            
            excess = system_tokens - max_system_tokens
            truncate_by = excess + 5  # +5 to handle unicode token boundaries
            
            if len(tokens) > truncate_by:
                system_msgs[largest_idx]["content"] = self.encoder.decode(tokens[:-truncate_by])
            else:
                system_msgs[largest_idx]["content"] = ""
                
            system_tokens = self._count_tokens(system_msgs)

        available_tokens = (
            self.max_tokens 
            - system_tokens 
            - self.max_completion_tokens 
            - 50 
        )
        
        if available_tokens <= 0:
            available_tokens = 0
        
        retained_chat_msgs = []
        current_tokens = 0
        
        for msg in reversed(chat_msgs):
            msg_tokens = self._count_tokens([msg])
            if current_tokens + msg_tokens > available_tokens:
                break
            retained_chat_msgs.append(msg)
            current_tokens += msg_tokens
            
        retained_chat_msgs.reverse()
        
        while retained_chat_msgs and retained_chat_msgs[0]["role"] != "user":
            retained_chat_msgs.pop(0)

        return system_msgs + retained_chat_msgs

    async def stream_response(
            self, 
            raw_history: List[Dict[str, str]]
            ) -> AsyncGenerator[str, None]:
        
        context_payload = self._prune_context(raw_history)

        async for chunk in self.connection.stream_chat(
            self.model, 
            context_payload, 
            self.max_completion_tokens
        ):
            yield chunk
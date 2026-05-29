"""
Service layer for the LLM module.

Handles token counting, context window truncation, and streaming generation.
"""
import copy
import re
from typing import AsyncGenerator, Dict, List

import tiktoken

from src.infra import LLMConnection


class LLMService:
    """Application Service handling domain-level LLM logic: 
    internal token management, context window truncation, and generation wrapper.
    
    Args:
        connection (LLMConnection): The low-level infrastructure connection.
        model (str): The model ID to use for token counting and generation.
        max_tokens (int): The absolute maximum context window size.
        max_completion_tokens (int): Tokens reserved for the generation output.
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
        """Count the number of tokens in a list of chat messages.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries.

        Returns:
            int: The estimated token count including OpenAI message formatting overhead.
        """
        total = 0
        for msg in messages:
            total += 3 
            for key, value in msg.items():
                total += len(self.encoder.encode(str(value)))
        total += 3
        return total

    def _prune_context(self, raw_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Truncate the message history to fit within the context window limits.

        Args:
            raw_history (List[Dict[str, str]]): The full chat history.

        Returns:
            List[Dict[str, str]]: The pruned history safe for LLM consumption.
        """
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

            # Detect the closing salt tag so we can preserve it after truncation.
            # Without this, truncating from the end would strip </system_auth_...>
            # and break the security boundary.
            closing_tag_match = re.search(r'</system_auth_[0-9a-f]+>\s*$', content)
            closing_tag = closing_tag_match.group(0).strip() if closing_tag_match else None
            closing_tag_tokens = len(self.encoder.encode(closing_tag)) if closing_tag else 0

            excess = system_tokens - max_system_tokens
            truncate_by = excess + 5  # +5 to handle unicode token boundaries

            # When a closing salt tag exists, cut extra tokens to make
            # room for re-appending it after truncation.
            effective_cut = truncate_by + closing_tag_tokens if closing_tag else truncate_by

            if len(tokens) > effective_cut:
                truncated = self.encoder.decode(tokens[:-effective_cut])
                if closing_tag:
                    truncated = truncated.rstrip() + "\n" + closing_tag
                system_msgs[largest_idx]["content"] = truncated
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
        """Stream a response from the LLM based on the chat history.

        Args:
            raw_history (List[Dict[str, str]]): The raw chat history before pruning.

        Yields:
            str: Chunks of text as they are generated by the LLM.
        """
        context_payload = self._prune_context(raw_history)

        # Expose the pruned payload for logging/debugging.
        # This is the exact message list sent to the LLM API.
        self.last_context_payload = context_payload

        async for chunk in self.connection.stream_chat(
            self.model, 
            context_payload, 
            self.max_completion_tokens
        ):
            yield chunk
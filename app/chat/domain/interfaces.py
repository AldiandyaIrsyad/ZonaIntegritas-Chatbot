from typing import Protocol, AsyncIterator, List, Optional, Dict, Any
from app.chat.domain.models import Session, Message

class ILLMConnection(Protocol):
    """Protocol for interacting with a Language Model."""
    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        ...
    async def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> str:
        """Generate a complete (non-streaming) chat completion.

        Used for tasks that need the full response as a string before
        proceeding (e.g. HyDE hypothetical document generation).

        Args:
            model: Model identifier.
            messages: Chat messages.
            max_tokens: Maximum tokens for the response.
            temperature: Sampling temperature.

        Returns:
            The full response text.
        """
        ...
    async def close(self) -> None:
        ...

class IChatRepository(Protocol):
    """Protocol for chat persistence operations."""
    async def get_all_sessions(self) -> List[Session]: ...
    async def create_session(self, session_id: str, title: str) -> Session: ...
    async def get_session_by_id(self, session_id: str, load_messages: bool = False) -> Optional[Session]: ...
    async def update_session_title(self, session: Session, new_title: str) -> Session: ...
    async def create_message(
        self,
        session_id: str,
        role: str,
        content: str,
        raw_content: Optional[str] = None,
        context: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Message: ...
    async def delete_session(self, session_id: str) -> bool: ...

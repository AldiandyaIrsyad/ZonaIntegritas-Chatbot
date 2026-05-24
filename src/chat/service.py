import logging
import uuid
from typing import List, Optional

from fastapi.responses import StreamingResponse

from src.chat.repository import ChatRepository
from src.llm.service import LLMService
from src.rag.retrieval import RetrievalService, RetrievedContext

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        repository: ChatRepository,
        llm_service: LLMService,
        retrieval_service: RetrievalService,
    ):
        self.repository = repository
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service

    async def list_sessions(self):
        sessions = await self.repository.get_all_sessions()
        return [{"id": s.id, "title": s.title} for s in sessions]

    async def create_new_session(self):
        session_id = str(uuid.uuid4())
        new_session = await self.repository.create_session(session_id, "New Chat")
        return {"id": new_session.id, "title": new_session.title}

    async def get_session_details(self, session_id: str):
        session = await self.repository.get_session_by_id(session_id, load_messages=True)
        if not session:
            return None
        return {
            "title": session.title,
            "messages": [{"role": m.role, "content": m.content} for m in session.messages]
        }

    async def process_chat_message(self, session_id: str, message_text: str):
        session = await self.repository.get_session_by_id(session_id, load_messages=True)
        
        if not session:
            session = await self.repository.create_session(session_id, message_text[:20] + "...")
            
        if session.title == "New Chat":
            new_title = message_text[:30] + ("..." if len(message_text) > 30 else "")
            await self.repository.update_session_title(session, new_title)
            
        await self.repository.create_message(session_id, "user", message_text)

        raw_history = [{"role": m.role, "content": m.content} for m in session.messages]

        # --- RAG Context Injection ---
        # Retrieve relevant context from the knowledge base before
        # sending the conversation to the LLM.
        contexts = await self._retrieve_rag_context(message_text)
        if contexts:
            context_prompt = self._build_context_prompt(contexts)
            # Insert as the first system message so it's always in context
            raw_history.insert(0, {"role": "system", "content": context_prompt})

        raw_history.append({"role": "user", "content": message_text})

        async def generate():
            response_content = ""
            try:
                async for chunk in self.llm_service.stream_response(raw_history):
                    response_content += chunk
                    yield chunk
            finally:
                if response_content.strip():
                    try:
                        import anyio
                        # Use anyio.CancelScope(shield=True) instead of asyncio.shield
                        # because FastAPI/Starlette manages concurrency via AnyIO.
                        # This guarantees the block executes even if the parent request is cancelled.
                        with anyio.CancelScope(shield=True):
                            await self.repository.create_message(session_id, "assistant", response_content)
                    except Exception as e:
                        logger.error(f"Failed to save partial response: {e}")
                
        return StreamingResponse(generate(), media_type="text/plain")

    async def delete_session(self, session_id: str):
        return await self.repository.delete_session(session_id)

    async def _retrieve_rag_context(
        self, query: str
    ) -> List[RetrievedContext]:
        """Attempt to retrieve RAG context; return empty list on failure.

        RAG failures should not break the chat — the LLM can still
        respond without knowledge base context.
        """
        try:
            return await self.retrieval_service.retrieve_context(query)
        except Exception as e:
            logger.warning(
                "RAG retrieval failed, proceeding without context: %s",
                str(e),
            )
            return []

    @staticmethod
    def _build_context_prompt(contexts: List[RetrievedContext]) -> str:
        """Format retrieved contexts as a system prompt for the LLM.

        Each context block is attributed to its source document for
        transparency and traceability in the LLM's response.
        """
        parts = [
            "You have access to the following reference documents. "
            "Use them to answer the user's question accurately. "
            "If the documents don't contain relevant information, "
            "say so rather than making up an answer."
        ]

        for ctx in contexts:
            parts.append(f"\n---\n[Source: {ctx.source_title}]\n{ctx.text}")

        parts.append("\n---")
        return "\n".join(parts)
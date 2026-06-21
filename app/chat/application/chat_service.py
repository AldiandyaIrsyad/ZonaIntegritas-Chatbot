"""
Ultimate Orchestrator for the Chat Module.

Coordinates the Knowledge Base, IVM (Safety/Relevance), RAM (Response Assessment),
and the LLM inference engine.
"""

import json
import uuid
import structlog
from typing import AsyncGenerator, List, Dict, Any, Optional

from app.chat.domain.interfaces import IChatRepository, ILLMConnection
from app.kb.application.search_service import SearchService
from app.thesis.ivm.service import IVMService, MaliciousPromptException, IrrelevantQueryException
from app.thesis.ram.service import RAMService
from app.thesis.ram.interfaces import RetrievedContext as RAMRetrievedContext

logger = structlog.get_logger(__name__)

class ChatService:
    """Orchestrates the chat request pipeline."""

    def __init__(
        self,
        chat_repo: IChatRepository,
        llm_conn: ILLMConnection,
        search_service: SearchService,
        ivm_service: IVMService,
        ram_service: RAMService,
        model_name: str,
        system_prompt: str,
    ):
        self.chat_repo = chat_repo
        self.llm_conn = llm_conn
        self.search_service = search_service
        self.ivm_service = ivm_service
        self.ram_service = ram_service
        self.model_name = model_name
        self.system_prompt = system_prompt

    async def create_session(self) -> Dict[str, Any]:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        session = await self.chat_repo.create_session(session_id, "New Chat")
        return {"id": session.id, "title": session.title}

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all chat sessions."""
        sessions = await self.chat_repo.get_all_sessions()
        return [{"id": s.id, "title": s.title} for s in sessions]

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details."""
        session = await self.chat_repo.get_session_by_id(session_id, load_messages=True)
        if not session:
            return None
        return {
            "id": session.id,
            "title": session.title,
            "messages": [{"role": m.role, "content": m.content} for m in session.messages],
        }

    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        return await self.chat_repo.delete_session(session_id)

    def _build_prompt(self, user_message: str, contexts: List[RAMRetrievedContext]) -> str:
        if not contexts:
            return user_message
        context_str = "\n\n".join([f"[{i+1}] {ctx.text}" for i, ctx in enumerate(contexts)])
        return f"Context:\n{context_str}\n\nQuestion:\n{user_message}"

    async def process_chat_message(self, session_id: str, message_text: str) -> AsyncGenerator[str, None]:
        """The main generation pipeline: Safety -> Pre-check -> Context -> Generate -> Assess."""
        
        # 1. Initialize or get session
        session = await self.chat_repo.get_session_by_id(session_id, load_messages=True)
        if not session:
            session = await self.chat_repo.create_session(session_id, message_text[:20] + "...")
        elif session.title == "New Chat":
            new_title = message_text[:30] + ("..." if len(message_text) > 30 else "")
            await self.chat_repo.update_session_title(session, new_title)

        # Record user message
        await self.chat_repo.create_message(session_id, "user", message_text, raw_content=message_text)

        try:
            # 2. Safety Check (IVM)
            await self.ivm_service.check_malicious(message_text)

            # 3. Pre-check Relevance (IVM + KB)
            precheck_contexts = await self.search_service.search(message_text, top_k=3, session_id=session_id)
            if not precheck_contexts:
                raise IrrelevantQueryException("No relevant contexts found in the knowledge base.")
            
            context_chunks = [ctx.text for ctx in precheck_contexts]
            await self.ivm_service.check_relevance(message_text, context_chunks)

            # 4. Deep Context Retrieval (KB)
            full_contexts = await self.search_service.search(message_text, top_k=15, session_id=session_id)
            
            # Map KB contexts to RAM contexts
            ram_contexts = [
                RAMRetrievedContext(
                    text=ctx.text,
                    source_title=ctx.source_title,
                    page=ctx.page,
                )
                for ctx in full_contexts
            ]

            # 5. Prompt Building
            final_prompt = self._build_prompt(message_text, ram_contexts)
            
            messages = [{"role": "system", "content": self.system_prompt}]
            # Add history (up to last 5 messages to avoid blowing up context window)
            history = session.messages[-10:] if session else []
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
            
            # Add current
            messages.append({"role": "user", "content": final_prompt})

            # Prepare RAM evaluation
            premise = self.ram_service.build_premise(ram_contexts)
            
            # We buffer the stream by sentence to evaluate each complete sentence
            buffer = ""
            final_output = ""
            
            # 6. Stream and Assess
            stream = self.llm_conn.stream_chat(
                model=self.model_name,
                messages=messages,
                max_tokens=1024
            )
            
            async for chunk in stream:
                buffer += chunk
                # Crude sentence splitting for streaming assessment
                if any(punct in chunk for punct in [". ", "? ", "! ", "\n"]):
                    sentences = buffer.replace("\n", " ").split(". ")
                    if len(sentences) > 1:
                        # Process all but the last incomplete fragment
                        for sentence in sentences[:-1]:
                            sentence = sentence.strip() + "."
                            if len(sentence) > 10: # Only assess meaningful sentences
                                result = await self.ram_service.assess_sentence(sentence, premise, ram_contexts)
                                
                                # Format output
                                out_sentence = sentence
                                if result.label == "contradiction":
                                    out_sentence += " *(contradicts context)*"
                                elif result.label == "neutral":
                                    out_sentence += " *(unverified)*"
                                else:
                                    # Entailment - add citation
                                    if result.source_title:
                                        out_sentence += f" [{result.source_title}]"
                                
                                final_output += out_sentence + " "
                                yield json.dumps({"type": "chunk", "content": out_sentence + " "}) + "\n"
                            else:
                                final_output += sentence + " "
                                yield json.dumps({"type": "chunk", "content": sentence + " "}) + "\n"
                        
                        buffer = sentences[-1]

            # Process any remaining buffer
            if buffer.strip():
                sentence = buffer.strip()
                if len(sentence) > 10:
                    result = await self.ram_service.assess_sentence(sentence, premise, ram_contexts)
                    if result.label == "contradiction":
                        sentence += " *(contradicts context)*"
                    elif result.label == "neutral":
                        sentence += " *(unverified)*"
                    else:
                        if result.source_title:
                            sentence += f" [{result.source_title}]"
                final_output += sentence
                yield json.dumps({"type": "chunk", "content": sentence}) + "\n"

            # Record assistant message
            await self.chat_repo.create_message(session_id, "assistant", final_output, raw_content=final_output)
            yield json.dumps({"type": "done"}) + "\n"

        except MaliciousPromptException:
            err_msg = "Your request was blocked by our safety filters."
            await self.chat_repo.create_message(session_id, "assistant", err_msg)
            yield json.dumps({"type": "error", "content": err_msg}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except IrrelevantQueryException:
            err_msg = "I'm sorry, but I can only answer questions related to the provided knowledge base documents."
            await self.chat_repo.create_message(session_id, "assistant", err_msg)
            yield json.dumps({"type": "error", "content": err_msg}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except Exception as e:
            logger.error("chat.pipeline.failed", error=str(e), exc_info=True)
            yield json.dumps({"type": "error", "content": "An unexpected error occurred."}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

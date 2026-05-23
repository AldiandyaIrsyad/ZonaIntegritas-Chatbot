import uuid
from fastapi.responses import StreamingResponse
from src.chat.repository import ChatRepository
from src.infra.llm_provider import LLM

class ChatService:
    def __init__(self, repository: ChatRepository, llm: LLM):
        self.repository = repository
        self.llm = llm

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
        raw_history.append({"role": "user", "content": message_text})

        async def generate():
            response_content = ""
            async for chunk in self.llm.input(raw_history):
                response_content += chunk
                yield chunk

            await self.repository.create_message(session_id, "assistant", response_content)
                
        return StreamingResponse(generate(), media_type="text/plain")

    async def delete_session(self, session_id: str):
        return await self.repository.delete_session(session_id)
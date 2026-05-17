import asyncio
import uuid
import os
import aiofiles
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from backend.repository import ChatRepository, PDFRepository
from backend.config import async_session

# Services
import services 

class ChatService:
    def __init__(self, repository: ChatRepository, llm: services.LLM):
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
        session = await self.repository.get_session_by_id(session_id)
        
        if not session:
            session = await self.repository.create_session(session_id, message_text[:20] + "...")
            
        if session.title == "New Chat":
            new_title = message_text[:30] + ("..." if len(message_text) > 30 else "")
            await self.repository.update_session_title(session, new_title)
            
        await self.repository.create_message(session_id, "user", message_text)

        async def generate():
            response_content = ""
            # Consumes identical interface regardless of OpenRouter/Ollama routing
            async for chunk in self.llm.input(message_text):
                response_content += chunk
                yield chunk

            await self.repository.create_message(session_id, "assistant", response_content)
                
        return StreamingResponse(generate(), media_type="text/plain")

    async def delete_session(self, session_id: str):
        return await self.repository.delete_session(session_id)

class PDFService:
    def __init__(self, repository: PDFRepository):
        self.repository = repository
        self.upload_dir = "upload"
        os.makedirs(self.upload_dir, exist_ok=True)
        
    async def list_pdfs(self):
        pdfs = await self.repository.get_all_pdfs()
        return [{"id": p.id, "title": p.title, "description": p.description, "pdf_path": p.pdf_path, "active": p.active} for p in pdfs]
        
    async def upload_pdf(self, title: str, description: str, file: UploadFile):
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(self.upload_dir, unique_filename)
        
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
            
        return await self.repository.create_pdf(title, description, file_path)
        
    async def update_pdf_status(self, pdf_id: str, active: bool):
        return await self.repository.update_pdf_active_status(pdf_id, active)
        
    async def delete_pdf(self, pdf_id: str):
        pdf = await self.repository.get_pdf_by_id(pdf_id)
        if pdf and pdf.pdf_path and os.path.exists(pdf.pdf_path):
            os.remove(pdf.pdf_path)
        return await self.repository.delete_pdf(pdf_id)

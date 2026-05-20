import uuid
import base64
import json
import fitz
import os
import aiofiles
from typing import List
from fastapi import UploadFile
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
            
        messages_out = []
        for m in session.messages:
            content = m.content
            attachments = []
            try:
                blocks = json.loads(content)
                if isinstance(blocks, list):
                    text_parts = []
                    for b in blocks:
                        if b["type"] == "text":
                            if not str(b["text"]).startswith("\n\n--- Content from "):
                                text_parts.append(b["text"])
                        elif b["type"] == "pdf_url":
                            attachments.append({"name": b["name"], "url": b["url"]})
                            
                    content = "\n".join(text_parts)
                    image_count = len([b for b in blocks if b["type"] == "image_url"])
                    if image_count > 0:
                        content = f"[Attached {image_count} image(s)]\n" + content
            except (json.JSONDecodeError, TypeError):
                pass
            messages_out.append({"role": m.role, "content": content, "attachments": attachments})
            
        return {
            "title": session.title,
            "messages": messages_out
        }

    async def process_chat_message(self, session_id: str, message_text: str, files: List[UploadFile] = None):
        session = await self.repository.get_session_by_id(session_id, load_messages=True)
        
        if not session:
            session = await self.repository.create_session(session_id, (message_text[:20] if message_text else "File Upload") + "...")
            
        if session.title == "New Chat":
            new_title = (message_text[:30] if message_text else "File Upload") + ("..." if len(message_text) > 30 else "")
            await self.repository.update_session_title(session, new_title)

        content_blocks = []
        if message_text:
            content_blocks.append({"type": "text", "text": message_text})
            
        if files:
            for file in files:
                if not file.filename:
                    continue
                file_bytes = await file.read()
                if file.content_type and file.content_type.startswith("image/"):
                    base64_img = base64.b64encode(file_bytes).decode('utf-8')
                    mime = file.content_type
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{base64_img}"}
                    })
                elif file.content_type == "application/pdf":
                    try:
                        file_extension = os.path.splitext(file.filename)[1]
                        unique_filename = f"{uuid.uuid4()}{file_extension}"
                        file_path = os.path.join("user_upload", unique_filename)
                        
                        async with aiofiles.open(file_path, 'wb') as out_file:
                            await out_file.write(file_bytes)
                            
                        doc = fitz.open(stream=file_bytes, filetype="pdf")
                        pdf_text = ""
                        for page in doc:
                            pdf_text += page.get_text()
                        content_blocks.append({
                            "type": "text",
                            "text": f"\n\n--- Content from {file.filename} ---\n{pdf_text}"
                        })
                        content_blocks.append({
                            "type": "pdf_url",
                            "name": file.filename,
                            "url": f"/user_upload/{unique_filename}"
                        })
                    except Exception as e:
                        print(f"Error parsing PDF: {e}")

        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})

        if len(content_blocks) == 1 and content_blocks[0]["type"] == "text":
            db_content = content_blocks[0]["text"]
        else:
            db_content = json.dumps(content_blocks)

        await self.repository.create_message(session_id, "user", db_content)

        raw_history = [{"role": m.role, "content": m.content} for m in session.messages]
        raw_history.append({"role": "user", "content": db_content})

        async def generate():
            response_content = ""
            async for chunk in self.llm.input(raw_history, max_tokens=4000):
                response_content += chunk
                yield chunk

            await self.repository.create_message(session_id, "assistant", response_content)
                
        return StreamingResponse(generate(), media_type="text/plain")

    async def delete_session(self, session_id: str):
        return await self.repository.delete_session(session_id)
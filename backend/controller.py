from fastapi import Request, UploadFile, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from backend.service import ChatService, PDFService

templates = Jinja2Templates(directory="./templates")

class ChatRequest(BaseModel):
    message: str

class PDFUpdateRequest(BaseModel):
    active: bool

class ChatController:
    def __init__(self, service: ChatService):
        self.service = service

    async def home(self, request: Request):
        return templates.TemplateResponse(
            request=request,
            name="pages/index.html",
            context={"title": "Chat", "current_year": 2026}
        )

    async def get_sessions(self):
        return await self.service.list_sessions()

    async def create_session(self):
        return await self.service.create_new_session()

    async def get_session(self, session_id: str):
        data = await self.service.get_session_details(session_id)
        if not data:
            return {"error": "Session not found"}
        return data

    async def chat_stream(self, session_id: str, req: ChatRequest):
        return await self.service.process_chat_message(session_id, req.message)

    async def delete_session(self, session_id: str):
        success = await self.service.delete_session(session_id)
        if not success:
            return {"error": "Session not found"}
        return {"status": "success"}

class PDFController:
    def __init__(self, service: PDFService):
        self.service = service
        
    async def admin_page(self, request: Request):
        return templates.TemplateResponse(
            request=request,
            name="pages/admin.html",
            context={"title": "Admin Dashboard", "current_year": 2026}
        )
        
    async def get_pdfs(self):
        return await self.service.list_pdfs()
        
    async def upload_pdf(self, title: str = Form(...), description: str = Form(""), file: UploadFile = Form(...)):
        pdf = await self.service.upload_pdf(title, description, file)
        return {"id": pdf.id, "title": pdf.title, "status": "success"}
        
    async def update_pdf_status(self, pdf_id: str, req: PDFUpdateRequest):
        pdf = await self.service.update_pdf_status(pdf_id, req.active)
        if not pdf:
            return {"error": "PDF not found"}
        return {"id": pdf.id, "active": pdf.active, "status": "success"}
        
    async def delete_pdf(self, pdf_id: str):
        success = await self.service.delete_pdf(pdf_id)
        if not success:
            return {"error": "PDF not found"}
        return {"status": "success"}

from fastapi import APIRouter, Depends, Request, UploadFile, Form
from sqlalchemy.ext.asyncio import AsyncSession
from backend.config import get_db
from backend.repository import ChatRepository, PDFRepository
from backend.service import ChatService, PDFService
from backend.controller import ChatController, ChatRequest, PDFController, PDFUpdateRequest

from backend.dependencies import get_chat_controller, get_pdf_controller

router = APIRouter()

@router.get("/")
async def home(request: Request, controller: ChatController = Depends(get_chat_controller)):
    return await controller.home(request)

@router.get("/api/sessions")
async def get_sessions(controller: ChatController = Depends(get_chat_controller)):
    return await controller.get_sessions()

@router.post("/api/sessions")
async def create_session(controller: ChatController = Depends(get_chat_controller)):
    return await controller.create_session()

@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str, controller: ChatController = Depends(get_chat_controller)):
    return await controller.get_session(session_id)

@router.post("/api/sessions/{session_id}/stream")
async def chat_stream(session_id: str, req: ChatRequest, controller: ChatController = Depends(get_chat_controller)):
    return await controller.chat_stream(session_id, req)

@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, controller: ChatController = Depends(get_chat_controller)):
    return await controller.delete_session(session_id)

# Admin Routes
@router.get("/admin/")
async def admin_page(request: Request, controller: PDFController = Depends(get_pdf_controller)):
    return await controller.admin_page(request)

@router.get("/api/admin/pdfs")
async def get_pdfs(controller: PDFController = Depends(get_pdf_controller)):
    return await controller.get_pdfs()

@router.post("/api/admin/pdfs")
async def upload_pdf(title: str = Form(...), description: str = Form(""), file: UploadFile = Form(...), controller: PDFController = Depends(get_pdf_controller)):
    return await controller.upload_pdf(title, description, file)

@router.put("/api/admin/pdfs/{pdf_id}/status")
async def update_pdf_status(pdf_id: str, req: PDFUpdateRequest, controller: PDFController = Depends(get_pdf_controller)):
    return await controller.update_pdf_status(pdf_id, req)

@router.delete("/api/admin/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str, controller: PDFController = Depends(get_pdf_controller)):
    return await controller.delete_pdf(pdf_id)

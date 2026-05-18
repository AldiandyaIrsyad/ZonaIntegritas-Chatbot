# src/knowledge_base/router.py
from fastapi import APIRouter, Request, UploadFile, Form, Depends
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.knowledge_base.service import KnowledgeBase
from src.knowledge_base.dependency import get_pdf_service

router = APIRouter(prefix="/admin", tags=["Knowledge Base"])
templates = Jinja2Templates(directory="templates")

class PDFUpdateRequest(BaseModel):
    active: bool

@router.get("/")
async def admin_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="pages/admin.html",
        context={"title": "Admin Dashboard", "current_year": 2026}
    )

@router.get("/pdfs")
async def get_pdfs(service: KnowledgeBase = Depends(get_pdf_service)):
    return await service.list_pdfs()

@router.post("/pdfs")
async def upload_pdf(
    title: str = Form(...), 
    description: str = Form(""), 
    file: UploadFile = Form(...),
    service: KnowledgeBase = Depends(get_pdf_service)
):
    pdf = await service.upload_pdf(title, description, file)
    return {"id": pdf.id, "title": pdf.title, "status": "success"}

@router.patch("/pdfs/{pdf_id}")
async def update_pdf_status(
    pdf_id: str, 
    req: PDFUpdateRequest,
    service: KnowledgeBase = Depends(get_pdf_service)
):
    pdf = await service.update_pdf_status(pdf_id, req.active)
    if not pdf:
        return {"error": "PDF not found"}
    return {"id": pdf.id, "active": pdf.active, "status": "success"}

@router.delete("/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str, service: KnowledgeBase = Depends(get_pdf_service)):
    success = await service.delete_pdf(pdf_id)
    if not success:
        return {"error": "PDF not found"}
    return {"status": "success"}
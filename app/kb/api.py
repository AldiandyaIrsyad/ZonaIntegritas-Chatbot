"""JSON API endpoints for knowledge base administration."""
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Any
from starlette.responses import JSONResponse

from app.kb.application.kb_service import KBApplicationService
from app.kb.dependency import get_kb_service

router = APIRouter()

class PDFUpdateRequest(BaseModel):
    active: bool

@router.get("/api/admin/pdfs")
async def get_pdfs(service: KBApplicationService = Depends(get_kb_service)) -> Any:
    """Retrieve all uploaded PDF documents with their metadata."""
    return await service.list_pdfs()

@router.post("/api/admin/pdfs", status_code=202)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    title: str = Form(...), 
    description: str = Form(""), 
    file: UploadFile = Form(...), 
    service: KBApplicationService = Depends(get_kb_service)
) -> Any:
    """Upload a new PDF document and trigger async ingestion."""
    pdf = await service.upload_pdf(title, description, file, background_tasks)
    return JSONResponse(
        status_code=202,
        content={
            "id": pdf.id,
            "title": pdf.title,
            "ingestion_status": pdf.ingestion_status,
            "status": "accepted",
        },
    )

@router.put("/api/admin/pdfs/{pdf_id}/status")
async def update_pdf_status(
    pdf_id: str, 
    req: PDFUpdateRequest, 
    service: KBApplicationService = Depends(get_kb_service)
) -> Any:
    """Update the active status of a PDF document."""
    pdf = await service.update_pdf_status(pdf_id, req.active)
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    return {"id": pdf.id, "active": pdf.active, "status": "success"}

@router.delete("/api/admin/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str, service: KBApplicationService = Depends(get_kb_service)) -> Any:
    """Delete a PDF document from the knowledge base."""
    success = await service.delete_pdf(pdf_id)
    if not success:
        raise HTTPException(status_code=404, detail="PDF not found")
    return {"status": "success", "message": "PDF deleted"}

@router.get("/api/admin/pdfs/{pdf_id}/ingestion-status")
async def get_ingestion_status(
    pdf_id: str,
    service: KBApplicationService = Depends(get_kb_service),
) -> Any:
    """Check the ingestion processing status of a PDF document."""
    result = await service.get_ingestion_status(pdf_id)
    if not result:
        raise HTTPException(status_code=404, detail="PDF not found")
    return result

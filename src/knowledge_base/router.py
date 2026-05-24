from fastapi import APIRouter, Request, Depends, HTTPException, UploadFile, Form
from starlette.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.knowledge_base.service import KnowledgeBase
from src.knowledge_base.dependency import get_pdf_service

router = APIRouter()
templates = Jinja2Templates(directory="templates")

class PDFUpdateRequest(BaseModel):
    active: bool

@router.get("/admin/")
async def admin_page(request: Request):
    """Render the admin dashboard for PDF document management.
    
    Displays the admin interface where users can upload, view, edit, and manage
    PDF documents used as knowledge base sources for the LLM.
    
    Args:
        request (Request): HTTP request object.
    
    Returns:
        TemplateResponse: Rendered HTML template for the admin dashboard.
    """
    return templates.TemplateResponse(
        request=request,
        name="pages/admin.html",
        context={"title": "Admin Dashboard", "current_year": 2026}
    )

@router.get("/api/admin/pdfs")
async def get_pdfs(service: KnowledgeBase = Depends(get_pdf_service)):
    """Retrieve all uploaded PDF documents with their metadata.
    
    Fetches a list of all PDFs stored in the knowledge base, including
    active status, upload date, file size, and content summary.
    
    Args:
        service (KnowledgeBase): Dependency-injected PDF service instance.
    
    Returns:
        list: List of dictionaries containing PDF metadata.
    """
    return await service.list_pdfs()

@router.post("/api/admin/pdfs", status_code=202)
async def upload_pdf(
    title: str = Form(...), 
    description: str = Form(""), 
    file: UploadFile = Form(...), 
    service: KnowledgeBase = Depends(get_pdf_service)
):
    """Upload a new PDF document and trigger async ingestion.
    
    Accepts a PDF file, stores it in the system, and enqueues it for
    background processing (parsing, chunking, embedding). Returns
    HTTP 202 Accepted to indicate async processing has started.
    
    Args:
        title (str): Name/title of the PDF document.
        description (str): Optional description or notes about the PDF content.
        file (UploadFile): The PDF file to upload.
        service (KnowledgeBase): Dependency-injected PDF service instance.
    
    Returns:
        JSONResponse: 202 Accepted with PDF metadata and ingestion status.
    """
    pdf = await service.upload_pdf(title, description, file)
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
    service: KnowledgeBase = Depends(get_pdf_service)
):
    """Update the active status of a PDF document.
    
    Allows admins to manually toggle whether a document is actively used in the knowledge base.
    
    Args:
        pdf_id (str): UUID of the PDF document to update.
        req (PDFUpdateRequest): Request body containing the boolean active status.
        service (KnowledgeBase): Dependency-injected PDF service instance.
    
    Returns:
        dict: Updated PDF metadata.
            
    Raises:
        HTTPException: 404 if pdf_id does not exist.
    """
    pdf = await service.update_pdf_status(pdf_id, req.active)
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    return {"id": pdf.id, "active": pdf.active, "status": "success"}

@router.delete("/api/admin/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str, service: KnowledgeBase = Depends(get_pdf_service)):
    """Delete a PDF document from the knowledge base.
    
    Permanently removes a PDF, its associated file, and all vectors
    from both PostgreSQL and Qdrant. This action is irreversible.
    
    Args:
        pdf_id (str): UUID of the PDF document to delete.
        service (KnowledgeBase): Dependency-injected PDF service instance.
    
    Returns:
        dict: Confirmation response.
            
    Raises:
        HTTPException: 404 if pdf_id does not exist.
    """
    success = await service.delete_pdf(pdf_id)
    if not success:
        raise HTTPException(status_code=404, detail="PDF not found")
    return {"status": "success", "message": "PDF deleted"}

@router.get("/api/admin/pdfs/{pdf_id}/ingestion-status")
async def get_ingestion_status(
    pdf_id: str,
    service: KnowledgeBase = Depends(get_pdf_service),
):
    """Check the ingestion processing status of a PDF document.
    
    Returns the current state of the async ingestion pipeline for a
    specific document. Useful for polling from the admin UI.
    
    Args:
        pdf_id (str): UUID of the PDF document.
        service (KnowledgeBase): Dependency-injected PDF service instance.
    
    Returns:
        dict: Document ID, title, and ingestion_status.
    
    Raises:
        HTTPException: 404 if pdf_id does not exist.
    """
    result = await service.get_ingestion_status(pdf_id)
    if not result:
        raise HTTPException(status_code=404, detail="PDF not found")
    return result
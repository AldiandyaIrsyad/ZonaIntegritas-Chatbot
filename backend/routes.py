from fastapi import APIRouter, Depends, Request, UploadFile, Form
from sqlalchemy.ext.asyncio import AsyncSession
from backend.repository import ChatRepository, PDFRepository
from backend.service import ChatService, KnowledgeBase
from backend.controller import ChatController, ChatRequest, PDFController, PDFUpdateRequest

from backend.dependencies import get_chat_controller, get_pdf_controller

router = APIRouter()

@router.get("/")
async def home(request: Request, controller: ChatController = Depends(get_chat_controller)):
    """Render the main chat interface homepage.
    
    This endpoint serves the primary user interface for the chat application.
    It initializes the session and loads the chat UI template with necessary
    assets for real-time messaging.
    
    Args:
        request (Request): The HTTP request object containing client information.
        controller (ChatController): Dependency-injected chat controller instance
            for handling business logic.
    
    Returns:
        TemplateResponse: Rendered HTML template for the chat interface.
        
    Raises:
        HTTPException: 500 if template rendering fails or session initialization
            encounters a critical error.
    """
    return await controller.home(request)

@router.get("/api/sessions")
async def get_sessions(controller: ChatController = Depends(get_chat_controller)):
    """Retrieve all active chat sessions.
    
    Fetches a list of all current chat sessions available to the user,
    including metadata such as session ID, creation timestamp, and last
    activity timestamp. Useful for session history and recovery.
    
    Args:
        controller (ChatController): Dependency-injected chat controller instance.
    
    Returns:
        dict: JSON object containing list of sessions with structure:
            {"sessions": [{"id": str, "created_at": datetime, ...}, ...]}
            
    Raises:
        HTTPException: 500 if database query fails.
    """
    return await controller.get_sessions()

@router.post("/api/sessions")
async def create_session(controller: ChatController = Depends(get_chat_controller)):
    """Create a new chat session.
    
    Initializes a fresh chat session with empty message history. Each session
    is independent and can have its own conversation context and LLM state.
    
    Args:
        controller (ChatController): Dependency-injected chat controller instance.
    
    Returns:
        dict: JSON object containing the new session details:
            {"id": str, "created_at": datetime}
            
    Raises:
        HTTPException: 500 if database write fails.
    """
    return await controller.create_session()

@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str, controller: ChatController = Depends(get_chat_controller)):
    """Retrieve a specific chat session by ID.
    
    Fetches complete session details including all messages, metadata, and
    conversation context for resuming a prior chat.
    
    Args:
        session_id (str): Unique identifier (UUID) of the session to retrieve.
        controller (ChatController): Dependency-injected chat controller instance.
    
    Returns:
        dict: JSON object containing session details:
            {"id": str, "created_at": datetime, "messages": [...]}
            
    Raises:
        HTTPException: 404 if session_id does not exist.
        HTTPException: 500 if database query fails.
    """
    return await controller.get_session(session_id)

@router.post("/api/sessions/{session_id}/stream")
async def chat_stream(session_id: str, req: ChatRequest, controller: ChatController = Depends(get_chat_controller)):
    """Stream a chat message and receive LLM response in real-time.
    
    Sends a user message to the LLM for a given session and streams back
    the response token-by-token. Uses Server-Sent Events (SSE) for real-time
    data transmission. Messages are persisted to the session history.
    
    Args:
        session_id (str): UUID of the session to send the message to.
        req (ChatRequest): Request body containing the user message and optional
            parameters (tone, context, etc.).
        controller (ChatController): Dependency-injected chat controller instance.
    
    Returns:
        StreamingResponse: Server-Sent Events stream of LLM response tokens.
            Each event contains {"type": "data", "message": "token"}.
            Final event: {"type": "end"}.
            
    Raises:
        HTTPException: 404 if session_id does not exist.
        HTTPException: 400 if ChatRequest validation fails (empty message).
        HTTPException: 503 if LLM service is unavailable.
        HTTPException: 500 if database or streaming fails.
    """
    return await controller.chat_stream(session_id, req)

@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, controller: ChatController = Depends(get_chat_controller)):
    """Delete a chat session and all its messages.
    
    Permanently removes a session from the database. This action is irreversible.
    All associated messages and conversation history are also deleted.
    
    Args:
        session_id (str): UUID of the session to delete.
        controller (ChatController): Dependency-injected chat controller instance.
    
    Returns:
        dict: Confirmation response:
            {"status": "success", "message": "Session deleted"}
            
    Raises:
        HTTPException: 404 if session_id does not exist.
        HTTPException: 500 if database delete fails.
    """
    return await controller.delete_session(session_id)

# Admin Routes
@router.get("/admin/")
async def admin_page(request: Request, controller: PDFController = Depends(get_pdf_controller)):
    """Render the admin dashboard for PDF document management.
    
    Displays the admin interface where users can upload, view, edit, and manage
    PDF documents used as knowledge base sources for the LLM. Requires admin
    permissions (authentication may be enforced by middleware).
    
    Args:
        request (Request): HTTP request object.
        controller (PDFController): Dependency-injected PDF controller instance.
    
    Returns:
        TemplateResponse: Rendered HTML template for the admin dashboard.
        
    Raises:
        HTTPException: 403 if user lacks admin permissions.
        HTTPException: 500 if template rendering fails.
    """
    return await controller.admin_page(request)

@router.get("/api/admin/pdfs")
async def get_pdfs(controller: PDFController = Depends(get_pdf_controller)):
    """Retrieve all uploaded PDF documents with their metadata.
    
    Fetches a list of all PDFs stored in the knowledge base, including
    processing status, upload date, file size, and content summary.
    
    Args:
        controller (PDFController): Dependency-injected PDF controller instance.
    
    Returns:
        dict: JSON object containing PDF list:
            {"pdfs": [{"id": str, "title": str, "status": str, ...}, ...]}
            Status values: "pending", "processing", "ready", "failed".
            
    Raises:
        HTTPException: 500 if database query fails.
    """
    return await controller.get_pdfs()

@router.post("/api/admin/pdfs")
async def upload_pdf(title: str = Form(...), description: str = Form(""), file: UploadFile = Form(...), controller: PDFController = Depends(get_pdf_controller)):
    """Upload a new PDF document to the knowledge base.
    
    Accepts a PDF file and stores it in the system. The file undergoes
    processing stages (text extraction, embedding generation, indexing)
    before becoming available for LLM queries. Status can be tracked via
    the /api/admin/pdfs endpoint.
    
    Args:
        title (str): Name/title of the PDF document (required).
        description (str): Optional description or notes about the PDF content.
            Defaults to empty string.
        file (UploadFile): The PDF file to upload (multipart/form-data).
            Maximum size and allowed formats enforced by controller.
        controller (PDFController): Dependency-injected PDF controller instance.
    
    Returns:
        dict: JSON object with new PDF metadata:
            {"id": str, "title": str, "status": "pending", "created_at": datetime}
            
    Raises:
        HTTPException: 400 if file format is invalid (not PDF) or title is empty.
        HTTPException: 413 if file size exceeds maximum allowed.
        HTTPException: 500 if file storage or database write fails.
    """
    return await controller.upload_pdf(title, description, file)

@router.put("/api/admin/pdfs/{pdf_id}/status")
async def update_pdf_status(pdf_id: str, req: PDFUpdateRequest, controller: PDFController = Depends(get_pdf_controller)):
    """Update the processing status of a PDF document.
    
    Allows admins to manually set or override the processing status of a PDF.
    Typically used to retry failed processing or manually mark documents as ready.
    
    Args:
        pdf_id (str): UUID of the PDF document to update.
        req (PDFUpdateRequest): Request body containing the new status.
            Expected status values: "pending", "processing", "ready", "failed".
        controller (PDFController): Dependency-injected PDF controller instance.
    
    Returns:
        dict: Updated PDF metadata:
            {"id": str, "title": str, "status": str, "updated_at": datetime}
            
    Raises:
        HTTPException: 404 if pdf_id does not exist.
        HTTPException: 400 if status value in request is invalid.
        HTTPException: 500 if database update fails.
    """
    return await controller.update_pdf_status(pdf_id, req)

@router.delete("/api/admin/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str, controller: PDFController = Depends(get_pdf_controller)):
    """Delete a PDF document from the knowledge base.
    
    Permanently removes a PDF and all associated embeddings/indexes from the system.
    This action is irreversible. Chat sessions may lose context if they relied on
    this PDF's content.
    
    Args:
        pdf_id (str): UUID of the PDF document to delete.
        controller (PDFController): Dependency-injected PDF controller instance.
    
    Returns:
        dict: Confirmation response:
            {"status": "success", "message": "PDF deleted"}
            
    Raises:
        HTTPException: 404 if pdf_id does not exist.
        HTTPException: 500 if database delete or file storage cleanup fails.
    """
    return await controller.delete_pdf(pdf_id)

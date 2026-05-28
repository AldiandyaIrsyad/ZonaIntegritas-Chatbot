from fastapi import APIRouter, Request, Depends, HTTPException, UploadFile, File
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.chat.service import ChatService
from src.chat.dependency import get_chat_service

router = APIRouter()
templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message: str

@router.post("/api/sessions/{session_id}/upload")
async def upload_session_file(session_id: str, file: UploadFile = File(...), service: ChatService = Depends(get_chat_service)):
    """Upload a file to a specific chat session.
    
    Args:
        session_id (str): UUID of the session.
        file (UploadFile): The file to upload.
        service (ChatService): Dependency-injected chat service instance.
    
    Returns:
        dict: Uploaded document metadata.
    """
    return await service.upload_pdf(session_id, file)

@router.get("/")
async def home(request: Request):
    """Render the main chat interface homepage.
    
    This endpoint serves the primary user interface for the chat application.
    It initializes the session and loads the chat UI template with necessary
    assets for real-time messaging.
    
    Args:
        request (Request): The HTTP request object containing client information.
    
    Returns:
        TemplateResponse: Rendered HTML template for the chat interface.
    """
    return templates.TemplateResponse(
        request=request,
        name="pages/index.html",
        context={"title": "Chat", "current_year": 2026}
    )

@router.get("/api/sessions")
async def get_sessions(service: ChatService = Depends(get_chat_service)):
    """Retrieve all active chat sessions.
    
    Fetches a list of all current chat sessions available to the user,
    including metadata such as session ID, creation timestamp, and last
    activity timestamp. Useful for session history and recovery.
    
    Args:
        service (ChatService): Dependency-injected chat service instance.
    
    Returns:
        list: List of dictionaries containing session metadata.
    """
    return await service.list_sessions()

@router.post("/api/sessions")
async def create_session(service: ChatService = Depends(get_chat_service)):
    """Create a new chat session.
    
    Initializes a fresh chat session with empty message history. Each session
    is independent and can have its own conversation context and LLM state.
    
    Args:
        service (ChatService): Dependency-injected chat service instance.
    
    Returns:
        dict: JSON object containing the new session details:
            {"id": str, "title": str}
    """
    return await service.create_new_session()

@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str, service: ChatService = Depends(get_chat_service)):
    """Retrieve a specific chat session by ID.
    
    Fetches complete session details including all messages, metadata, and
    conversation context for resuming a prior chat.
    
    Args:
        session_id (str): Unique identifier (UUID) of the session to retrieve.
        service (ChatService): Dependency-injected chat service instance.
    
    Returns:
        dict: JSON object containing session details.
            
    Raises:
        HTTPException: 404 if session_id does not exist.
    """
    data = await service.get_session_details(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return data

@router.post("/api/sessions/{session_id}/stream")
async def chat_stream(session_id: str, req: ChatRequest, service: ChatService = Depends(get_chat_service)):
    """Stream a chat message and receive LLM response in real-time.
    
    Sends a user message to the LLM for a given session and streams back
    the response token-by-token. Uses Server-Sent Events (SSE) for real-time
    data transmission. Messages are persisted to the session history.
    
    Args:
        session_id (str): UUID of the session to send the message to.
        req (ChatRequest): Request body containing the user message.
        service (ChatService): Dependency-injected chat service instance.
    
    Returns:
        StreamingResponse: Text stream of LLM response tokens.
    """
    return await service.process_chat_message(session_id, req.message)

@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, service: ChatService = Depends(get_chat_service)):
    """Delete a chat session and all its messages.
    
    Permanently removes a session from the database. This action is irreversible.
    All associated messages and conversation history are also deleted.
    
    Args:
        session_id (str): UUID of the session to delete.
        service (ChatService): Dependency-injected chat service instance.
    
    Returns:
        dict: Confirmation response.
            
    Raises:
        HTTPException: 404 if session_id does not exist.
    """
    success = await service.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "message": "Session deleted"}
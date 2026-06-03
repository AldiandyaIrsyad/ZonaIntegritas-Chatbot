"""API routing for the chat module.

Defines endpoints for session management, PDF uploads, and real-time streaming chat.
"""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import Any

from app.chat.dependency import get_chat_service
from app.chat.service import ChatService
from app.orchestrator.service import ChatOrchestrator
from app.orchestrator.dependency import get_chat_orchestrator

router = APIRouter()

class ChatRequest(BaseModel):
    """Payload for incoming chat messages."""
    message: str


@router.post("/sessions/{session_id}/upload")
async def upload_session_file(
    session_id: str, 
    file: UploadFile = File(...), 
    service: ChatService = Depends(get_chat_service),
    orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator)
) -> dict[str, Any]:
    return await service.upload_pdf(session_id, file, orchestrator)


@router.get("/sessions")
async def get_sessions(service: ChatService = Depends(get_chat_service)) -> list[dict[str, Any]]:
    return await service.list_sessions()


@router.post("/sessions")
async def create_session(service: ChatService = Depends(get_chat_service)) -> dict[str, Any]:
    return await service.create_new_session()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, service: ChatService = Depends(get_chat_service)) -> dict[str, Any]:
    data = await service.get_session_details(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return data


@router.post("/sessions/{session_id}/stream")
async def chat_stream(
    session_id: str, 
    req: ChatRequest, 
    service: ChatService = Depends(get_chat_service),
    orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator)
) -> StreamingResponse:
    generator = service.process_chat_message(session_id, req.message, orchestrator)
    return StreamingResponse(generator, media_type="text/plain")


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str, 
    service: ChatService = Depends(get_chat_service),
    orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator)
) -> dict[str, Any]:
    success = await service.delete_session(session_id, orchestrator)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "message": "Session deleted"}

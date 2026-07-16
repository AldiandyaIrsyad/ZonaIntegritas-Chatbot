"""
JSON API endpoints for the Chat module.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any

from app.chat.application.chat_service import ChatService
from app.chat.dependency import get_chat_service

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/api/chat/sessions", status_code=201)
async def create_session(service: ChatService = Depends(get_chat_service)) -> Any:
    """Create a new chat session."""
    return await service.create_session()

@router.get("/api/chat/sessions")
async def list_sessions(service: ChatService = Depends(get_chat_service)) -> Any:
    """List all chat sessions."""
    return await service.list_sessions()

@router.get("/api/chat/sessions/{session_id}")
async def get_session(session_id: str, service: ChatService = Depends(get_chat_service)) -> Any:
    """Get chat session details."""
    session = await service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.delete("/api/chat/sessions/{session_id}")
async def delete_session(session_id: str, service: ChatService = Depends(get_chat_service)) -> Any:
    """Delete a chat session."""
    success = await service.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success"}

@router.post("/api/chat/sessions/{session_id}/stream")
async def chat_stream(
    session_id: str,
    request: ChatRequest,
    skip_guardrails: bool = Query(default=False, description="Skip IVM + RAM (baseline mode)"),
    service: ChatService = Depends(get_chat_service),
) -> Any:
    """Stream a chat response from the LLM, passing through IVM and RAM.

    When ``skip_guardrails`` is true, the IVM safety/relevance checks and the
    RAM per-sentence assessment are bypassed (baseline mode for Experiment 4).
    Retrieval still runs so the LLM has context.
    """
    return StreamingResponse(
        service.process_chat_message(session_id, request.message, skip_guardrails=skip_guardrails),
        media_type="application/x-ndjson"
    )

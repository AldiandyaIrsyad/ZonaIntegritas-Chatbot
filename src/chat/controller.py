from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.chat.service import ChatService
from src.chat.dependency import get_chat_service

router = APIRouter(tags=["Chat"])
templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message: str

@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="pages/index.html",
        context={"title": "Chat", "current_year": 2026}
    )

@router.get("/sessions")
async def get_sessions(service: ChatService = Depends(get_chat_service)):
    return await service.list_sessions()

@router.post("/sessions")
async def create_session(service: ChatService = Depends(get_chat_service)):
    return await service.create_new_session()

@router.get("/sessions/{session_id}")
async def get_session(session_id: str, service: ChatService = Depends(get_chat_service)):
    data = await service.get_session_details(session_id)
    if not data:
        return {"error": "Session not found"}
    return data

@router.post("/sessions/{session_id}/chat")
async def chat_stream(session_id: str, req: ChatRequest, service: ChatService = Depends(get_chat_service)):
    return await service.process_chat_message(session_id, req.message)

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, service: ChatService = Depends(get_chat_service)):
    success = await service.delete_session(session_id)
    if not success:
        return {"error": "Session not found"}
    return {"status": "success"}
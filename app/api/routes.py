"""API routes for the application."""

from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    """Root endpoint serving the main dashboard.
    
    Args:
        request (Request): The incoming HTTP request.
        
    Returns:
        HTMLResponse: The rendered index page.
    """
    context: Dict[str, Any] = {"request": request, "title": "Chatbot Dashboard"}
    return templates.TemplateResponse(request, "pages/index.html", context)  # type: ignore[return-value]

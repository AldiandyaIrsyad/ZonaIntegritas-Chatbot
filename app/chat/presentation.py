"""Presentation routing for the chat module.

Defines endpoints for HTML template rendering.
"""
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()

# Resolve absolute path to app/templates
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/")
async def home(request: Request) -> Any:
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

"""HTML presentation endpoints for RAM administration."""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/ram/")
async def ram_dashboard(request: Request):
    """Render a placeholder RAM dashboard."""
    logger.info("Accessing RAM dashboard")
    return templates.TemplateResponse(
        request=request,
        name="pages/admin.html",  # Using existing template as placeholder
        context={"title": "RAM Dashboard", "current_year": 2026}
    )

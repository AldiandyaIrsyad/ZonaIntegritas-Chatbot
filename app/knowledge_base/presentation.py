"""HTML presentation endpoints for knowledge base administration."""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

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

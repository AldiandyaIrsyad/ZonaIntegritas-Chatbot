from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory="app/templates")

from typing import Any

@router.get("/")
async def get_chat_page(request: Request) -> Any:
    return templates.TemplateResponse(
        request=request, name="pages/index.html"
    )

@router.get("/admin/")
async def get_admin_page(request: Request) -> Any:
    return templates.TemplateResponse(
        request=request, name="pages/admin.html"
    )

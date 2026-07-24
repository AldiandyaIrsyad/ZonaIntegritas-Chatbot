"""Server-rendered HTML page routes (Jinja2 templates).

These routes serve the browser UI and are excluded from the OpenAPI schema
(``include_in_schema=False``) so they don't clutter ``/docs``. They are
mounted last in ``app/main.py`` so the ``/api/*`` JSON routers match first.

Pages:
    - ``/``       — chat UI for end users.
    - ``/admin/`` — KB admin UI for uploading/managing documents.
    - ``/demo/``  — demo/evaluation UI.
"""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory="app/templates")

from typing import Any


@router.get("/")
async def get_chat_page(request: Request) -> Any:
    """Render the end-user chat page (``pages/index.html``)."""
    return templates.TemplateResponse(
        request=request, name="pages/index.html"
    )


@router.get("/admin/")
async def get_admin_page(request: Request) -> Any:
    """Render the KB admin page (``pages/admin.html``)."""
    return templates.TemplateResponse(
        request=request, name="pages/admin.html"
    )


@router.get("/demo/")
async def get_demo_page(request: Request) -> Any:
    """Render the demo/evaluation page (``pages/demo.html``)."""
    return templates.TemplateResponse(
        request=request, name="pages/demo.html"
    )

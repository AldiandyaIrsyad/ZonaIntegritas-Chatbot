from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from backend import engine, Base, router

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(
    title="Chat Application with PDF Knowledge Base",
    description="An intelligent chat system that leverages Large Language Models (LLMs) and PDF document management to provide context-aware responses. Users can upload PDF documents which are processed and indexed for semantic search, enabling the LLM to reference relevant content in its responses.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "Support Team",
        "url": "https://example.com/support",
    },
    license_info={
        "name": "MIT",
    },
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

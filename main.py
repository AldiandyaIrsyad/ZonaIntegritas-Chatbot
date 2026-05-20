from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from dotenv import load_dotenv

from backend import engine, Base, router

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(router)

app.mount("/user_upload", StaticFiles(directory="user_upload"), name="user_upload")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

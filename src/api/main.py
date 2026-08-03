from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routers import chat, collections, documents, health
from src.store.db import init_db


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Chat Engine", version="1.0.0")
    init_db()
    app.include_router(health.router)
    app.include_router(collections.router)
    app.include_router(documents.router)
    app.include_router(chat.router)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})

    return app

app = create_app()

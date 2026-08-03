from fastapi import FastAPI

from src.api.routers import chat, collections, documents, health
from src.store.db import init_db


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Chat Engine", version="1.0.0")
    init_db()
    app.include_router(health.router)
    app.include_router(collections.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    return app


app = create_app()

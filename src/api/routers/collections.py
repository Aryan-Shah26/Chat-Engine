from fastapi import APIRouter, HTTPException

from src.store import db
from src.store.schemas import CollectionCreate, CollectionOut, DocumentOut

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionOut)
def create_collection(payload: CollectionCreate):
    return db.create_collection(payload.name)


@router.get("", response_model=list[CollectionOut])
def list_collections():
    return db.list_collections()


@router.get("/{collection_id}/documents", response_model=list[DocumentOut])
def list_documents(collection_id: str):
    if not db.get_collection(collection_id):
        raise HTTPException(404, "Collection not found")
    return db.list_documents(collection_id)

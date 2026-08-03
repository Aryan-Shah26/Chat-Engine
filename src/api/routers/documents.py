import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.deps import get_collection_index, invalidate_collection_cache
from src.ingestion.pipeline import ingest_document
from src.store import db
from src.store.schemas import DocumentOut

router = APIRouter(prefix="/collections", tags=["documents"])

ALLOWED_SUFFIXES = {".pdf", ".html", ".txt", ".md"}


@router.post("/{collection_id}/documents", response_model=DocumentOut)
async def upload_document(collection_id: str, file: UploadFile = File(...)):
    if not db.get_collection(collection_id):
        raise HTTPException(404, "Collection not found")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Write into a temp dir under the ORIGINAL filename (not a random temp
    # name) so `source` metadata used in citations is the real filename.
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / file.filename
    tmp_path.write_bytes(await file.read())

    try:
        existing_docs = db.list_documents(collection_id)
        vectorstore = None
        if existing_docs:
            _, _, vectorstore = get_collection_index(collection_id)
        chunk_count, _, _, _ = ingest_document(tmp_path, collection_id, vectorstore=vectorstore)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    invalidate_collection_cache(collection_id)
    return db.register_document(collection_id, file.filename, chunk_count)

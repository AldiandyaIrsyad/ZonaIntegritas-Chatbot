"""
Dependency injection for the KB domain.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.db import get_db_session
from app.kb.config import get_qdrant_settings, get_infinity_settings, get_unstructured_settings, get_storage_settings
from app.kb.infra import PostgresKBRepository, QdrantStore, UnstructuredClient, InfinityEmbeddings
from app.kb.application.ingest_worker import IngestWorker
from app.kb.application.search_service import SearchService
from app.kb.application.kb_service import KBApplicationService

async def get_kb_repo(db: AsyncSession = Depends(get_db_session)) -> PostgresKBRepository:
    return PostgresKBRepository(db)

def get_vector_store() -> QdrantStore:
    config = get_qdrant_settings()
    return QdrantStore(host=config.host, port=config.port, collection_name=config.collection_name)

def get_document_parser() -> UnstructuredClient:
    config = get_unstructured_settings()
    return UnstructuredClient(base_url=config.base_url)

def get_text_embedder() -> InfinityEmbeddings:
    config = get_infinity_settings()
    return InfinityEmbeddings(base_url=config.base_url, model=config.embedding_model)

async def get_ingest_worker(
    db: AsyncSession = Depends(get_db_session),
    repo: PostgresKBRepository = Depends(get_kb_repo),
    parser: UnstructuredClient = Depends(get_document_parser),
    embedder: InfinityEmbeddings = Depends(get_text_embedder),
    vstore: QdrantStore = Depends(get_vector_store)
) -> IngestWorker:
    return IngestWorker(db, parser, embedder, vstore, repo)

async def get_kb_service(
    repo: PostgresKBRepository = Depends(get_kb_repo),
    vstore: QdrantStore = Depends(get_vector_store),
    worker: IngestWorker = Depends(get_ingest_worker)
) -> KBApplicationService:
    config = get_storage_settings()
    return KBApplicationService(
        kb_repo=repo,
        vector_store=vstore,
        ingest_worker=worker,
        upload_dir=config.upload_dir
    )

async def get_search_service(
    repo: PostgresKBRepository = Depends(get_kb_repo),
    vstore: QdrantStore = Depends(get_vector_store),
    embedder: InfinityEmbeddings = Depends(get_text_embedder)
) -> SearchService:
    return SearchService(text_embedder=embedder, vector_store=vstore, kb_repo=repo)

"""
Dependency injection for the KB domain.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.shared.db import get_db_session
from app.kb.config import get_qdrant_settings, get_infinity_settings, get_unstructured_settings, get_storage_settings, get_vlm_settings
from app.kb.infra import PostgresKBRepository, QdrantStore, UnstructuredClient, InfinityEmbeddings, InfinityReranker
from app.kb.infra.vlm_client import create_vlm_enricher
from app.thesis.chunking.interfaces import IVLMEnricher
from app.kb.application.ingest_worker import IngestWorker
from app.kb.application.search_service import SearchService
from app.kb.application.kb_service import KBApplicationService
from app.thesis.ivm.service import IVMService

async def get_kb_repo(db: AsyncSession = Depends(get_db_session)) -> PostgresKBRepository:
    return PostgresKBRepository(db)

def get_vector_store() -> QdrantStore:
    config = get_qdrant_settings()
    return QdrantStore(host=config.host, port=config.port, collection_name=config.collection_name)

def get_document_parser() -> UnstructuredClient:
    config = get_unstructured_settings()
    return UnstructuredClient(base_url=config.base_url, extract_images=config.extract_images)

def get_text_embedder() -> InfinityEmbeddings:
    config = get_infinity_settings()
    return InfinityEmbeddings(base_url=config.base_url, model=config.embedding_model)

def get_reranker() -> Optional[InfinityReranker]:
    config = get_infinity_settings()
    if not config.reranker_enabled:
        return None
    return InfinityReranker(base_url=config.base_url, model=config.reranker_model)

def get_ivm_service() -> Optional[IVMService]:
    """Returns None by default — IVM document validation is optional.

    The IVMService requires an LLMJudge (cloud LLM) and a safety model. Wiring
    these here would violate the dependency rule that ``kb/`` must not import
    ``chat/infra``. When document-relevance validation is desired, override
    this provider at the application composition layer (``app/main.py``) to
    inject an ``IVMService`` built from ``chat/infra`` adapters.
    """
    return None

def get_vlm_enricher() -> Optional[IVLMEnricher]:
    """Create a VLM enricher based on VLMSettings.

    Returns None if VLM is disabled. The enricher is created per-request
    because the fallback mode needs the PDF path (set later by the
    IngestWorker). For cloud/local modes, a new client is created each
    time — this is acceptable since ingestion is not high-frequency.

    Returns:
        IVLMEnricher instance or None.
    """
    settings = get_vlm_settings()
    return create_vlm_enricher(settings)

async def get_ingest_worker(
    db: AsyncSession = Depends(get_db_session),
    repo: PostgresKBRepository = Depends(get_kb_repo),
    parser: UnstructuredClient = Depends(get_document_parser),
    embedder: InfinityEmbeddings = Depends(get_text_embedder),
    vstore: QdrantStore = Depends(get_vector_store),
    ivm_service: Optional[IVMService] = Depends(get_ivm_service),
    vlm_enricher: Optional[IVLMEnricher] = Depends(get_vlm_enricher),
) -> IngestWorker:
    storage_config = get_storage_settings()
    return IngestWorker(
        db=db,
        document_parser=parser,
        text_embedder=embedder,
        vector_store=vstore,
        kb_repo=repo,
        ivm_service=ivm_service,
        vlm_enricher=vlm_enricher,
        image_dir=storage_config.image_dir,
    )

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
    embedder: InfinityEmbeddings = Depends(get_text_embedder),
    reranker: Optional[InfinityReranker] = Depends(get_reranker),
) -> SearchService:
    return SearchService(text_embedder=embedder, vector_store=vstore, kb_repo=repo, reranker=reranker)

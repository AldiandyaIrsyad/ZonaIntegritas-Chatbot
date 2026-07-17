"""
Dependency injection for the KB domain.
"""

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.shared.db import get_db_session
from app.kb.config import (
    get_qdrant_settings, get_infinity_settings, get_unstructured_settings,
    get_storage_settings, get_vlm_settings, get_bge_m3_settings,
)
from app.kb.infra import PostgresKBRepository, QdrantStore, UnstructuredClient, BGEM3Embeddings, InfinityReranker
from app.thesis.vlm import FallbackVLMClient, IVLMEnricher, OllamaVLMClient, OpenRouterVLMClient
from app.kb.application.ingest_worker import IngestWorker
from app.kb.application.search_service import SearchService
from app.kb.application.kb_service import KBApplicationService
from app.kb.domain.interfaces import IQueryExpander

async def get_kb_repo(db: AsyncSession = Depends(get_db_session)) -> PostgresKBRepository:
    return PostgresKBRepository(db)

# These four are process-lifetime singletons (not per-request factories):
# each wraps an httpx.AsyncClient (or, for BGEM3Embeddings, an in-process
# model) that's expensive to open/load and was previously being recreated
# on every dependency resolution with nothing ever closing it — a real
# connection/resource leak. app/main.py's lifespan closes these on shutdown.

@lru_cache
def get_vector_store() -> QdrantStore:
    config = get_qdrant_settings()
    return QdrantStore(
        host=config.host,
        port=config.port,
        collection_name=config.collection_name,
    )

@lru_cache
def get_document_parser() -> UnstructuredClient:
    config = get_unstructured_settings()
    return UnstructuredClient(
        base_url=config.base_url,
        extract_images=config.extract_images,
        api_key=config.api_key,
    )

@lru_cache
def get_text_embedder() -> BGEM3Embeddings:
    config = get_bge_m3_settings()
    return BGEM3Embeddings(
        model_name=config.model,
        device=config.device,
        use_fp16=config.use_fp16,
        batch_size=config.batch_size,
    )

@lru_cache
def get_reranker() -> Optional[InfinityReranker]:
    config = get_infinity_settings()
    if not config.reranker_enabled:
        return None
    return InfinityReranker(base_url=config.base_url, model=config.reranker_model)

def get_query_expander() -> Optional[IQueryExpander]:
    """Returns None by default — HyDE query expansion is optional.

    The HyDEExpander requires an LLM connection (``chat/infra``). Wiring it
    here would violate the dependency rule that ``kb/`` must not import
    ``chat/infra``. When HyDE is desired, override this provider at the
    application composition layer (``chat/dependency.py``) to inject a
    ``HyDEExpander`` built from ``chat/infra``.
    """
    return None

def get_vlm_enricher() -> Optional[IVLMEnricher]:
    """Create a VLM enricher based on VLMSettings.

    This is the composition root for the VLM adapter: it reads
    ``VLMSettings`` and selects the concrete ``app.thesis.vlm`` client
    based on ``settings.mode``. Returns None if VLM is disabled. The
    enricher is created per-request because the fallback mode needs the
    PDF path (set later by the IngestWorker). For cloud/local modes, a
    new client is created each time — this is acceptable since ingestion
    is not high-frequency.

    Returns:
        IVLMEnricher instance or None.
    """
    settings = get_vlm_settings()
    if not settings.enabled:
        return None

    mode = settings.mode.lower().strip()

    if mode == "cloud":
        if not settings.cloud_api_key:
            return FallbackVLMClient()
        return OpenRouterVLMClient(
            api_key=settings.cloud_api_key,
            model=settings.cloud_model,
            base_url=settings.cloud_base_url,
            timeout=settings.timeout,
        )

    if mode == "local":
        return OllamaVLMClient(
            base_url=settings.local_base_url,
            model=settings.local_model,
            timeout=settings.timeout,
        )

    # Default: fallback mode
    return FallbackVLMClient()

async def get_ingest_worker(
    db: AsyncSession = Depends(get_db_session),
    repo: PostgresKBRepository = Depends(get_kb_repo),
    parser: UnstructuredClient = Depends(get_document_parser),
    embedder: BGEM3Embeddings = Depends(get_text_embedder),
    vstore: QdrantStore = Depends(get_vector_store),
    vlm_enricher: Optional[IVLMEnricher] = Depends(get_vlm_enricher),
) -> IngestWorker:
    storage_config = get_storage_settings()
    vlm_settings = get_vlm_settings()
    return IngestWorker(
        db=db,
        document_parser=parser,
        text_embedder=embedder,
        vector_store=vstore,
        kb_repo=repo,
        vlm_enricher=vlm_enricher,
        image_dir=storage_config.image_dir,
        page_image_ratio_threshold=vlm_settings.page_image_ratio_threshold,
        page_garbage_ratio_threshold=vlm_settings.page_garbage_ratio_threshold,
        image_description_prompt=vlm_settings.image_description_prompt,
        page_extraction_prompt=vlm_settings.page_extraction_prompt,
    )

async def get_kb_service(
    repo: PostgresKBRepository = Depends(get_kb_repo),
    vstore: QdrantStore = Depends(get_vector_store),
    worker: IngestWorker = Depends(get_ingest_worker),
) -> KBApplicationService:
    config = get_storage_settings()
    return KBApplicationService(
        kb_repo=repo,
        vector_store=vstore,
        ingest_worker=worker,
        upload_dir=config.upload_dir,
    )

async def get_search_service(
    repo: PostgresKBRepository = Depends(get_kb_repo),
    vstore: QdrantStore = Depends(get_vector_store),
    embedder: BGEM3Embeddings = Depends(get_text_embedder),
    reranker: Optional[InfinityReranker] = Depends(get_reranker),
    query_expander: Optional[IQueryExpander] = Depends(get_query_expander),
) -> SearchService:
    return SearchService(
        text_embedder=embedder,
        vector_store=vstore,
        kb_repo=repo,
        reranker=reranker,
        query_expander=query_expander,
    )

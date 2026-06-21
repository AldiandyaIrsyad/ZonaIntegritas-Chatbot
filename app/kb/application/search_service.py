"""
Search workflow for the KB domain.
"""

from typing import Dict, List, Optional
import structlog

from app.kb.domain.interfaces import ITextEmbedder, IVectorStore, IKBRepository
from app.kb.domain.models import PDFDocument, RetrievedContext

logger = structlog.get_logger(__name__)

class SearchService:
    """Orchestrates document search combining Qdrant vectors and Postgres full-text chunks."""

    def __init__(
        self,
        text_embedder: ITextEmbedder,
        vector_store: IVectorStore,
        kb_repo: IKBRepository,
    ):
        self.text_embedder = text_embedder
        self.vector_store = vector_store
        self.kb_repo = kb_repo

    async def search(self, query: str, top_k: int = 15, session_id: Optional[str] = None) -> List[RetrievedContext]:
        """Search the Knowledge Base for relevant contexts."""
        if not query.strip():
            return []

        logger.info("kb.search.started", query_length=len(query), top_k=top_k)

        # 1. Embed query
        embeddings = await self.text_embedder.embed_texts([query])
        if not embeddings:
            return []
        query_emb = embeddings[0]

        # 2. Vector search in Qdrant
        search_results = await self.vector_store.hybrid_search(
            dense_vector=query_emb.dense,
            sparse_indices=query_emb.sparse_indices,
            sparse_values=query_emb.sparse_values,
            top_k=top_k,
            session_id=session_id,
        )

        if not search_results:
            return []

        # 3. Retrieve parent chunks from Postgres
        # We need unique parent chunk IDs
        parent_chunk_ids = list(set([r.parent_chunk_id for r in search_results]))
        parent_chunks = await self.kb_repo.get_parent_chunks_by_ids(parent_chunk_ids)
        parent_chunk_map = {pc.id: pc for pc in parent_chunks}

        # 4. Collect unique doc_ids and fetch titles
        doc_ids = list(set(r.doc_id for r in search_results))
        pdf_docs: List[PDFDocument] = await self.kb_repo.get_pdfs_by_ids(doc_ids)
        doc_title_map: Dict[str, str] = {doc.id: doc.title or doc.id for doc in pdf_docs}

        # 5. Assemble RetrievedContext
        contexts: List[RetrievedContext] = []
        for result in search_results:
            parent = parent_chunk_map.get(result.parent_chunk_id)
            if parent:
                contexts.append(
                    RetrievedContext(
                        chunk_id=result.chunk_id,
                        parent_chunk_id=result.parent_chunk_id,
                        doc_id=result.doc_id,
                        text=parent.text,
                        score=result.score,
                        source_title=doc_title_map.get(result.doc_id, result.doc_id),
                        page=parent.page,
                        breadcrumbs=parent.breadcrumbs or [],
                    )
                )

        logger.info("kb.search.completed", results_count=len(contexts))
        return contexts

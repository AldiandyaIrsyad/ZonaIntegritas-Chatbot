import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import Column, String

from app.infra.db import Base

from sqlalchemy.orm import relationship

class PDFDocument(Base):
    __tablename__ = "pdf_documents"
    id = Column(String, primary_key=True)
    parent_chunks = relationship("ParentChunk", back_populates="document")

from app.rag.ingestion import IngestionService
from app.rag.retrieval import RetrievalService
from app.core.interfaces.ai import EmbeddingResult, RankedResult
from app.core.interfaces.infra import ParsedElement, SearchResult
from app.rag.model import IngestionTask, ParentChunk

@pytest.fixture
def mock_db():
    return AsyncMock()

@pytest.fixture
def mock_parser():
    parser = AsyncMock()
    # Mock return values for parse_pdf
    parser.parse_pdf.return_value = [
        ParsedElement(element_type="Title", text="Document Title", metadata={"page_number": 1}),
        ParsedElement(element_type="NarrativeText", text="This is some body text that will be chunked.", metadata={"page_number": 1}),
    ]
    return parser

@pytest.fixture
def mock_embedder():
    embedder = AsyncMock()
    embedder.embed_texts.return_value = [
        EmbeddingResult(dense=[0.1]*1024, sparse_indices=[1, 2, 3], sparse_values=[0.5, 0.3, 0.2])
    ]
    return embedder

@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    store.hybrid_search.return_value = [
        SearchResult(chunk_id="child_1", parent_chunk_id="parent_1", doc_id="doc_1", score=0.9)
    ]
    return store

@pytest.fixture
def mock_reranker():
    reranker = AsyncMock()
    reranker.rerank.return_value = [
        RankedResult(index=0, text="This is some body text that will be chunked.", score=0.85)
    ]
    return reranker

@pytest.fixture
def mock_repo():
    repo = AsyncMock()
    repo.create_ingestion_task.return_value = IngestionTask(id="task_1", doc_id="doc_1", status="pending")
    # Return a mocked parent chunk for retrieval testing
    mock_parent = ParentChunk(id="parent_1", doc_id="doc_1", text="This is some body text that will be chunked.", chunk_index=0, page=1)
    mock_parent.document = MagicMock()
    mock_parent.document.title = "Mock PDF Document"
    repo.get_parent_chunks_by_ids.return_value = [mock_parent]
    return repo


@pytest.mark.asyncio
@patch("app.rag.ingestion.RAGRepository")
async def test_ingest_document(mock_repo_class, mock_db, mock_parser, mock_embedder, mock_vector_store, mock_repo):
    """Test the document ingestion pipeline."""
    mock_repo_class.return_value = mock_repo
    
    # Mocking the internal document fetch
    mock_pdf_doc = MagicMock()
    mock_pdf_doc.pdf_path = "/path/to/doc.pdf"
    mock_pdf_doc.title = "Test PDF"
    
    # We need to mock _get_document because it uses DB execute natively
    service = IngestionService(mock_db, mock_parser, mock_embedder, mock_vector_store)
    service._get_document = AsyncMock(return_value=mock_pdf_doc)
    service._update_document_status = AsyncMock()
    
    await service.ingest_document("doc_1")
    
    # Assertions
    mock_parser.parse_pdf.assert_called_once_with("/path/to/doc.pdf")
    mock_embedder.embed_texts.assert_called_once()
    mock_vector_store.upsert_chunks.assert_called_once()
    mock_repo.update_ingestion_task.assert_called_with("task_1", "completed")


@pytest.mark.asyncio
@patch("app.rag.retrieval.RAGRepository")
async def test_retrieve_context(mock_repo_class, mock_db, mock_embedder, mock_vector_store, mock_reranker, mock_repo):
    """Test the retrieval pipeline."""
    mock_repo_class.return_value = mock_repo
    
    service = RetrievalService(mock_db, mock_embedder, mock_vector_store, mock_reranker)
    
    contexts = await service.retrieve_context("What is the document about?")
    
    # Assertions
    assert len(contexts) == 1
    assert contexts[0].text == "This is some body text that will be chunked."
    assert contexts[0].doc_id == "doc_1"
    assert contexts[0].parent_chunk_id == "parent_1"
    
    mock_embedder.embed_texts.assert_called_once_with(["What is the document about?"])
    mock_vector_store.hybrid_search.assert_called_once()
    mock_repo.get_parent_chunks_by_ids.assert_called_once_with(["parent_1"])
    mock_reranker.rerank.assert_called_once()

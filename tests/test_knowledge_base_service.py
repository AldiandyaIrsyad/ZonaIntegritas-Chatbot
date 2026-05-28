from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import UploadFile

from src.knowledge_base import KnowledgeBase


@pytest.fixture
def mock_repository():
    return AsyncMock()

@pytest.fixture
def mock_storage():
    return AsyncMock()

@pytest.fixture
def mock_vector_store():
    return AsyncMock()

@pytest.fixture
def mock_ingestion_service():
    return AsyncMock()

@pytest.fixture
def kb_service(mock_repository, mock_storage, mock_vector_store, mock_ingestion_service):
    return KnowledgeBase(
        repository=mock_repository,
        storage=mock_storage,
        vector_store=mock_vector_store,
        ingestion_service=mock_ingestion_service,
    )

@pytest.mark.asyncio
async def test_list_pdfs(kb_service, mock_repository):
    mock_pdf = MagicMock()
    mock_pdf.id = "1"
    mock_pdf.title = "Test PDF"
    mock_pdf.description = "Test Desc"
    mock_pdf.pdf_path = "/path/to/pdf"
    mock_pdf.active = True
    mock_pdf.ingestion_status = "completed"
    mock_repository.get_all_pdfs.return_value = [mock_pdf]

    result = await kb_service.list_pdfs()
    assert len(result) == 1
    assert result[0] == {
        "id": "1", 
        "title": "Test PDF", 
        "description": "Test Desc", 
        "pdf_path": "/path/to/pdf", 
        "active": True,
        "ingestion_status": "completed",
    }
    mock_repository.get_all_pdfs.assert_called_once()

@pytest.mark.asyncio
async def test_upload_pdf_success(kb_service, mock_repository, mock_storage, mock_ingestion_service):
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.pdf"
    mock_file.content_type = "application/pdf"
    
    mock_storage.save_file.return_value = "/mock/path.pdf"
    
    mock_pdf = MagicMock()
    mock_pdf.id = "1"
    mock_pdf.title = "Test PDF"
    mock_repository.create_pdf.return_value = mock_pdf

    mock_background_tasks = MagicMock()

    result = await kb_service.upload_pdf("Test PDF", "Test Desc", mock_file, mock_background_tasks)
    
    assert result == mock_pdf
    mock_storage.save_file.assert_called_once_with(mock_file, ".pdf")
    mock_repository.create_pdf.assert_called_once_with("Test PDF", "Test Desc", "/mock/path.pdf")
    # Verify ingestion task was enqueued
    mock_background_tasks.add_task.assert_called_once()

@pytest.mark.asyncio
async def test_upload_pdf_invalid_extension(kb_service, mock_repository, mock_storage):
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.txt"
    mock_file.content_type = "application/pdf"
    mock_background_tasks = MagicMock()
    
    with pytest.raises(ValueError, match="Only PDF files are allowed"):
        await kb_service.upload_pdf("Test", "Desc", mock_file, mock_background_tasks)
        
    mock_storage.save_file.assert_not_called()
    mock_repository.create_pdf.assert_not_called()

@pytest.mark.asyncio
async def test_upload_pdf_invalid_content_type(kb_service, mock_repository, mock_storage):
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.pdf"
    mock_file.content_type = "text/plain"
    mock_background_tasks = MagicMock()
    
    with pytest.raises(ValueError, match="Only PDF files are allowed"):
        await kb_service.upload_pdf("Test", "Desc", mock_file, mock_background_tasks)
        
    mock_storage.save_file.assert_not_called()

@pytest.mark.asyncio
async def test_update_pdf_status(kb_service, mock_repository, mock_vector_store):
    mock_pdf = MagicMock()
    mock_pdf.id = "1"
    mock_repository.update_pdf_active_status.return_value = mock_pdf
    
    result = await kb_service.update_pdf_status("1", True)
    
    assert result == mock_pdf
    mock_repository.update_pdf_active_status.assert_called_once_with("1", True)
    # Verify Qdrant state sync
    mock_vector_store.update_payload.assert_called_once_with(
        doc_id="1", payload={"is_active": True}
    )

@pytest.mark.asyncio
async def test_delete_pdf(kb_service, mock_repository, mock_storage, mock_vector_store):
    mock_pdf = MagicMock()
    mock_pdf.pdf_path = "/path/to/pdf"
    mock_repository.get_pdf_by_id.return_value = mock_pdf
    mock_repository.delete_pdf.return_value = True

    result = await kb_service.delete_pdf("1")
    
    assert result is True
    mock_repository.get_pdf_by_id.assert_called_once_with("1")
    mock_storage.delete_file.assert_called_once_with("/path/to/pdf")
    mock_repository.delete_pdf.assert_called_once_with("1")
    # Verify Qdrant cleanup
    mock_vector_store.delete_by_doc_id.assert_called_once_with("1")

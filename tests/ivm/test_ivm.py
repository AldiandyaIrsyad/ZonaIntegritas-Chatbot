import sys
import unittest.mock as mock
import pytest
from fastapi import HTTPException

# Patch src so that app/ivm can import from src.infra without modifying the code
import app
sys.modules['src'] = app

from app.ivm.service import IVMService

@pytest.fixture
def mock_prompt_guard():
    pg = mock.AsyncMock()
    result = mock.MagicMock()
    result.is_safe = True
    result.message = "Safe"
    pg.check_prompt.return_value = result
    return pg

@pytest.fixture
def mock_embedding_provider():
    ep = mock.AsyncMock()
    ep_result = mock.MagicMock()
    ep_result.dense = [0.1, 0.2]
    ep_result.sparse_indices = [1, 2]
    ep_result.sparse_values = [0.5, 0.5]
    ep.embed_texts.return_value = [ep_result]
    return ep

@pytest.fixture
def mock_vector_store():
    vs = mock.AsyncMock()
    search_result = mock.MagicMock()
    search_result.score = 0.9
    vs.hybrid_search.return_value = [search_result]
    return vs

@pytest.fixture
def ivm_service(mock_prompt_guard, mock_embedding_provider, mock_vector_store):
    return IVMService(
        prompt_guard=mock_prompt_guard,
        security_threshold=0.8,
        similarity_threshold=0.7,
        embedding_provider=mock_embedding_provider,
        vector_store=mock_vector_store,
    )

@pytest.mark.asyncio
async def test_validate_prompt_safe_and_relevant(ivm_service):
    await ivm_service.validate_prompt("This is a safe and relevant prompt.")
    # Should not raise exception
    assert True

@pytest.mark.asyncio
async def test_validate_prompt_malicious(ivm_service, mock_prompt_guard):
    result = mock.MagicMock()
    result.is_safe = False
    result.message = "Malicious"
    mock_prompt_guard.check_prompt.return_value = result
    with pytest.raises(HTTPException) as exc:
        await ivm_service.validate_prompt("This is malicious.")
    assert exc.value.status_code == 400
    assert "Malicious prompt" in exc.value.detail

@pytest.mark.asyncio
async def test_validate_prompt_irrelevant(ivm_service, mock_vector_store):
    irrelevant_result = mock.MagicMock()
    irrelevant_result.score = 0.5  # Below 0.7
    mock_vector_store.hybrid_search.return_value = [irrelevant_result]
    
    with pytest.raises(HTTPException) as exc:
        await ivm_service.validate_prompt("This is irrelevant.")
    assert exc.value.status_code == 400
    assert "not relevant" in exc.value.detail

@pytest.mark.asyncio
async def test_validate_document_relevance_relevant(ivm_service):
    emb = mock.MagicMock()
    emb.dense = [0.1]
    emb.sparse_indices = [1]
    emb.sparse_values = [0.5]
    
    await ivm_service.validate_document_relevance([emb])
    # Should not raise
    assert True

@pytest.mark.asyncio
async def test_validate_document_relevance_irrelevant(ivm_service, mock_vector_store):
    irrelevant_result = mock.MagicMock()
    irrelevant_result.score = 0.5  # Below 0.7
    mock_vector_store.hybrid_search.return_value = [irrelevant_result]
    
    emb = mock.MagicMock()
    emb.dense = [0.1]
    emb.sparse_indices = [1]
    emb.sparse_values = [0.5]
    
    with pytest.raises(HTTPException) as exc:
        await ivm_service.validate_document_relevance([emb])
    assert exc.value.status_code == 400
    assert "not relevant" in exc.value.detail

"""Tests for reranker infrastructure module."""

import httpx
import pytest
import respx

from app.infra.reranker import Reranker

@pytest.fixture
def reranker():
    """Fixture providing a Reranker instance."""
    return Reranker(
        base_url="http://infinity:7997",
        model="BAAI/bge-reranker-v2-m3"
    )

@respx.mock
async def test_rerank_success(reranker):
    """Test successful reranking and sorting."""
    # Server returns results, usually sorted, but we test the sorting logic
    respx.post("http://infinity:7997/rerank").respond(
        json={
            "results": [
                {"index": 1, "relevance_score": 0.5},
                {"index": 0, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.1},
            ]
        }
    )
    
    docs = ["doc0", "doc1", "doc2"]
    results = await reranker.rerank("query", docs, top_k=2)
    
    assert len(results) == 2
    # Ensure they are sorted by score descending
    assert results[0].index == 0
    assert results[0].score == 0.9
    assert results[0].text == "doc0"
    
    assert results[1].index == 1
    assert results[1].score == 0.5
    assert results[1].text == "doc1"

async def test_rerank_empty_documents(reranker):
    """Test reranking with empty documents list."""
    results = await reranker.rerank("query", [], top_k=3)
    assert results == []

async def test_rerank_zero_top_k(reranker):
    """Test reranking with zero top_k."""
    results = await reranker.rerank("query", ["doc"], top_k=0)
    assert results == []

@respx.mock
async def test_rerank_invalid_items(reranker):
    """Test filtering out invalid items."""
    respx.post("http://infinity:7997/rerank").respond(
        json={
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 5, "relevance_score": 0.5},  # Invalid index (out of bounds)
                {"index": -1, "relevance_score": 0.4}, # Invalid index (negative)
                {"index": 1, "relevance_score": "high"}, # Invalid score type
                {"missing": "index"}, # Missing index/score
            ]
        }
    )
    
    docs = ["doc0", "doc1"]
    results = await reranker.rerank("query", docs, top_k=5)
    
    assert len(results) == 1
    assert results[0].index == 0

@respx.mock
async def test_rerank_http_error(reranker):
    """Test handling of HTTP errors gracefully."""
    respx.post("http://infinity:7997/rerank").respond(status_code=500)
    
    results = await reranker.rerank("query", ["doc"], top_k=3)
    assert results == []

@respx.mock
async def test_rerank_network_error(reranker):
    """Test handling of network exceptions gracefully."""
    respx.post("http://infinity:7997/rerank").mock(side_effect=httpx.ConnectError("Connection refused"))
    
    results = await reranker.rerank("query", ["doc"], top_k=3)
    assert results == []

async def test_close(reranker):
    """Test closing the client."""
    await reranker.close()
    assert reranker._client.is_closed
